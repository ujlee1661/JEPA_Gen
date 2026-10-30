"""Frozen learned future teacher; never a generation-decoder condition.

The AE uses continuous reconstruction surrogates for all standardized mark
coordinates, including one-hot/mask coordinates. It does not claim a categorical
observation likelihood or a raw-path ELBO. True face projection is teacher-only
reconstruction layout, not an oracle face available to a generator.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .count import discretized_count_log_mass
from .spacing import face_log_mass, project_clr


TEACHER_PREDECLARED = {'updates': 5000, 'batch_size': 32, 'learning_rate': 3e-4,
                       'optimizer': 'AdamW', 'seed': 9, 'code_dim': 64,
                       'width': 128, 'layers': 2, 'heads': 4,
                       'student_target_noise_std': .03}


class FutureTeacher(nn.Module):
    def __init__(self, token_dim: int, attrs_dim: int, observation_end: float,
                 code_dim: int = 64, width: int = 128, layers: int = 2,
                 heads: int = 4, count_components: int = 4):
        super().__init__()
        if min(token_dim, code_dim, width, layers, heads, count_components) < 1 or attrs_dim < 0:
            raise ValueError('Positive teacher dimensions required')
        if width % heads or observation_end <= 0:
            raise ValueError('Invalid attention width or observation end')
        self.token_dim, self.attrs_dim, self.code_dim = token_dim, attrs_dim, code_dim
        self.count_components = count_components
        self.input = nn.Linear(token_dim+2, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Linear(attrs_dim+4, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.empty = nn.Parameter(torch.zeros(width))
        def transformer():
            layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.encoder = transformer()
        self.to_code = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, code_dim))
        self.from_code = nn.Linear(code_dim, width)
        self.decoder = transformer()
        self.coordinates = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_dim))
        self.face = nn.Linear(width, 1)
        self.count_head = nn.Sequential(nn.Linear(code_dim, width), nn.SiLU(),
                                        nn.Linear(width, 3*count_components))
        self.register_buffer('observation_end', torch.tensor(float(observation_end)))
        self.register_buffer('code_mean', torch.zeros(code_dim))
        self.register_buffer('code_std', torch.ones(code_dim))

    def _layout(self, batch: dict[str, Tensor]):
        valid, active, events = batch['valid'].bool(), batch['active'].bool(), batch['event_valid'].bool()
        count, attrs = batch['count'], batch['attrs']
        if valid.ndim != 2 or valid.shape[1] < 1 or active.shape != valid.shape or events.shape != valid.shape:
            raise ValueError('Teacher requires N+1 token layouts including terminal slack')
        if count.shape != (len(valid),) or attrs.shape != (len(valid), self.attrs_dim):
            raise ValueError('Teacher metadata dimensions disagree')
        if bool((count < 0).any()) or bool((count != count.round()).any()):
            raise ValueError('Nonnegative integer event counts required')
        positions = torch.arange(valid.shape[1], device=valid.device)[None]
        if not torch.equal(valid, positions <= count[:, None]) or not torch.equal(events, positions < count[:, None]):
            raise ValueError('Contiguous N events and one terminal token required')
        anchor, delta = batch['anchor'].to(attrs), batch['delta'].to(attrs)
        if anchor.shape != count.shape or delta.shape != count.shape:
            raise ValueError('One calendar per path required')
        if not bool(torch.isfinite(attrs).all()) or not bool(torch.isfinite(anchor).all()) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Finite calendar and attributes required')
        if bool((anchor < 0).any()) or bool((delta < 0).any()):
            raise ValueError('Nonnegative calendar required')
        if bool((active & ~valid).any()) or bool((active.any(-1) != (delta > 0)).any()):
            raise ValueError('Positive windows need nonempty faces; zero windows need empty faces')
        end = self.observation_end.to(attrs)
        if bool((anchor+delta > end+8*torch.finfo(attrs.dtype).eps*end).any()):
            raise ValueError('Teacher query exceeds observation window')
        calendar = torch.stack([anchor.log1p()/end.log1p(), delta.log1p()/end.log1p(), delta/end], -1)
        condition = torch.cat([attrs, calendar, count.to(attrs).log1p()[:, None]], -1)
        rank = positions.to(attrs)/(count.to(attrs)[:, None]).clamp_min(1)
        position = torch.stack([rank, rank.sin(), rank.cos()], -1)
        return valid, active, events, count, condition, position

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        valid, active, events, count, condition, position = self._layout(batch)
        tokens = batch['tokens']
        if tokens.shape != (*valid.shape, self.token_dim) or not bool(torch.isfinite(tokens[valid]).all()):
            raise ValueError('Finite native teacher tokens required')
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        inputs = torch.cat([tokens, active[..., None].to(tokens), events[..., None].to(tokens)], -1)
        hidden = self.input(inputs)+self.position(position)+self.condition(condition)[:, None]
        cls = self.cls.expand(len(tokens), -1, -1)+self.condition(condition)[:, None]
        cls = cls+(count == 0).to(tokens)[:, None, None]*self.empty[None, None]
        all_valid = torch.cat([torch.ones(len(tokens), 1, dtype=torch.bool, device=tokens.device), valid], -1)
        result = self.encoder(torch.cat([cls, hidden], 1), src_key_padding_mask=~all_valid)
        raw = self.to_code(result[:, 0])
        return (raw-self.code_mean)/self.code_std

    def reconstruct(self, z: Tensor, batch_layout: dict[str, Tensor]) -> dict[str, Tensor]:
        """Only layout/metadata are read: no clean token/encoder skip path."""
        valid, active, events, count, condition, position = self._layout(batch_layout)
        if z.shape != (len(valid), self.code_dim):
            raise ValueError('One teacher code per path required')
        raw_z = z*self.code_std+self.code_mean
        queries = self.position(position)+self.condition(condition)[:, None]+self.from_code(raw_z)[:, None]
        hidden = self.decoder(queries, src_key_padding_mask=~valid)
        raw = self.coordinates(hidden)
        timing = project_clr(raw[..., 0], active)
        marks = torch.where(events[..., None], raw[..., 1:], torch.zeros_like(raw[..., 1:]))
        coords = torch.cat([timing[..., None], marks], -1)
        params = self.count_head(raw_z).reshape(len(valid), self.count_components, 3)
        return {'tokens': coords, 'face_logits': self.face(hidden).squeeze(-1),
                'count_logits': params[..., 0], 'count_mean': params[..., 1],
                'count_scale': F.softplus(params[..., 2])+.03}

    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        valid, active, events, count, _, _ = self._layout(batch)
        z = self.encode(batch)
        pred = self.reconstruct(z, batch)
        target = torch.where(valid[..., None], batch['tokens'], torch.zeros_like(batch['tokens']))
        error = pred['tokens']-target
        time_error = project_clr(error[..., 0], active).square().sum(-1)
        mark_error = torch.where(events[..., None], error[..., 1:].square(), torch.zeros_like(error[..., 1:])).sum((1, 2))
        dof = (active.sum(-1)-1).clamp_min(0)+count*(self.token_dim-1)
        mse = (time_error+mark_error)/dof.clamp_min(1)
        positive = batch['delta'] > 0
        face_terms = z.new_zeros(len(z))
        if bool(positive.any()):
            log_mass = face_log_mass(pred['face_logits'][positive], active[positive], valid[positive])
            face_terms = face_terms.index_copy(0, positive.nonzero().flatten(), (-log_mass/valid[positive].sum(-1)).to(z))
        count_log = discretized_count_log_mass(count, pred['count_mean'], pred['count_scale'])
        count_nll = -torch.logsumexp(F.log_softmax(pred['count_logits'], -1)+count_log, -1)
        loss = (mse+face_terms+count_nll).mean()
        return loss, {'teacher_loss': float(loss.detach()), 'teacher_mse': float(mse.detach().mean()),
                      'teacher_face_nll_per_coordinate': float(face_terms.detach().mean()),
                      'teacher_count_nll': float(count_nll.detach().mean()),
                      'teacher_code_variance': float(z.detach().var(0, unbiased=False).mean())}

    @torch.no_grad()
    def set_code_calibration(self, training_codes: Tensor) -> None:
        """Fit once on TRAIN codes encoded before calibration, then freeze.

        Whitening is numerical calibration, not evidence against collapse.
        """
        if training_codes.ndim != 2 or training_codes.shape[1] != self.code_dim or len(training_codes) < 2:
            raise ValueError('At least two training codes required')
        if not bool(torch.isfinite(training_codes).all()):
            raise ValueError('Finite calibration codes required')
        raw = training_codes.to(self.code_mean)*self.code_std+self.code_mean
        self.code_mean.copy_(raw.mean(0))
        self.code_std.copy_(raw.std(0, unbiased=False).clamp_min(1e-6))

    def freeze(self) -> 'FutureTeacher':
        self.requires_grad_(False)
        self.eval()
        return self
