"""Conditional distributional reconstruction teacher for a matched ES/MSE test.

The law is conditional on observed N/face/calendar. Relaxed one-hot/mask mark
coordinates are scored in embedded Euclidean space, not as a native categorical
likelihood. Learned codes may still collapse or encode metadata through length.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers.spacing import project_clr
from .base import FutureTargetEncoder


DISTRIBUTIONAL_TEACHER_SPEC = {
    'updates_per_arm': 5000, 'batch_size': 32, 'learning_rate': 3e-4,
    'optimizer': 'AdamW', 'seed': 9, 'code_dim': 64, 'width': 128,
    'encoder_layers': 2, 'decoder_layers': 4, 'heads': 4,
    'noise': 'independent unit Gaussian global and local, independently per pair member',
    'objectives': ['energy', 'mse'], 'energy_exponent': 1,
    'paired_control': 'same architecture/init/data/noise/update count; objective only differs'}


class DistributionTargetEncoder(FutureTargetEncoder):
    def __init__(self, token_dim: int, attrs_dim: int, observation_end: float,
                 code_dim: int = 64, width: int = 128, layers: int = 2,
                 heads: int = 4, decoder_layers: int = 4):
        super().__init__(token_dim, attrs_dim, observation_end, code_dim, width, layers, heads)
        if decoder_layers < 1:
            raise ValueError('Positive stochastic decoder depth required')
        del self.count_head
        del self.face
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, decoder_layers, enable_nested_tensor=False)
        self.global_noise_input = nn.Linear(code_dim, width)
        self.local_noise_input = nn.Linear(token_dim, width)
        self.face_input = nn.Linear(2, width)

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        valid, active, events, count, _, position = self._layout(batch)
        tokens = batch['tokens']
        if tokens.shape != (*valid.shape, self.token_dim) or not bool(torch.isfinite(tokens[valid]).all()):
            raise ValueError('Finite native teacher inputs required')
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        inputs = torch.cat([tokens, active[..., None].to(tokens), events[..., None].to(tokens)], -1)
        # No attributes/calendar/count scalar enters the encoder. Rank/padding/
        # face still reveal length and are not claimed to remove N information.
        hidden = self.input(inputs)+self.position(position)
        cls = self.cls.expand(len(tokens), -1, -1)
        cls = cls+(count == 0).to(tokens)[:, None, None]*self.empty[None, None]
        all_valid = torch.cat([torch.ones(len(tokens), 1, dtype=torch.bool, device=tokens.device), valid], -1)
        result = self.encoder(torch.cat([cls, hidden], 1), src_key_padding_mask=~all_valid)
        return (self.to_code(result[:, 0])-self.code_mean)/self.code_std

    def sample_reconstruction(self, z: Tensor, batch_layout: dict[str, Tensor],
                              global_noise: Tensor | None = None,
                              local_noise: Tensor | None = None) -> Tensor:
        """Draw one path using layout only; never read clean tokens."""
        valid, active, events, _, condition, position = self._layout(batch_layout)
        if z.shape != (len(valid), self.code_dim):
            raise ValueError('One code per reconstruction path required')
        if global_noise is None:
            global_noise = torch.randn_like(z)
        if local_noise is None:
            local_noise = z.new_empty(*valid.shape, self.token_dim).normal_()
        if global_noise.shape != z.shape or local_noise.shape != (*valid.shape, self.token_dim):
            raise ValueError('Global/local noise dimensions disagree')
        if not bool(torch.isfinite(global_noise).all()) or not bool(torch.isfinite(local_noise).all()):
            raise ValueError('Finite reconstruction noise required')
        raw_code = z*self.code_std+self.code_mean
        flags = torch.stack([active, events], -1).to(z)
        hidden = (self.position(position)+self.condition(condition)[:, None]
                  +self.from_code(raw_code)[:, None]+self.global_noise_input(global_noise)[:, None]
                  +self.local_noise_input(local_noise)+self.face_input(flags))
        hidden = self.decoder(hidden, src_key_padding_mask=~valid)
        raw = self.coordinates(hidden)
        timing = project_clr(raw[..., 0], active)
        marks = torch.where(events[..., None], raw[..., 1:], torch.zeros_like(raw[..., 1:]))
        return torch.cat([timing[..., None], marks], -1)

    def reconstruct(self, z: Tensor, batch_layout: dict[str, Tensor]) -> dict[str, Tensor]:
        raise RuntimeError('Distributional teacher requires sample_reconstruction; no deterministic reconstruction API')

    def draw_noise_pair(self, batch_layout: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        valid = batch_layout['valid']
        template = self.code_mean
        # Every member receives its own global AND local noise. Expanding a
        # single global draw across the pair would not estimate an energy score.
        global_pair = template.new_empty(2, len(valid), self.code_dim).normal_()
        local_pair = template.new_empty(2, *valid.shape, self.token_dim).normal_()
        return global_pair, local_pair

    def loss(self, batch: dict[str, Tensor], objective: str = 'energy',
             global_noise_pair: Tensor | None = None,
             local_noise_pair: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        if objective not in ('energy', 'mse'):
            raise ValueError('Use the predeclared energy or mse objective')
        valid, active, events, count, _, _ = self._layout(batch)
        if (global_noise_pair is None) != (local_noise_pair is None):
            raise ValueError('Supply both complete noise pairs, or draw both independently')
        if global_noise_pair is None:
            global_noise_pair, local_noise_pair = self.draw_noise_pair(batch)
        assert local_noise_pair is not None
        if global_noise_pair.shape != (2, len(valid), self.code_dim) or local_noise_pair.shape != (2, *valid.shape, self.token_dim):
            raise ValueError('Exactly two complete noise draws per example required')
        z = self.encode(batch)
        first = self.sample_reconstruction(z, batch, global_noise_pair[0], local_noise_pair[0])
        second = self.sample_reconstruction(z, batch, global_noise_pair[1], local_noise_pair[1])
        target = torch.where(valid[..., None], batch['tokens'], torch.zeros_like(batch['tokens']))
        dof = (active.sum(-1)-1).clamp_min(0)+count*(self.token_dim-1)
        denominator = dof.clamp_min(1).to(z)
        def difference(left, right):
            error = left-right
            time = project_clr(error[..., 0], active)
            marks = torch.where(events[..., None], error[..., 1:], torch.zeros_like(error[..., 1:]))
            return torch.cat([time, marks.flatten(1)], -1)
        error_one, error_two = difference(first, target), difference(second, target)
        pair = difference(first, second)
        # vector_norm has a finite zero subgradient; sqrt(sum(square)) need not.
        norm_one = torch.linalg.vector_norm(error_one, dim=-1)
        norm_two = torch.linalg.vector_norm(error_two, dim=-1)
        norm_pair = torch.linalg.vector_norm(pair, dim=-1)
        energy = .5*(norm_one+norm_two-norm_pair)/denominator.sqrt()
        mse = .5*(error_one.square().sum(-1)+error_two.square().sum(-1))/denominator
        terms = energy if objective == 'energy' else mse
        loss = terms.mean()
        return loss, {'teacher_loss': float(loss.detach()),
                      'teacher_energy': float(energy.detach().mean()),
                      'teacher_mse': float(mse.detach().mean()),
                      'sample_pair_distance': float((norm_pair/denominator.sqrt()).detach().mean()),
                      'zero_dof_paths': float((dof == 0).sum()),
                      'teacher_code_variance': float(z.detach().var(0, unbiased=False).mean())}
