"""A9: A7 predictive dynamics with an additional feature-target branch."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .a7_model import A7Generator


class A9Generator(A7Generator):
    """Feature prediction receives only history/noisy prefix, never teacher codes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden, width, code = self.hidden_dim, self.token_input.out_features, self.code_dim
        self.feature_prediction = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2*code))
        self.feature_residual = nn.Sequential(nn.Linear(4*code, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.feature_residual[-1].weight)
        nn.init.zeros_(self.feature_residual[-1].bias)

    def forward_velocity(self, noisy: Tensor, steps: Tensor, batch: dict) -> dict[str, Tensor]:
        self._validate(batch, noisy)
        if steps.shape != (len(noisy),) or bool((steps < 0).any() or (steps >= self.diffusion_steps).any()):
            raise ValueError('One valid diffusion step per window required')
        attrs, count = batch['attrs'], batch['count']
        calendar = self._calendar(attrs, batch['anchor'], batch['delta'])
        projected = self.project_tokens(noisy, batch['active'], batch['event_valid'])
        flags = torch.stack([batch['active'], batch['event_valid']], -1).to(noisy)
        inputs = torch.cat([projected, flags], -1)
        history = self.encode_history(attrs, batch.get('past_tokens'), batch.get('past_valid'))
        initial = self._initial(attrs, calendar, steps, count, history)
        states, _ = self.prefix_gru(self.prefix_input(inputs), initial[None])
        prefix = torch.cat([initial[:, None], states[:, :-1]], 1)
        mean, scale, features = self._distribution(prefix)
        feature_mean, raw_scale = self.feature_prediction(prefix).split(self.code_dim, -1)
        feature_scale = F.softplus(raw_scale)+.03
        extra = self.feature_residual(torch.cat([features, feature_mean, feature_scale], -1))
        outer = self.outer_condition(torch.cat([attrs, calendar, count.to(attrs).log1p()[:, None]], -1))
        outer = outer+self.noise_condition(self._noise(steps, noisy.dtype))
        hidden = (self.token_input(inputs)+self.position(self._position(count, noisy.shape[1], noisy.dtype))
                  +self.predictive_condition(features)+extra+outer[:, None])
        hidden = self.transformer(hidden, src_key_padding_mask=~batch['valid'])
        velocity = self.project_tokens(self.output(hidden), batch['active'], batch['event_valid'])
        return {'velocity':velocity, 'prefix_mean':mean, 'prefix_scale':scale, 'prefix_state':prefix,
                'feature_mean':feature_mean, 'feature_scale':feature_scale}

    def loss(self, batch: dict, teacher_codes: Tensor | None = None,
             bos_teacher_codes: Tensor | None = None, aux_weight: float = 0., *,
             feature_teacher_codes: Tensor | None = None, feature_aux_weight: float = 0.,
             noise: Tensor | None = None, steps: Tensor | None = None):
        if not math.isfinite(feature_aux_weight) or feature_aux_weight < 0:
            raise ValueError('Feature auxiliary weight must be finite and nonnegative')
        if feature_aux_weight and feature_teacher_codes is None:
            raise ValueError('Feature auxiliary target required')
        if steps is None:
            steps = torch.randint(self.diffusion_steps, (len(batch['tokens']),), device=batch['tokens'].device)
        if noise is None:
            noise = torch.randn_like(batch['tokens'])
        total, stats = super().loss(batch, teacher_codes, bos_teacher_codes, aux_weight, noise=noise, steps=steps)
        feature_aux = total.new_zeros(())
        if feature_aux_weight:
            clean = self.project_tokens(batch['tokens'], batch['active'], batch['event_valid'])
            projected_noise = self.project_tokens(noise, batch['active'], batch['event_valid'])
            alpha = self.alpha_bar[steps][:, None, None]
            noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*projected_noise
            result = self.forward_velocity(noisy, steps, batch)
            terms = self._normal_terms(feature_teacher_codes, result['feature_mean'], result['feature_scale'])
            feature_aux = ((terms*batch['valid']).sum(-1)/batch['valid'].sum(-1)).mean()
            total = total+feature_aux_weight*feature_aux
        stats.update(loss=float(total.detach()), feature_aux_nll_per_dim=float(feature_aux.detach()),
                     feature_aux_weight=float(feature_aux_weight))
        return total, stats
