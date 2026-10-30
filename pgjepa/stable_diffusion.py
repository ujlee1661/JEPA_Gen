"""Architecture-preserving v prediction with a matched zero-terminal-SNR prior.

Only the diffusion target and schedule differ from PGGenerator. The MLP event
codec, GRU history, JEPA branches and FiLM denoiser retain the original shapes.
The x0 recovery identity has coefficients bounded by one, avoiding epsilon
prediction's division by sqrt(alpha_bar) near the terminal noise level.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .models import PGGenerator, prefix_of


def _alpha_like(alpha: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(alpha, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('alpha_bar must be finite and in [0, 1]')
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def v_target(clean: Tensor, noise: Tensor, alpha: Tensor | float) -> Tensor:
    """Velocity target for x_t=sqrt(alpha)*x0+sqrt(1-alpha)*epsilon."""
    coefficient = _alpha_like(alpha, clean)
    return coefficient.sqrt() * noise - (1 - coefficient).sqrt() * clean


def v_to_x0_epsilon(noisy: Tensor, velocity: Tensor,
                    alpha: Tensor | float) -> tuple[Tensor, Tensor]:
    coefficient = _alpha_like(alpha, noisy)
    clean = coefficient.sqrt() * noisy - (1 - coefficient).sqrt() * velocity
    noise = (1 - coefficient).sqrt() * noisy + coefficient.sqrt() * velocity
    return clean, noise


def zero_terminal_snr(alpha_bar: Tensor) -> Tensor:
    """Rescale sqrt cumulative alphas, preserving the first step exactly."""
    if alpha_bar.ndim != 1 or len(alpha_bar) < 2:
        raise ValueError('At least two cumulative noise levels required')
    if not bool(torch.isfinite(alpha_bar).all()) or bool((alpha_bar <= 0).any()):
        raise ValueError('Original cumulative alphas must be finite and positive')
    if bool((alpha_bar > 1).any()) or bool((alpha_bar[1:] > alpha_bar[:-1]).any()):
        raise ValueError('Original cumulative alphas must decrease within (0, 1]')
    roots = alpha_bar.double().sqrt()
    if not bool(roots[0] > roots[-1]):
        raise ValueError('Noise schedule must have a nonzero range')
    result = ((roots - roots[-1]) * roots[0] / (roots[0] - roots[-1])).square()
    result[0] = alpha_bar[0].double()
    result[-1] = 0
    return result.to(alpha_bar.dtype)


class StablePGGenerator(PGGenerator):
    """Original PG-JEPA network trained with velocity rather than epsilon."""

    def __init__(self, *args: Any, observation_loss_weight: float = 0., **kwargs: Any):
        if not math.isfinite(observation_loss_weight) or observation_loss_weight < 0:
            raise ValueError('Observation loss weight must be finite and nonnegative')
        super().__init__(*args, **kwargs)
        self.alpha_bar.copy_(zero_terminal_snr(self.alpha_bar))
        self.prediction_type = 'v'
        self.observation_loss_weight = observation_loss_weight

    def diffusion_terms(self, raw_target: Tensor, condition: Tensor, delta: Tensor,
                        *, steps: Tensor | None = None, noise: Tensor | None = None) -> dict[str, Tensor]:
        clean = self.ae.normalize(raw_target.detach())
        if steps is None:
            steps = torch.randint(self.diffusion_steps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        alpha = self.alpha_bar[steps].unsqueeze(-1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        target = v_target(clean, noise, alpha)
        predicted = self.denoiser(noisy, steps, torch.cat([condition, delta.log1p()], -1))
        recovered, _ = v_to_x0_epsilon(noisy, predicted, alpha)
        return {'per_row_loss': (predicted - target).square().mean(-1),
                'x0_pred': self.ae.denormalize(recovered),
                'normalized_x0_pred': recovered, 'predicted_v': predicted,
                'target_v': target, 'steps': steps}

    def latent_diffusion_loss(self, raw_target: Tensor, condition: Tensor, delta: Tensor) -> Tensor:
        return self.diffusion_terms(raw_target, condition, delta)['per_row_loss']

    def generation_loss(self, batch: dict[str, Tensor], cut: int) -> tuple[Tensor, dict[str, float]]:
        if cut < 1 or cut >= batch['x_num'].shape[1]:
            raise ValueError('Training cut requires a prefix and a next event')
        z, state = self.init_state(prefix_of(batch, cut))
        condition = self._condition(z, state)
        delta = batch['delta'][:, cut]
        with torch.no_grad():
            target = self.ae.encode(batch)[:, cut]
        errors = self.latent_diffusion_loss(target, condition, delta)
        valid = batch.get('t_valid', torch.ones(batch['x_num'].shape[:2],
                          dtype=torch.bool, device=target.device))[:, cut].float()
        diffusion = (errors * valid).sum() / valid.sum().clamp_min(1)
        gaps = self.time_head.nll(self._time_condition(condition), delta)
        timing = (gaps * valid).sum() / valid.sum().clamp_min(1)
        return diffusion + timing, {'diffusion_loss': float(diffusion.detach()),
                                     'time_nll': float(timing.detach())}

    @torch.no_grad()
    def sample_latent(self, condition: Tensor, delta: Tensor, sample_steps: int = 20) -> Tensor:
        count = min(max(int(sample_steps), 1), self.diffusion_steps)
        schedule = torch.linspace(self.diffusion_steps - 1, 0, count,
                                   device=condition.device).round().long()
        noisy = torch.randn((len(condition), self.ae.latent_dim), device=condition.device)
        full_condition = torch.cat([condition, delta.log1p()], -1)
        for index, step in enumerate(schedule):
            velocity = self.denoiser(noisy, step.expand(len(condition)), full_condition)
            clean, epsilon = v_to_x0_epsilon(noisy, velocity, self.alpha_bar[step])
            if index + 1 < len(schedule):
                previous = self.alpha_bar[schedule[index + 1]]
                noisy = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
            else:
                noisy = clean
        return self.ae.denormalize(noisy)
