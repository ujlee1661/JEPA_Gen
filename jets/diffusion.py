"""Velocity diffusion transforms and zero-terminal-SNR schedule."""
from __future__ import annotations
import torch
from torch import Tensor

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
