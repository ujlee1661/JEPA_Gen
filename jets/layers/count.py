"""Discretized log-normal mixture count probabilities."""
from __future__ import annotations
import torch
from torch import Tensor

def _log_sub_exp(large: Tensor, small: Tensor) -> Tensor:
    difference = (small-large).clamp_max(0)
    return large+torch.log(-torch.expm1(difference))


def discretized_count_log_mass(count: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Log mass for N=max(round(exp(U)-1),0), U~Normal(mean,scale).

    Batch count may be [B] and parameter matrices [B,K]. No population-specific
    minimum/maximum is imposed. The zero bin integrates all U<log(1.5).
    """
    if bool((count < 0).any()) or bool((count != count.round()).any()):
        raise ValueError('Nonnegative integer cardinalities required')
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError('Finite Gaussian parameters and positive scales required')
    dtype = mean.dtype
    count = count.to(device=mean.device, dtype=torch.float64)
    if count.ndim+1 == mean.ndim:
        count = count.unsqueeze(-1)
    mu, sigma = mean.double(), scale.double()
    lower = ((count+.5).log()-mu)/sigma
    upper = ((count+1.5).log()-mu)/sigma
    use_sf = lower > 0
    # Mask inputs to the unused subtraction too: an unused log(0) derivative
    # could otherwise poison gradients through torch.where.
    cdf_lower = torch.where(use_sf, torch.full_like(lower, -1), lower)
    cdf_upper = torch.where(use_sf, torch.zeros_like(upper), upper)
    sf_lower = torch.where(use_sf, lower, torch.zeros_like(lower))
    sf_upper = torch.where(use_sf, upper, torch.ones_like(upper))
    cdf = _log_sub_exp(torch.special.log_ndtr(cdf_upper), torch.special.log_ndtr(cdf_lower))
    sf = _log_sub_exp(torch.special.log_ndtr(-sf_lower), torch.special.log_ndtr(-sf_upper))
    positive = torch.where(use_sf, sf, cdf)
    result = torch.where(count == 0, torch.special.log_ndtr(upper), positive)
    return result.to(dtype)
