"""Native ordered-window spacing coordinates; no calendar correction.

Positive-duration paths lie on a simplex, including leading and terminal slack.
An explicit face represents exact zero intervals. CLR coordinates live in its
zero-sum subspace. All calendar arithmetic here is float64; mark coordinates
reuse the original fitted preprocessing without its absolute-time coordinate.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _calendar(anchor: float, horizon: float) -> tuple[float, float]:
    values = torch.tensor([anchor, horizon], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or horizon < 0:
        raise ValueError('Finite anchor and nonnegative horizon required')
    if not bool(torch.isfinite(values.sum())):
        raise ValueError('Window endpoint must be finite')
    return float(anchor), float(horizon)


def project_clr(values: Tensor, active: Tensor) -> Tensor:
    """Orthogonal projection along the final dimension; empty faces map to zero.

    Broadcasting active across leading dimensions is supported (e.g. noise
    samples). Empty faces are reserved for zero-duration windows, not positive
    simplex paths. Inactive coordinates have exactly zero derivative.
    """
    if values.ndim < 1 or not values.is_floating_point():
        raise ValueError('Floating CLR arrays with a final coordinate axis required')
    active = torch.as_tensor(active, device=values.device, dtype=torch.bool)
    try:
        active = torch.broadcast_to(active, values.shape)
    except RuntimeError as error:
        raise ValueError('CLR values and support shapes disagree') from error
    selected = torch.where(active, values, torch.zeros_like(values))
    if not bool(torch.isfinite(selected).all()):
        raise ValueError('Active CLR values must be finite')
    size = active.sum(-1, keepdim=True).clamp_min(1).to(values)
    return torch.where(active, selected-selected.sum(-1, keepdim=True)/size, torch.zeros_like(values))


def encode_spacing(times: Tensor, anchor: float, horizon: float) -> dict[str, Tensor]:
    """Encode one ordered path with N+1 gaps; no epsilon or quantization.

    The input must already be in stable event order. Tied rows keep that order.
    Float32 inputs cannot regain precision: callers should pass source times.
    """
    anchor, horizon = _calendar(anchor, horizon)
    times = torch.as_tensor(times, dtype=torch.float64)
    if times.ndim != 1 or not bool(torch.isfinite(times).all()):
        raise ValueError('Finite one-dimensional ordered times required')
    if bool((times < anchor).any()) or bool((times > anchor+horizon).any()):
        raise ValueError('Event outside the declared window')
    if bool((times[1:] < times[:-1]).any()):
        raise ValueError('Event times must already be ordered')
    endpoints = torch.cat([times.new_tensor([anchor]), times, times.new_tensor([anchor+horizon])])
    gaps = endpoints[1:]-endpoints[:-1]
    active = gaps > 0
    clr = torch.zeros_like(gaps)
    if horizon > 0:
        if not bool(active.any()):
            raise ValueError('Positive duration is not representable at this calendar precision')
        log_gap = gaps[active].log()
        clr[active] = log_gap-log_gap.mean()
    return {'clr':clr, 'active':active, 'gaps':gaps}


def decode_spacing(clr: Tensor, active: Tensor, anchor: float,
                   horizon: float) -> dict[str, Tensor]:
    """Inverse native coordinate map; reject numerical support violations.

    Gauge projection is intrinsic to CLR, not a fitted calendar adjustment.
    Returns N event times plus N+1 native gaps, all in float64.
    """
    anchor, horizon = _calendar(anchor, horizon)
    clr = torch.as_tensor(clr, dtype=torch.float64)
    active = torch.as_tensor(active, device=clr.device, dtype=torch.bool)
    if clr.ndim != 1 or not len(clr) or active.shape != clr.shape:
        raise ValueError('One N+1 CLR vector and matching support required')
    if not bool(torch.isfinite(clr).all()):
        raise ValueError('Finite CLR vector required')
    if horizon == 0:
        if bool(active.any()):
            raise ValueError('Zero-duration windows have an empty timing support')
        return {'times':clr.new_full((len(clr)-1,), anchor), 'gaps':torch.zeros_like(clr)}
    if not bool(active.any()):
        raise ValueError('Positive-duration windows require a nonempty face')
    centered = project_clr(clr, active)
    positive = horizon*torch.softmax(centered[active], -1)
    if not bool(torch.isfinite(positive).all()) or bool((positive <= 0).any()):
        raise FloatingPointError('CLR inversion underflowed or produced invalid positive gaps')
    gaps = torch.zeros_like(clr).masked_scatter(active, positive)
    times = anchor+gaps.cumsum(0)[:-1]
    # Structural boundary equality: after the final active gap, every event is
    # at the declared endpoint. This includes several rows tied at that end.
    last_active = int(torch.nonzero(active, as_tuple=False)[-1, 0])
    if last_active < len(times):
        times = torch.where(torch.arange(len(times),device=times.device) >= last_active,
                            times.new_tensor(anchor+horizon), times)
    if bool((times < anchor).any()) or bool((times > anchor+horizon).any()):
        raise FloatingPointError('Decoded calendar exceeds the native domain')
    return {'times':times, 'gaps':gaps}


def _face_terms(logits: Tensor, valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if logits.ndim < 1 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite floating face logits required')
    valid = torch.ones_like(logits,dtype=torch.bool) if valid is None else torch.as_tensor(valid,device=logits.device,dtype=torch.bool)
    if valid.shape != logits.shape or not bool(valid.any(-1).all()):
        raise ValueError('Each face must have at least one valid coordinate')
    # Float64 avoids low-precision cancellation for extreme but finite logits.
    values = logits.double()
    log_p = F.logsigmoid(values)
    log_q = F.logsigmoid(-values)
    log_q_valid = torch.where(valid,log_q,torch.zeros_like(log_q))
    prefix = torch.cat([torch.zeros_like(log_q_valid[..., :1]),
                        log_q_valid.cumsum(-1)[..., :-1]], -1)
    first_log = torch.where(valid,log_p+prefix,torch.full_like(values,-torch.inf))
    log_z = torch.logsumexp(first_log,-1)
    return log_p,log_q,first_log,log_z


def face_log_mass(logits: Tensor, active: Tensor,
                  valid: Tensor | None = None) -> Tensor:
    """Log mass of independent Bernoulli bits conditioned on being nonempty.

    Empty or padding-violating outcomes have log mass minus infinity. This law
    applies only to positive-duration windows; zero-duration time is degenerate.
    Output is float64 for stable rare nonempty probabilities and gradients.
    """
    log_p,log_q,_,log_z = _face_terms(logits,valid)
    active = torch.as_tensor(active,device=logits.device,dtype=torch.bool)
    if active.shape != logits.shape:
        raise ValueError('Face and logits shapes disagree')
    valid = torch.ones_like(active) if valid is None else torch.as_tensor(valid,device=logits.device,dtype=torch.bool)
    impossible = (~active.any(-1)) | (active & ~valid).any(-1)
    terms = torch.where(valid,torch.where(active,log_p,log_q),torch.zeros_like(log_p))
    result = terms.sum(-1)-log_z
    return torch.where(impossible,torch.full_like(result,-torch.inf),result)


def first_active_log_probs(logits: Tensor, valid: Tensor | None = None) -> Tensor:
    """Normalized exact first-positive-index law, with padding at minus infinity."""
    _,_,first_log,log_z = _face_terms(logits,valid)
    return first_log-log_z.unsqueeze(-1)


@torch.no_grad()
def sample_face(logits: Tensor, valid: Tensor | None = None,
                generator: torch.Generator | None = None) -> Tensor:
    """Exact nonempty sampler; no rejection, probability clipping or zero quota."""
    log_first = first_active_log_probs(logits,valid)
    shape = logits.shape
    flat = log_first.reshape(-1,shape[-1])
    first = torch.multinomial(flat.exp(),1,generator=generator).reshape(shape[:-1])
    # Logistic noise comparison avoids sigmoid rounding to exactly zero/one.
    uniform = torch.rand(shape,device=logits.device,dtype=torch.float64,generator=generator)
    random_logit = uniform.log()-torch.log1p(-uniform)
    later_bits = random_logit < logits.double()
    positions = torch.arange(shape[-1],device=logits.device)
    active = (positions == first.unsqueeze(-1)) | ((positions > first.unsqueeze(-1)) & later_bits)
    if valid is not None:
        active = active & torch.as_tensor(valid,device=logits.device,dtype=torch.bool)
    return active


def encode_record_window(record: dict, cut: int, horizon: float,
                         token_state: dict) -> dict:
    """Adapter using original QT/masks/categories and exact source calendars.

    Profile targets remain the original A6 profile, computed separately. The
    returned mark tokens omit the original probit time channel only.
    """
    from .a6_data import encode_window, _times

    item = encode_window(record,cut,horizon,token_state)
    count = item['count']
    times = torch.as_tensor(_times(record)[cut:cut+count],dtype=torch.float64)
    spacing = encode_spacing(times,item['anchor'],item['delta'])
    return {'count':count,'anchor':item['anchor'],'delta':item['delta'],
            'marks':item['future_tokens'][:,1:].clone(), 'times':times,
            'past_tokens':item['past_tokens'],'attrs':item['attrs'], **spacing}
