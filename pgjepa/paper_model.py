"""Parent-only, full-history PG-JEPA variant for the paper fidelity contract.

The original event MLP, GRUCell parameters, JEPA branches and FiLM denoiser are
unchanged. Unlike the forecasting model, z uses static parent attributes only,
empty BOS histories are valid, and the first event gap is its absolute offset
from the dataset calendar origin. This is a distinct conditioning protocol.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn.utils.rnn import pack_padded_sequence

from .stable_diffusion import StablePGGenerator


class PaperPGGenerator(StablePGGenerator):
    """Original PG network under parent-only full-population fidelity.

    ``fast_history=False`` is the explicit GRUCell reference path. The default
    packed path substitutes the same four GRUCell tensors into a parameterless-
    from-this-model's-perspective GRU template. No extra weights enter state_dict
    or the optimizer, and gradients flow to the original GRUCell parameters.
    """

    conditioning_protocol = 'parent_only_full_fidelity'

    def __init__(self, *args: Any, fast_history: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fast_history = fast_history
        # A call template, not a model component. Its initial dummy weights are
        # always replaced, and constructing it must not consume model RNG.
        with torch.random.fork_rng(devices=[]):
            template = nn.GRU(self.ae.latent_dim + 1, self.hidden_dim, batch_first=True)
        object.__setattr__(self, '_sequence_template', template)

    def parent_state(self, attrs: Tensor) -> tuple[Tensor, Tensor]:
        if attrs.ndim != 2 or attrs.shape[1] != self.attrs_dim:
            raise ValueError('Parent attributes must have shape [entity, attrs_dim]')
        stats = attrs.new_zeros((len(attrs), 2 * (self.ae.latent_dim + 1)))
        z = self.entity_encoder(torch.cat([stats, attrs], -1))
        if self.arm in {'A0', 'A1'}:
            z = torch.zeros_like(z)
        state = self.initial(z)
        if self.arm == 'A0':
            state = torch.zeros_like(state)
        return z, state

    def history_state(self, values: Tensor, valid: Tensor, initial: Tensor,
                      *, fast: bool | None = None) -> Tensor:
        """Read all prefix events, including their original first calendar gap."""
        if values.ndim != 3 or valid.shape != values.shape[:2]:
            raise ValueError('History values and validity dimensions differ')
        if initial.shape != (len(values), self.hidden_dim):
            raise ValueError('History initial state dimensions differ')
        lengths = valid.long().sum(1)
        expected = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
        if not bool(torch.equal(valid.bool(), expected)):
            raise ValueError('Full histories require contiguous valid prefixes followed by padding')
        use_fast = self.fast_history if fast is None else fast
        if not use_fast:
            state = initial
            for index in range(values.shape[1]):
                updated = self.context(values[:, index], state)
                state = torch.where(valid[:, index:index + 1], updated, state)
            return state
        selected = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if not len(selected):
            return initial
        packed = pack_padded_sequence(values[selected], lengths[selected].cpu(),
                                      batch_first=True, enforce_sorted=False)
        params = {name + '_l0': parameter for name, parameter in self.context.named_parameters()}
        template = self._sequence_template
        template.training = self.context.training
        _, hidden = functional_call(template, params, (packed, initial[selected].unsqueeze(0)))
        return initial.index_copy(0, selected, hidden[0])

    def init_state(self, prefix: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        numeric = prefix['x_num']
        if numeric.ndim != 3:
            raise ValueError('Event prefixes must have shape [entity, event, feature]')
        attrs = prefix.get('attrs', numeric.new_zeros((len(numeric), self.attrs_dim)))
        z, initial = self.parent_state(attrs)
        if self.arm == 'A0' or numeric.shape[1] == 0:
            return z, initial
        latent = self.ae.encode(prefix)
        valid = prefix.get('t_valid', torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device))
        state = self.history_state(self._time_input(latent, prefix['delta']), valid, initial)
        return z, state


def paper_sample_batch(records: list[dict], count: int, rng: np.random.Generator,
                       device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Sample BOS, every observed prefix, and the terminal censored term.

    All records must share a known calendar observation end. A record's final
    event is never used as its stopping boundary. Real future count and endpoint
    are loss metadata only; neither is passed to ``prefix`` or the conditioner.
    No history is truncated, including long histories.
    """
    if not records or count < 1:
        raise ValueError('Require a nonempty record population and positive batch size')
    ends = [float(torch.as_tensor(record['exposure_end'])) for record in records]
    if not np.isfinite(ends).all() or min(ends) < 0 or not np.allclose(ends, ends[0], rtol=0, atol=1e-6):
        raise ValueError('Paper fidelity requires one finite common calendar observation end')
    selected = rng.integers(len(records), size=count)
    examples = []
    for index in selected:
        record = records[int(index)]
        gaps = record['delta'].reshape(-1).double()
        if not bool(torch.isfinite(gaps).all()) or bool((gaps < 0).any()):
            raise ValueError('Calendar offsets/gaps must be finite and nonnegative')
        if 'calendar_times' in record:
            times = torch.as_tensor(record['calendar_times'], dtype=torch.float64).reshape(-1)
            if len(times) != len(gaps) or not bool(torch.isfinite(times).all()):
                raise ValueError('Exact calendar timestamps must match the observed event rows')
            differences = torch.diff(torch.cat([times.new_zeros(1), times]))
            if bool((differences < 0).any()):
                raise ValueError('Calendar timestamps must be nonnegative and ordered')
            epsilon = torch.finfo(record['delta'].dtype).eps if record['delta'].is_floating_point() else 0.
            if not torch.allclose(gaps, differences, rtol=4 * epsilon, atol=max(1e-10, 4 * epsilon)):
                raise ValueError('Neural gaps differ from the original calendar timestamps')
            clock_tolerance = 1e-8
        else:
            times = gaps.cumsum(0)
            epsilon = torch.finfo(record['delta'].dtype).eps if record['delta'].is_floating_point() else 0.
            clock_tolerance = max(1e-8, 4 * epsilon * max(1., ends[int(index)]))
        if len(times) and float(times[-1]) > ends[int(index)] + clock_tolerance:
            raise ValueError('Calendar observation end precedes an observed event')
        size = len(gaps)
        cut = int(rng.integers(0, size + 1))
        elapsed = float(times[cut - 1]) if cut else 0.
        examples.append((record, cut, size, max(0., ends[int(index)] - elapsed)))
    width = max(cut for _, cut, _, _ in examples)
    prototype = records[0]
    keys = ('x_num', 'x_cat', 'mask', 'delta')
    prefix = {key: prototype[key].new_zeros((count, width, prototype[key].shape[-1])) for key in keys}
    prefix['t_valid'] = torch.zeros(count, width, dtype=torch.bool)
    prefix['attrs'] = torch.stack([record.get('attrs', torch.zeros(0)) for record, *_ in examples])
    target = {key: prototype[key].new_zeros((count, prototype[key].shape[-1])) for key in keys}
    observed = torch.zeros(count, dtype=torch.bool)
    for row, (record, cut, size, _) in enumerate(examples):
        for key in keys:
            prefix[key][row, :cut] = record[key][:cut]
            if cut < size:
                target[key][row] = record[key][cut]
        prefix['t_valid'][row, :cut] = True
        observed[row] = cut < size
    denominator = float(np.mean([len(record['delta']) + 1 for record in records]))
    return {'prefix': {key: value.to(device) for key, value in prefix.items()},
            'target': {key: value.to(device) for key, value in target.items()},
            'observed': observed.to(device),
            'remaining': torch.tensor([remaining for *_, remaining in examples], device=device),
            'weights': torch.tensor([(size + 1) / denominator for _, _, size, _ in examples],
                                    dtype=torch.float32, device=device),
            'cuts': [cut for _, cut, *_ in examples], 'entity_indices': [int(index) for index in selected]}


@torch.no_grad()
def parent_generate_clock(model: PaperPGGenerator, attrs: Tensor, observation_end: float,
                          max_events: int = 4096, sample_steps: int = 50,
                          intervention: str = 'normal') -> dict[str, Tensor]:
    """Generate parent-conditioned complete histories on the closed [0, end] clock.

    The neural time head still draws float32 gaps. Their accumulation and exported
    absolute calendar timestamps are float64, so small later gaps are not lost
    when added to a large day offset. ``cap_hit`` means incomplete generation,
    never an observed sequence end. No real counts, first times or endpoints
    are consumed; only the common observation end and static parent table enter.

    A supplied TimeFIRST timeline must instead be preserved as its original
    float64 timestamps by its scaffold adapter, with float32 gaps used only as
    neural conditions. This native sampler does not rewrite such a timeline.
    """
    if not np.isfinite(observation_end) or observation_end < 0:
        raise ValueError('Common calendar observation end must be finite and nonnegative')
    if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
        raise ValueError('Positive integer event safety cap required')
    if sample_steps < 1:
        raise ValueError('Positive denoising sample steps required')
    if intervention not in {'normal', 'zero_plan', 'shuffle_plan', 'direct_latent', 'no_feedback'}:
        raise ValueError(f'Unknown intervention {intervention}')
    weights = model.initial.weight
    attrs = attrs.to(device=weights.device, dtype=weights.dtype)
    if attrs.ndim != 2 or attrs.shape[1] != model.attrs_dim or not bool(torch.isfinite(attrs).all()):
        raise ValueError('Finite static parent attributes must match attrs_dim')
    size = len(attrs)
    prefix = {
        'x_num': attrs.new_zeros((size, 0, model.ae.n_num)),
        'x_cat': torch.zeros((size, 0, len(model.ae.cat_sizes)), dtype=torch.long, device=attrs.device),
        'mask': attrs.new_zeros((size, 0, model.ae.n_num + len(model.ae.cat_sizes))),
        'delta': attrs.new_zeros((size, 0, 1)),
        't_valid': torch.zeros((size, 0), dtype=torch.bool, device=attrs.device),
        'attrs': attrs,
    }
    z, state = model.init_state(prefix)
    elapsed = torch.zeros(size, dtype=torch.float64, device=attrs.device)
    active = torch.ones(size, dtype=torch.bool, device=attrs.device)
    names = ('x_num', 'x_cat', 'mask', 'delta', 't_offset')
    rows: dict[str, list[Tensor]] = {name: [] for name in names}
    validity = []
    for _ in range(max_events):
        if not bool(active.any()):
            break
        condition = (model.condition_at(z, state, elapsed, intervention=intervention)
                     if hasattr(model, 'condition_at') else model._condition(z, state, intervention))
        delta = model.time_head.sample(model._time_condition(condition)).to(dtype=weights.dtype)
        if delta.shape != (size, 1):
            raise ValueError('Time head must return [parent, 1] gaps')
        if not bool(torch.isfinite(delta[active]).all()) or bool((delta[active] < 0).any()):
            raise ValueError('Generated neural gaps must be finite and nonnegative')
        next_time = elapsed + delta[:, 0].double()
        accepted = active & (next_time <= observation_end)
        active = accepted
        if not bool(accepted.any()):
            break
        # Inactive rows stay in their original cohort positions. Their unused
        # future gaps must not send infinities into the mark decoder.
        neural_delta = torch.where(accepted[:, None], delta, torch.zeros_like(delta))
        latent = model.sample_latent(condition, neural_delta, sample_steps)
        row = model.ae.decode(latent, sample=True)
        for name in ('x_num', 'x_cat', 'mask'):
            rows[name].append(torch.where(accepted[:, None], row[name], torch.zeros_like(row[name])))
        rows['delta'].append(neural_delta)
        rows['t_offset'].append(torch.where(accepted[:, None], next_time[:, None],
                                           torch.zeros_like(next_time[:, None])))
        validity.append(accepted.clone())
        elapsed = torch.where(accepted, next_time, elapsed)
        if intervention != 'no_feedback' and model.arm != 'A0':
            updated = model.feedback(state, row, neural_delta,
                                     latent if intervention == 'direct_latent' else None)
            state = torch.where(accepted[:, None], updated, state)
    prototypes = {**prefix, 't_offset': elapsed[:, None, None]}
    output = {name: torch.stack(values, 1) if values else
              prototypes[name].new_zeros((size, 0, prototypes[name].shape[-1]))
              for name, values in rows.items()}
    output['t_valid'] = torch.stack(validity, 1) if validity else prefix['t_valid']
    output['lengths'] = output['t_valid'].sum(1)
    output['cap_hit'] = active.clone()
    output['attrs'] = attrs
    output['observation_end'] = torch.tensor(observation_end, dtype=torch.float64, device=attrs.device)
    if hasattr(model, 'posterior_sample'):
        output['customer_z'] = z
    return output
