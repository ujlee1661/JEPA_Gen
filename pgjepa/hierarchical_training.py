"""Training-only amortized customer inference and a censored sequence objective.

The posterior may read full TRAINING histories. Native generation never calls
these functions and samples customer identity once from the static-attribute prior.
Diffusion MSE and normalized JEPA error make this a composite variational
surrogate, not an exact data likelihood or exact ELBO.
"""
from __future__ import annotations

import torch

from .window_train import log_gap_survival


@torch.no_grad()
def cache_customer_summaries(model, records, device, batch_size=32):
    """Cache frozen-codec sufficient features; trainable posterior stays live."""
    summaries = []
    for start in range(0, len(records), batch_size):
        part = records[start:start + batch_size]
        width = max(len(record['delta']) for record in part)
        keys = ('x_num', 'x_cat', 'mask', 'delta')
        batch = {key: part[0][key].new_zeros((len(part), width, part[0][key].shape[-1]))
                 for key in keys}
        batch['t_valid'] = torch.zeros(len(part), width, dtype=torch.bool)
        batch['attrs'] = torch.stack([record['attrs'] for record in part])
        for index, record in enumerate(part):
            count = len(record['delta'])
            for key in keys:
                batch[key][index, :count] = record[key]
            batch['t_valid'][index, :count] = True
        summaries.append(model.posterior_summary({key: value.to(device) for key, value in batch.items()}))
    return torch.cat(summaries)


def hierarchical_generation_loss(model, batch, summaries, *, mean_terms):
    """Uniform-parent/cut estimator, including one KL cost per full sequence.

Likelihood terms carry (N_i+1)/mean(N+1). KL is divided only by the
population mean term count, so it is not accidentally charged N_i+1 times.
JEPA is an explicit unit-weight auxiliary objective for A4 only.
"""
    if mean_terms <= 0:
        raise ValueError('Positive population mean likelihood term count required')
    prefix = dict(batch['prefix'])
    indices = torch.as_tensor(batch['entity_indices'], device=summaries.device)
    customer_z, kl = model.posterior_sample(prefix['attrs'], summaries[indices])
    prefix['customer_z'] = customer_z
    z, state = model.init_state(prefix)
    valid = prefix['t_valid'].to(prefix['delta'].dtype)
    elapsed = (prefix['delta'][..., 0] * valid).sum(1)
    condition = model.condition_at(z, state, elapsed)
    observed, target, weights = batch['observed'], batch['target'], batch['weights']
    time_condition = model._time_condition(condition)
    density = model.time_head.nll(time_condition, target['delta'])
    survival = -log_gap_survival(model.time_head, time_condition, batch['remaining'])
    time_loss = (torch.where(observed, density, survival) * weights).mean()
    diffusion = condition.sum() * 0
    if bool(observed.any()):
        with torch.no_grad():
            raw = model.ae.encode({key: value[observed] for key, value in target.items()})
        errors = model.latent_diffusion_loss(raw, condition[observed], target['delta'][observed])
        diffusion = (errors * weights[observed]).sum() / len(weights)
    kl_loss = kl.mean() / mean_terms
    auxiliary = condition.sum() * 0
    extra = {}
    if model.arm == 'A4':
        auxiliary, extra = model.next_event_jepa_loss(prefix, z, target, observed, weights=weights)
    total = time_loss + diffusion + kl_loss + auxiliary
    stats = {'time_nll': float(time_loss.detach()), 'diffusion_loss': float(diffusion.detach()),
             'customer_kl_per_sequence': float(kl.detach().mean()),
             'customer_kl_scaled': float(kl_loss.detach()),
             'jepa_auxiliary': float(auxiliary.detach()),
             'customer_sample_std': float(customer_z.detach().std(0, unbiased=False).mean()),
             'censored_fraction': float((~observed).float().mean()), **extra}
    return total, stats
