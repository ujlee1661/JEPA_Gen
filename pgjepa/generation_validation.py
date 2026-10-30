"""Fixed validation-only observation diagnostics for generative snapshots.

This module evaluates conditional next-row quality at the actual next-event
gap. It does not select a future event count, read files, calibrate output
values, or replace a model's decoder. Free clock-window evaluation remains a
separate test of the complete event process.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import random
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

@contextmanager
def _preserve_runtime(model=None, prep=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()] if model is not None else []
    audit = copy.deepcopy(getattr(prep, 'last_inverse_audit', None))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for module, training in modes:
            module.training = training
        with torch.no_grad():
            for buffer, previous in buffers:
                buffer.copy_(previous)
        if prep is not None and hasattr(prep, 'last_inverse_audit'):
            prep.last_inverse_audit = audit


def _options(config):
    options = dict(config.get('generation_validation', {}))
    options.setdefault('population', 512)
    options.setdefault('draws', 3)
    options.setdefault('sample_steps', 50)
    options.setdefault('seed', 551311)
    options.setdefault('draw_seed', 650011)
    options.setdefault('batch_size', 128)
    return options


def prepare_generation_validation(records, prep, config, device='cpu', data=None):
    """Cache model-independent observed-next examples from supplied validation.

    ``records`` must be the caller's validation split. ``data`` is accepted for
    interface convenience but never opened. Prefixes are sampled independently
    of future activity, including terminal terms; only the observed-next terms
    are retained for this conditional mark-generation diagnostic. Exposure is
    never passed in the cached prefix or target conditioner.
    """
    from .window_train import sample_exposure_batch

    if not records:
        raise ValueError('A nonempty validation record list is required')
    options = _options(config)
    count, batch_size = int(options['population']), int(options['batch_size'])
    if count < 1 or batch_size < 1:
        raise ValueError('Positive validation population and batch size required')
    cache: dict[str, Any] = {
        'status': 'ok', 'population': 0, 'requested_population': count, 'batches': [],
        'sampling_seed': int(options['seed']),
        'scope': 'validation-only observed-next rows; actual next gap; fixed paired truth masks',
        'truth_scope': 'inverse train-fitted quantile-encoded validation targets, including training-support saturation',
        'reads_files': False,
    }
    if not any(len(record['delta']) > 1 for record in records):
        cache['status'] = 'no_observed_next_events'
        return cache
    maximum = int(options.get('max_batches', max(100, math.ceil(count / batch_size) * 100)))
    with _preserve_runtime(prep=prep), torch.no_grad():
        rng = np.random.default_rng(int(options['seed']))
        for _ in range(maximum):
            batch = sample_exposure_batch(records, batch_size, rng, str(device),
                                          config.get('history_max_events', 64))
            observed = torch.nonzero(batch['observed'], as_tuple=False).flatten()
            take = min(len(observed), count - cache['population'])
            if take:
                indices = observed[:take]
                cached_batch = {
                    'prefix': {key: value[indices].detach().clone() for key, value in batch['prefix'].items()},
                    'target': {key: value[indices].detach().clone() for key, value in batch['target'].items()},
                }
                if 'raw_target_numeric' in batch:
                    cached_batch['raw_target_numeric'] = batch['raw_target_numeric'][indices].detach().clone()
                cache['batches'].append(cached_batch)
                cache['population'] += take
            if cache['population'] == count:
                break
        if not cache['population']:
            cache['status'] = 'no_observed_next_events'
            return cache
        if cache['population'] < count:
            cache['status'] = 'partial_observed_population'
        target = {key: torch.cat([batch['target'][key] for batch in cache['batches']])
                  for key in ('x_num', 'x_cat', 'mask', 'delta')}
        if all('raw_target_numeric' in batch for batch in cache['batches']):
            cache['truth_numeric'] = torch.cat([batch['raw_target_numeric'] for batch in cache['batches']]).cpu().numpy()
            cache['truth_scope'] = 'original raw validation numeric sidecar; no inverse-quantile clipping'
        else:
            cache['truth_numeric'] = prep.inverse_numeric(target['x_num'])
        cache['truth_mask'] = target['mask'].cpu().numpy()
        cache['truth_categories'] = target['x_cat'].cpu().numpy()
    return cache


def _numeric_stats(raw, mask, names):
    result = {}
    for index, name in enumerate(names):
        values = np.asarray(raw)[..., index][np.asarray(mask)[..., index] >= .5]
        finite = values[np.isfinite(values)]
        result[name] = {
            'observed': int(len(values)), 'finite_observed': int(len(finite)),
            'nonfinite_fraction': float(1 - len(finite) / len(values)) if len(values) else None,
            'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std()) if len(finite) else None,
            'zero_fraction': float(np.mean(abs(finite) < 1e-6)) if len(finite) else None,
            'negative_fraction': float(np.mean(finite < -1e-6)) if len(finite) else None,
            'below_minus_10000': float(np.mean(finite < -10000)) if len(finite) else None,
            'above_10000': float(np.mean(finite > 10000)) if len(finite) else None,
            'q01_q05_q50_q95_q99': np.quantile(finite, [.01, .05, .5, .95, .99]).tolist() if len(finite) else None,
        }
    return result


def _finite(value):
    return value is not None and bool(np.isfinite(value))


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def generation_validation(model, cache, prep, config):
    """Score a snapshot and gate zeros, spread, mean bias and finite output.

    The reported selection score is mean numeric KS + mean categorical TVD +
    mask marginal MAE. Gates are separate, so a low aggregate score cannot hide
    one badly distorted variable. All global RNG states, module modes, decoder
    audit buffers and the preprocessing inverse audit are restored afterwards.
    """
    options = _options(config)
    draws, steps = int(options['draws']), int(options['sample_steps'])
    if draws < 1 or steps < 1:
        raise ValueError('Positive draw count and sample steps required')
    result: dict[str, Any] = {
        'population': cache['population'], 'draws': draws, 'sample_steps': steps,
        'scope': cache['scope'], 'truth_scope': cache.get('truth_scope', 'no observed validation targets'),
        'status': cache['status'], 'all_gates_pass': False, 'selection_score': None,
    }
    if not cache['population']:
        result['gates'] = {'population_supported': False}
        return result
    with _preserve_runtime(model, prep), torch.no_grad():
        model.eval()
        conditions, targets, latent_truth, reconstructed = [], [], [], []
        for batch in cache['batches']:
            z, state = model.init_state(batch['prefix'])
            conditions.append(model._condition(z, state))
            targets.append(batch['target']['delta'])
            latent = model.ae.encode(batch['target'])
            latent_truth.append(model.ae.normalize(latent).cpu().numpy())
            reconstructed.append(prep.inverse_numeric(model.ae.decode(latent, sample=False)['x_num']))
        condition, delta = torch.cat(conditions), torch.cat(targets)
        numeric_parts, mask_parts, category_parts, latent_parts = [], [], [], []
        for draw in range(draws):
            seed = int(options['draw_seed']) + draw
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            latent = model.sample_latent(condition, delta, steps)
            row = model.ae.decode(latent, sample=True)
            numeric_parts.append(prep.inverse_numeric(row['x_num']))
            mask_parts.append(row['mask'].cpu().numpy())
            category_parts.append(row['x_cat'].cpu().numpy())
            latent_parts.append(model.ae.normalize(latent).cpu().numpy())
        truth, actual_mask = cache['truth_numeric'], cache['truth_mask']
        raw, generated_mask = np.concatenate(numeric_parts), np.concatenate(mask_parts)
        matched_mask = np.tile(actual_mask, (draws, 1))
        generated_categories = np.concatenate(category_parts)
        real_stats = _numeric_stats(truth, actual_mask, prep.numeric_names)
        generated_stats = _numeric_stats(raw, matched_mask, prep.numeric_names)
        result.update(truth_numeric=real_stats,
                      reconstruction_numeric=_numeric_stats(np.concatenate(reconstructed), actual_mask, prep.numeric_names),
                      generated_numeric=generated_stats)
        defaults = {'zero_fraction_abs_max': .05, 'std_ratio_max': 1.5,
                    'std_ratio_min': 0., 'mean_error_std_max': .2}
        defaults.update(options.get('gates', {}))
        defaults.update(config.get('generation_validation_gates', {}))
        aliases = config.get('validation_generation_gate', {})
        for old, new in {'count_zero_abs_error_max': 'zero_fraction_abs_max',
                         'numeric_std_ratio_max': 'std_ratio_max',
                         'numeric_mean_error_in_real_std_max': 'mean_error_std_max'}.items():
            if old in aliases:
                defaults[new] = aliases[old]
        gates: dict[str, Any] = {}
        ks: dict[str, float | None] = {}
        for index, name in enumerate(prep.numeric_names):
            actual = truth[:, index][actual_mask[:, index] >= .5]
            synthetic = raw[:, index][matched_mask[:, index] >= .5]
            a, b = real_stats[name], generated_stats[name]
            if not len(actual):
                ks[name] = None
                gates[name] = {'supported': False, 'all_pass': None}
                continue
            finite = bool(np.isfinite(actual).all() and np.isfinite(synthetic).all() and len(synthetic))
            ks[name] = float(ks_2samp(actual, synthetic).statistic) if finite else 1.
            zero_error = abs(a['zero_fraction'] - b['zero_fraction']) if _finite(b['zero_fraction']) else None
            std_floor = max(1e-6, abs(a['mean']) * 1e-6)
            ratio = (b['std'] / a['std'] if a['std'] > std_floor else
                     1. if _finite(b['std']) and b['std'] <= std_floor else None) if _finite(b['std']) else None
            mean_error = abs(b['mean'] - a['mean']) / max(a['std'], std_floor) if _finite(b['mean']) else None
            variable_gate = {
                'supported': True, 'finite': finite,
                'zero_fraction_absolute_error': zero_error,
                'zero_fraction_pass': _finite(zero_error) and zero_error <= defaults['zero_fraction_abs_max'],
                'std_ratio': ratio,
                'std_ratio_pass': _finite(ratio) and defaults['std_ratio_min'] <= ratio <= defaults['std_ratio_max'],
                'mean_error_in_real_std': mean_error,
                'mean_error_pass': _finite(mean_error) and mean_error <= defaults['mean_error_std_max'],
            }
            variable_gate['all_pass'] = all(variable_gate[key] for key in
                                            ('finite', 'zero_fraction_pass', 'std_ratio_pass', 'mean_error_pass'))
            gates[name] = variable_gate
        category_errors: list[float] = []
        category_details: dict[str, Any] = {}
        categories_valid = True
        for index, size in enumerate(prep.cat_sizes):
            am = actual_mask[:, prep.n_num + index] >= .5
            bm = matched_mask[:, prep.n_num + index] >= .5
            a = cache['truth_categories'][:, index][am]
            b = generated_categories[:, index][bm]
            name = prep.categorical_names[index]
            valid = bool(np.all((b >= 0) & (b < size)))
            categories_valid &= valid
            if not len(a):
                category_details[name] = {'supported': False, 'tvd': None}
                continue
            if valid and len(b):
                ac = np.bincount(a.astype(int), minlength=size).astype(float)
                bc = np.bincount(b.astype(int), minlength=size).astype(float)
                error = float(abs(ac / ac.sum() - bc / bc.sum()).sum() / 2)
            else:
                error = 1.
            category_errors.append(error)
            category_details[name] = {'supported': True, 'tvd': error, 'valid_categories': valid}
        finite_mask = bool(np.isfinite(generated_mask).all()
                           and ((generated_mask >= 0) & (generated_mask <= 1)).all())
        mask_error = float(abs(generated_mask.mean(0) - actual_mask.mean(0)).mean()) if finite_mask else 1.
        numeric_ks = [value for value in ks.values() if value is not None]
        result.update(numeric_ks=ks, categorical_tvd=category_details,
                      categorical_tvd_mean=float(np.mean(category_errors)) if category_errors else 0.,
                      mask_rate_error=mask_error,
                      selection_score=(float(np.mean(numeric_ks)) if numeric_ks else 0.)
                                      + (float(np.mean(category_errors)) if category_errors else 0.) + mask_error,
                      gates=gates, gate_thresholds=defaults)
        true_geometry, sampled_geometry = np.concatenate(latent_truth), np.concatenate(latent_parts)
        true_rms = float(np.sqrt(np.mean(true_geometry ** 2)))
        generated_rms = float(np.sqrt(np.mean(sampled_geometry ** 2)))
        finite_latent = bool(np.isfinite(sampled_geometry).all())
        result['true_normalized_latent_rms'] = true_rms if np.isfinite(true_rms) else None
        result['generated_normalized_latent_rms'] = generated_rms if np.isfinite(generated_rms) else None
        result['all_gates_pass'] = (all(row['all_pass'] for row in gates.values() if row['supported'])
                                   and finite_mask and finite_latent and categories_valid)
    return _json_finite(result)
