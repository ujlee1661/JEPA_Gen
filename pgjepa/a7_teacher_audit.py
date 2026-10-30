"""Entity-paired diagnostics for a frozen A7 teacher, not generation fidelity.

All controls retain the trained decoder and the true reconstruction layout.
Zero-code is a train-mean-code control, not an optimized constant baseline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from .a7_data import collate_a7, inverse_marks
from .a7_spacing import face_log_mass, project_clr
from .a6_heldout_data import calibration_digest
from .experiment import write_json


class AuditSpecification(TypedDict):
    shuffle_seed: int
    bootstrap_seed: int
    bootstrap_replicates: int
    raw_code_variance_threshold: float
    confidence_level: float
    gate: str
    calibration: str


AUDIT_SPEC: AuditSpecification = {'shuffle_seed': 73, 'bootstrap_seed': 9147, 'bootstrap_replicates': 5000,
              'raw_code_variance_threshold': 1e-8, 'confidence_level': .95,
              'gate': 'finite nonconstant raw codes AND entity-paired mark MSE benefit lower CI > 0 versus zero and marks_zero',
              'calibration': 'unchanged supplied training calibration; never fit on held-out queries'}


def _load(value):
    return torch.load(value, map_location='cpu', weights_only=False) if isinstance(value, (str, Path)) else value


def _state_hash(model):
    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(key.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def raw_code_statistics(codes, teacher):
    raw = (codes.double()*teacher.code_std.detach().cpu().double()+teacher.code_mean.detach().cpu().double()).numpy()
    if not np.isfinite(raw).all():
        return {'finite': False, 'nonconstant': False}
    centered = raw-raw.mean(0)
    covariance = centered.T@centered/max(1, len(raw))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.)
    variance = float(np.trace(covariance)/raw.shape[1])
    if eigenvalues.sum() > 0:
        p = eigenvalues[eigenvalues > 0]/eigenvalues.sum()
        effective_rank = float(np.exp(-(p*np.log(p)).sum()))
    else:
        effective_rank = 0.
    return {'finite': True, 'nonconstant': variance > AUDIT_SPEC['raw_code_variance_threshold'],
            'samples': len(raw), 'raw_mean': raw.mean(0).tolist(),
            'raw_std': raw.std(0).tolist(), 'mean_raw_coordinate_variance': variance,
            'covariance_eigenvalues_descending': eigenvalues[::-1].tolist(),
            'entropy_effective_rank': effective_rank}


def paired_entity_bootstrap(entity_ids, reference, control, seed=9147, replicates=5000):
    """Average query deltas within entity, then resample entities equally."""
    groups: dict[str, list[float]] = {}
    for entity, left, right in zip(entity_ids, reference, control):
        if left is not None and right is not None and np.isfinite(left) and np.isfinite(right):
            groups.setdefault(str(entity), []).append(float(right-left))
    deltas = {key: float(np.mean(values)) for key, values in sorted(groups.items())}
    if not deltas:
        return {'entities': 0, 'mean_control_minus_reference': None, 'ci95': [None, None], 'per_entity_delta': {}}
    values = np.asarray(list(deltas.values()))
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(replicates, len(values)))].mean(1)
    return {'entities': len(values), 'mean_control_minus_reference': float(values.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'per_entity_delta': deltas}


def _metrics(prediction, batch):
    target = batch['tokens']
    error = prediction['tokens']-target
    active, events, valid = batch['active'], batch['event_valid'], batch['valid']
    time_error = project_clr(error[..., 0], active).square().sum(-1)
    time_dof = (active.sum(-1)-1).clamp_min(0)
    mark_error = torch.where(events[..., None], error[..., 1:].square(), torch.zeros_like(error[..., 1:])).sum((1, 2))
    mark_dof = events.sum(-1)*(target.shape[-1]-1)
    positive = batch['delta'] > 0
    faces = torch.zeros(len(target), device=target.device, dtype=torch.float64)
    if bool(positive.any()):
        faces[positive] = -face_log_mass(prediction['face_logits'][positive], active[positive], valid[positive])
    return {'mark_mse': [float(v) if int(d) else None for v, d in zip(mark_error/mark_dof.clamp_min(1), mark_dof)],
            'time_clr_scaled_mse': [float(v) if int(d) else None for v, d in zip(time_error/time_dof.clamp_min(1), time_dof)],
            'face_nll': faces.cpu().tolist()}


def _summary(metrics):
    result = {}
    for key, values in metrics.items():
        selected = [value for value in values if value is not None]
        result[key] = float(np.mean(selected)) if selected else None
    return result


def _raw_statistics(arrays):
    if not arrays:
        return {'rows': 0, 'columns': []}
    data = np.concatenate(arrays)
    columns = []
    for column in data.T:
        selected = column[np.isfinite(column)]
        columns.append({'finite_rows': len(selected), 'mean': float(selected.mean()) if len(selected) else None,
                        'std_population': float(selected.std()) if len(selected) else None,
                        'p99': float(np.quantile(selected, .99)) if len(selected) else None,
                        'zero_fraction': float((selected == 0).mean()) if len(selected) else None})
    return {'rows': len(data), 'columns': columns}


@torch.no_grad()
def audit_teacher(teacher, train_data, heldout_bundle, output, device='cuda', batch_size=32, prep=None):
    """Audit only when called by the trainer after strict train-only freezing.

    Inputs may be loaded bundles or .pt paths. Optional ``prep`` is the original
    TRAIN-fitted QuantilePreprocessor; without it raw inverse stats are omitted.
    Per-query overlapping rows are intentionally not called population fidelity.
    """
    if batch_size < 1:
        raise ValueError('Positive audit batch size required')
    if any(parameter.requires_grad for parameter in teacher.parameters()) or teacher.training:
        raise ValueError('Freeze teacher before calibration and audit')
    training, heldout = _load(train_data), _load(heldout_bundle)
    if not training['windows'] or not heldout['queries']:
        raise ValueError('Nonempty training and held-out banks required')
    if training['state']['time_scale'] != heldout['state']['time_scale']:
        raise ValueError('Held-out data must retain the training timing scale')
    if calibration_digest(training['state']) != calibration_digest(heldout['state']):
        raise ValueError('Held-out marks/time calibration differs from training')
    if 'training_state_digest' in heldout and heldout['training_state_digest'] != calibration_digest(training['state']):
        raise ValueError('Recorded held-out training calibration digest differs')
    if heldout.get('fit_split') not in (None, 'train only'):
        raise ValueError('Held-out calibrations must be training-only')
    teacher = teacher.to(device)
    before_hash = _state_hash(teacher)
    queries = heldout['queries']
    train_windows = training['windows']
    rng = np.random.default_rng(AUDIT_SPEC['shuffle_seed'])
    train_indices = np.sort(rng.choice(len(train_windows), min(4096, len(train_windows)), replace=False))
    def encode(items):
        return torch.cat([teacher.encode(collate_a7(items[start:start+batch_size], device)).cpu()
                          for start in range(0, len(items), batch_size)])
    train_codes = encode([train_windows[int(index)] for index in train_indices])
    codes = encode(queries)
    permutation = np.random.default_rng(AUDIT_SPEC['shuffle_seed']).permutation(len(queries))
    controls = {'proper': codes, 'zero': torch.zeros_like(codes), 'shuffle': codes[permutation]}
    marks_zero_codes = []
    for start in range(0, len(queries), batch_size):
        batch = collate_a7(queries[start:start+batch_size], device)
        batch['tokens'] = batch['tokens'].clone()
        batch['tokens'][..., 1:] = 0
        marks_zero_codes.append(teacher.encode(batch).cpu())
    controls['marks_zero'] = torch.cat(marks_zero_codes)
    metrics: dict[str, dict[str, list[float | None]]] = {
        name: {'mark_mse': [], 'time_clr_scaled_mse': [], 'face_nll': []} for name in controls}
    raw_arrays: dict[str, list[np.ndarray]] = {name: [] for name in [*controls, 'target']}
    finite_outputs = True
    for start in range(0, len(queries), batch_size):
        items = queries[start:start+batch_size]
        batch = collate_a7(items, device)
        if prep is not None:
            target_marks = batch['tokens'][..., 1:][batch['event_valid']]
            if len(target_marks):
                raw_arrays['target'].append(inverse_marks(target_marks, prep, training['state']['mark_token_state'])['x_num'])
        for name, values in controls.items():
            prediction = teacher.reconstruct(values[start:start+len(items)].to(device), batch)
            finite_outputs &= all(bool(torch.isfinite(value).all()) for value in prediction.values())
            values_by_metric = _metrics(prediction, batch)
            for key, values_list in values_by_metric.items():
                metrics[name][key].extend(values_list)
            if prep is not None:
                predicted_marks = prediction['tokens'][..., 1:][batch['event_valid']]
                if len(predicted_marks):
                    raw_arrays[name].append(inverse_marks(predicted_marks, prep, training['state']['mark_token_state'])['x_num'])
    entities = [item.get('entity_id', item.get('entity')) for item in queries]
    if any(value is None for value in entities):
        raise ValueError('Entity identities required for paired bootstrap')
    comparisons = {name: paired_entity_bootstrap(entities, metrics['proper']['mark_mse'], metrics[name]['mark_mse'])
                   for name in ('zero', 'shuffle', 'marks_zero')}
    code_stats = raw_code_statistics(codes, teacher)
    decision = finite_outputs and code_stats['finite'] and code_stats['nonconstant']
    for name in ('zero', 'marks_zero'):
        lower = comparisons[name]['ci95'][0]
        decision &= lower is not None and lower > 0
    after_hash = _state_hash(teacher)
    if before_hash != after_hash:
        raise RuntimeError('Teacher bytes changed during audit')
    result = {'version': 'a7_frozen_teacher_audit_v1', 'specification': AUDIT_SPEC,
              'decision': 'PASS' if decision else 'NEEDS_REVIEW',
              'teacher_state_sha256_before': before_hash, 'teacher_state_sha256_after': after_hash,
              'train_queries_encoded': len(train_indices), 'heldout_queries': len(queries),
              'heldout_entities': len(set(map(str, entities))),
              'finite_outputs': bool(finite_outputs), 'code_statistics_heldout_raw': code_stats,
              'code_statistics_training_raw': raw_code_statistics(train_codes, teacher),
              'summary': {name: _summary(value) for name, value in metrics.items()},
              'paired_entity_mark_mse': comparisons, 'per_query_metrics': metrics,
              'raw_numeric_query_weighted': {name: _raw_statistics(arrays) for name, arrays in raw_arrays.items()} if prep is not None else
                  {'available': False, 'reason': 'TRAIN-fitted original preprocessor not supplied'},
              'interpretation': ['Teacher reconstruction uses true N/face/calendar; not free generation.',
                  'Zero/shuffle controls share the SAME decoder, not optimized alternative baselines.',
                  'Raw rows repeat across overlapping held-out queries; not population synthesis scores.',
                  'Variance/rank and positive code-use effects do not prove sufficient or lossless embeddings.',
                  'Audit does not fit any calibration or choose thresholds from held-out outcomes.']}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'teacher_audit.json', result)
    lines = ['# Frozen A7 teacher audit', '', f"Decision: **{result['decision']}**.", '',
             f"Held-out entities: {result['heldout_entities']}; queries: {len(queries)}.", '',
             'Teacher reconstruction only, with true N/face/calendar. Controls retain the same trained decoder.', '',
             '| Code | Mark MSE | Scaled CLR MSE | Face NLL |', '|---|---:|---:|---:|']
    for name, value in result['summary'].items():
        lines.append(f"| {name} | {value['mark_mse']} | {value['time_clr_scaled_mse']} | {value['face_nll']} |")
    lines += ['', 'Entity-paired control-minus-proper mark MSE (positive favors proper code):', '']
    for name, value in comparisons.items():
        lines.append(f"- {name}: {value['mean_control_minus_reference']}; 95% CI {value['ci95']} ({value['entities']} entities).")
    lines += ['', *result['interpretation']]
    (output/'teacher_audit.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
