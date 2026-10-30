"""Matched fixed-budget distribution teacher component experiment.

The evaluation is privileged-layout oracle reconstruction, never native future
generation fidelity. All learned calibrations use training entities only.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .a6_heldout_data import calibration_digest
from .a7_data import collate_a7, inverse_marks
from .a7_distribution_teacher import A7DistributionTeacher
from .a7_spacing import project_clr
from .a7_teacher_audit import paired_entity_bootstrap, raw_code_statistics
from .a7_training import atomic_save, load_data, tensor_digest, source_hashes as legacy_source_hashes
from .experiment import digest, read_json, seed_all, write_json
from .paper_data import sha256


DEFAULT = {'seed': 9, 'steps': 5000, 'batch_size': 32, 'learning_rate': .0003,
    'weight_decay': .01, 'gradient_clip': 1., 'log_interval': 100,
    'objectives': ['energy', 'mse'], 'selection': 'fixed_final_update_no_validation',
    'eval_samples': [16, 32], 'eval_batch_size': 8, 'eval_seed': 9147,
    'teacher': {'code_dim': 64, 'width': 128, 'layers': 2, 'heads': 4, 'decoder_layers': 4}}


def distribution_config(overrides=None):
    config = copy.deepcopy(DEFAULT)
    for key, value in (overrides or {}).items():
        if key == 'teacher':
            config[key].update(value)
        else:
            config[key] = copy.deepcopy(value)
    if config['objectives'] != ['energy', 'mse'] or config['selection'] != DEFAULT['selection']:
        raise ValueError('Keep the matched objectives and fixed final selection')
    if min(config[k] for k in ('steps', 'batch_size', 'log_interval', 'eval_batch_size')) < 1:
        raise ValueError('Positive fixed budgets required')
    if any(int(n) != n or n < 2 for n in config['eval_samples']):
        raise ValueError('Energy evaluation needs at least two IID samples')
    return config


def experiment_sources():
    from . import a7_distribution_teacher
    sources = legacy_source_hashes()
    paths = [Path(__file__), Path(a7_distribution_teacher.__file__),
        Path(__file__).resolve().parents[1]/'scripts/run_a7_distribution_teacher.py']
    # Follow local imports transitively, including function-local imports. This
    # closes legacy receipt gaps rather than assuming direct imports suffice.
    pending = [Path(__file__), Path(a7_distribution_teacher.__file__)]
    seen = set()
    package = Path(__file__).resolve().parent
    while pending:
        path = pending.pop().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                modules = [node.module] if node.module else [alias.name for alias in node.names]
                pending.extend(package.joinpath(*module.split('.')).with_suffix('.py') for module in modules)
    config_path = package.parent/'configs/retail_a7_distribution_teacher.json'
    return {**sources, **{str(p): sha256(p) for p in [*paths, *sorted(seen), config_path]}}


def make_teacher(data, config, device='cpu'):
    return A7DistributionTeacher(data['state']['token_dim'], data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'], **config['teacher']).to(device)


def prepare_distribution_run(data_dir, output, config, heldout_queries, split_manifest=None):
    data_dir, output = Path(data_dir), Path(output)
    data = load_data(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'code': experiment_sources(),
        'data_reference': str(data_dir), 'data_sha256': sha256(data_dir/'data.pt'),
        'data_manifest_sha256': sha256(data_dir/'manifest.json'),
        'split_reference': str(split_manifest) if split_manifest else None,
        'split_sha256': sha256(split_manifest) if split_manifest else None,
        'heldout_reference': str(heldout_queries), 'heldout_sha256': sha256(heldout_queries),
        'training_state_digest': calibration_digest(data['state']),
        'fit_entities': len(data['records']),
        'comparison': 'same nonlinear stochastic architecture, metadata-free encoder, common init/data/2IID noises; objective only differs',
        'evaluation_scope': 'oracle reconstruction with true N/face/calendar; no native fidelity claim',
        'reference_teacher': 'old V1 metadata/count teacher is frozen context, not an objective-only control'}
    path = output/'manifest.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('Distribution teacher experiment identity changed')
    write_json(path, identity)
    initialization = output/'initialization.pt'
    if not initialization.exists():
        seed_all(config['seed'])
        teacher = make_teacher(data, config)
        atomic_save({'model': teacher.state_dict(), 'digest': tensor_digest(teacher.state_dict()),
            'config': config, 'data_sha256': identity['data_sha256']}, initialization)
    saved = torch.load(initialization, map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['data_sha256'] != identity['data_sha256']:
        raise ValueError('Shared initialization belongs to a different experiment')
    return data


def paired_step_inputs(data, config, step, device):
    # Separate stateless streams guarantee resume and objective-independent draws.
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], step, 0]))
    indices = rng.integers(len(data['windows']), size=config['batch_size'])
    batch = collate_a7([data['windows'][int(i)] for i in indices], device)
    generator = torch.Generator(device=device).manual_seed(config['seed']+104729*step)
    global_noise = torch.randn(2, len(indices), config['teacher']['code_dim'],
        generator=generator, device=device)
    local_noise = torch.randn(2, *batch['valid'].shape, data['state']['token_dim'],
        generator=generator, device=device)
    receipt = {'indices': indices.tolist(), 'noise_digest': tensor_digest(
        {'global': global_noise, 'local': local_noise})}
    return batch, global_noise, local_noise, receipt


@torch.no_grad()
def encode_codes(teacher, windows, batch_size=32, device='cpu'):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Code caching requires a frozen teacher')
    return torch.cat([teacher.encode(collate_a7(windows[i:i+batch_size], device)).cpu()
        for i in range(0, len(windows), batch_size)])


def train_distribution_teacher(data, output, config, objective, device='cuda', stop_after=None):
    if objective not in config['objectives']:
        raise ValueError('Unknown predeclared objective')
    output = Path(output)
    arm = output/objective.upper()
    arm.mkdir(exist_ok=True)
    manifest = read_json(output/'manifest.json')
    initial = torch.load(output/'initialization.pt', map_location='cpu', weights_only=False)
    seed_all(config['seed'])
    teacher = make_teacher(data, config, device)
    teacher.load_state_dict(initial['model'])
    if (arm/'final.pt').exists():
        saved = torch.load(arm/'final.pt', map_location=device, weights_only=False)
        if saved['config'] != config or saved['step'] != config['steps'] or saved['objective'] != objective:
            raise ValueError('Completed checkpoint identity differs')
        teacher.load_state_dict(saved['model'])
        return teacher.freeze()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    first, elapsed, history = 1, 0., []
    if (arm/'latest.pt').exists():
        saved = torch.load(arm/'latest.pt', map_location=device, weights_only=False)
        if saved['config'] != config or saved['objective'] != objective or saved['initialization_digest'] != initial['digest']:
            raise ValueError('Resume identity differs')
        teacher.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        first, elapsed, history = saved['step']+1, saved['elapsed_seconds'], saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    started = time.perf_counter()
    last_step = min(config['steps'], stop_after) if stop_after is not None else config['steps']
    for step in range(first, last_step+1):
        teacher.train()
        batch, global_noise, local_noise, receipt = paired_step_inputs(data, config, step, device)
        seed_all(config['seed']+104729*step+1)
        loss, stats = teacher.loss(batch, objective=objective,
            global_noise_pair=global_noise, local_noise_pair=local_noise)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite distribution teacher objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(teacher.parameters(), config['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite distribution teacher gradient')
        optimizer.step()
        # Compact receipt every update proves the complete paired input schedule.
        history.append({'step': step, 'loss': float(loss.detach()), **receipt})
        if step == 1 or step % config['log_interval'] == 0 or step == last_step:
            atomic_save({'model': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'objective': objective, 'step': step, 'history': history,
                'initialization_digest': initial['digest'], 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'elapsed_seconds': elapsed+time.perf_counter()-started}, arm/'latest.pt')
            write_json(arm/'progress.json', {'step': step, 'budget': config['steps'], 'diagnostics': stats,
                'gradient_norm': float(norm), 'elapsed_seconds': elapsed+time.perf_counter()-started})
            print(f'{objective.upper()} teacher {step}/{config["steps"]} loss={float(loss):.6f}', flush=True)
    if last_step < config['steps']:
        return None
    teacher.freeze()
    raw_codes = encode_codes(teacher, data['windows'], config['batch_size'], device)
    teacher.set_code_calibration(raw_codes.to(device))
    final = {'model': teacher.state_dict(), 'config': config, 'objective': objective,
        'step': config['steps'], 'selection': config['selection'], 'calibrated_after_freeze': True,
        'initialization_digest': initial['digest'], 'data_sha256': manifest['data_sha256'],
        'training_state_digest': manifest['training_state_digest'], 'experiment_code': manifest['code'],
        'elapsed_seconds': elapsed+time.perf_counter()-started}
    atomic_save(final, arm/'final.pt')
    normalized = encode_codes(teacher, data['windows'], config['batch_size'], device)
    atomic_save({'raw_codes': raw_codes, 'codes': normalized,
        'teacher_sha256': sha256(arm/'final.pt'), 'data_sha256': manifest['data_sha256'],
        'fit_split': 'training windows only'}, arm/'train_codes.pt')
    write_json(arm/'history.json', history)
    write_json(arm/'complete.json', {k: final[k] for k in ('step', 'selection', 'objective',
        'calibrated_after_freeze', 'initialization_digest', 'elapsed_seconds')})
    return teacher


def normalized_sample_energy(samples, batch):
    """Unbiased sample energy score per query, normalized by intrinsic sqrt-dof."""
    target = batch['tokens']
    active, event = batch['active'], batch['event_valid']
    def vectors(values):
        timing = project_clr(values[..., 0], active)
        marks = torch.where(event[..., None], values[..., 1:], torch.zeros_like(values[..., 1:]))
        return torch.cat([timing, marks.flatten(-2)], -1)
    prediction = vectors(samples).transpose(0, 1)
    truth = vectors(target)
    distance = torch.linalg.vector_norm(prediction-truth[:, None], dim=-1).mean(-1)
    count = samples.shape[0]
    pair_distances = torch.cdist(prediction, prediction)
    pair_distances.diagonal(dim1=-2, dim2=-1).zero_()
    pair = pair_distances.sum((1, 2))/(count*(count-1))
    dof = (active.sum(-1)-1).clamp_min(0)+event.sum(-1)*(target.shape[-1]-1)
    return (distance-.5*pair)/dof.clamp_min(1).sqrt()


def raw_summary(arrays, names):
    data = np.concatenate(arrays) if arrays else np.empty((0, len(names)))
    result = {}
    for name, column in zip(names, data.T):
        values = column[np.isfinite(column)]
        result[name] = {'finite_rows': len(values), 'std': float(values.std()) if len(values) else None,
            'p99': float(np.quantile(values, .99)) if len(values) else None,
            'zero_fraction': float((values == 0).mean()) if len(values) else None,
            'mean': float(values.mean()) if len(values) else None}
    return result


@torch.no_grad()
def evaluate_distribution_teacher(teacher, data, heldout, output, config, prep, device='cuda'):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Freeze and train-calibrate teacher before evaluation')
    if isinstance(heldout, (str, Path)):
        heldout = torch.load(heldout, map_location='cpu', weights_only=False)
    if calibration_digest(data['state']) != calibration_digest(heldout['state']):
        raise ValueError('Heldout calibration differs from training')
    if heldout.get('fit_split') not in (None, 'train only'):
        raise ValueError('Heldout query calibration must be training-only')
    if 'preprocessor_hash' in heldout and digest(prep.state_dict()) != heldout['preprocessor_hash']:
        raise ValueError('Raw inverse preprocessor differs from training calibration')
    before = tensor_digest(teacher.state_dict())
    queries = heldout['queries']
    size = config['eval_batch_size']
    proper = encode_codes(teacher, queries, size, device)
    mean_code = torch.zeros_like(proper)  # normalized TRAIN-mean code
    permutation = np.random.default_rng(config['eval_seed']).permutation(len(queries))
    zero_mark_codes = []
    for start in range(0, len(queries), size):
        batch = collate_a7(queries[start:start+size], device)
        batch['tokens'] = batch['tokens'].clone()
        batch['tokens'][..., 1:] = 0
        zero_mark_codes.append(teacher.encode(batch).cpu())
    controls = {'true': proper, 'train_mean': mean_code, 'shuffle': proper[permutation],
        'marks_zero': torch.cat(zero_mark_codes)}
    metrics: dict[str, dict[str, list[float]]] = {str(n): {key: [] for key in controls} for n in config['eval_samples']}
    raw: dict[str, dict[str, list[np.ndarray]]] = {str(n): {key: [] for key in controls} for n in config['eval_samples']}
    target_raw = []
    names = list(prep.numeric_names)
    def native_values(tokens):
        decoded = inverse_marks(tokens, prep, data['state']['mark_token_state'])
        values = decoded['x_num'].copy()
        for j, name in enumerate(names):
            if name.startswith('log_'):
                values[:, j] = np.expm1(values[:, j])
            values[decoded['mask'][:, j] == 0, j] = np.nan
        return values
    for start in range(0, len(queries), size):
        batch = collate_a7(queries[start:start+size], device)
        target_raw.append(native_values(batch['tokens'][..., 1:][batch['event_valid']]))
        generator = torch.Generator(device=device).manual_seed(config['eval_seed']+start)
        samples_by_control: dict[str, list[torch.Tensor]] = {key: [] for key in controls}
        for _ in range(max(config['eval_samples'])):
            global_noise = torch.randn(len(batch['count']), teacher.code_dim, generator=generator, device=device)
            local_noise = torch.randn(*batch['valid'].shape, teacher.token_dim, generator=generator, device=device)
            for key, code in controls.items():
                prediction = teacher.sample_reconstruction(code[start:start+len(batch['count'])].to(device),
                    batch, global_noise, local_noise)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError('Nonfinite oracle reconstruction')
                samples_by_control[key].append(prediction)
        for key, draws in samples_by_control.items():
            for n in config['eval_samples']:
                samples = torch.stack(draws[:n])
                metrics[str(n)][key].extend(normalized_sample_energy(samples, batch).cpu().tolist())
                for draw in draws[:n]:
                    raw[str(n)][key].append(native_values(draw[..., 1:][batch['event_valid']]))
    entity_ids = [query.get('entity_id', query['entity']) for query in queries]
    scores: dict[str, Any] = {}
    for n, arms in metrics.items():
        scores[n] = {}
        for key, values in arms.items():
            groups: dict[Any, list[float]] = {}
            for entity, value in zip(entity_ids, values):
                groups.setdefault(entity, []).append(value)
            scores[n][key] = {'entity_balanced_energy': float(np.mean([np.mean(v) for v in groups.values()])),
                'per_query_energy': values,
                'raw_sample_mixture': raw_summary(raw[n][key], names)}
        scores[n]['code_use_entity_paired'] = {key: paired_entity_bootstrap(entity_ids,
            arms['true'], arms[key]) for key in controls if key != 'true'}
    report = {'scope': 'ORACLE stochastic reconstruction: actual N/face/calendar; not native generation fidelity',
        'raw_scope': 'overlapping validation query rows pooled across iid reconstructions; not independent population rows',
        'raw_transform': 'rowwise inverse training QT, then expm1 for schema log_ fields; never inverse of decoder mean',
        'samples': config['eval_samples'], 'queries': len(queries), 'entities': len(set(entity_ids)),
        'common_noise_across_controls': True, 'target_raw': raw_summary(target_raw, names),
        'raw_code_statistics': raw_code_statistics(proper, teacher), 'scores': scores,
        'teacher_state_unchanged': tensor_digest(teacher.state_dict()) == before}
    if not report['teacher_state_unchanged']:
        raise RuntimeError('Oracle evaluation mutated frozen teacher')
    write_json(Path(output)/'oracle_reconstruction.json', report)
    return report
