"""Fixed-budget A7 teacher and matched auxiliary-on/off experiments.

Future teacher embeddings are frozen supervision only. The generator's forward
and native sampling interfaces never receive a true future code or cardinality.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .a7_data import build_suffix_window, collate_a7, inverse_marks
from .a7_model import A7Generator
from .a7_spacing import decode_spacing
from .a7_teacher import A7FutureTeacher
from .experiment import read_json, seed_all, write_json
from .paper_data import SCHEMAS, load_prepared_paper, sha256


DEFAULT = {'model_name': 'A7 observed-past and noisy-prefix predictive diffusion',
    'training_seed': 9, 'generation_seeds': [1, 2, 3],
    'teacher_steps': 5000, 'teacher_batch_size': 32, 'teacher_learning_rate': .0003,
    'generator_steps': 20000, 'batch_size': 16, 'learning_rate': .0003,
    'weight_decay': .01, 'gradient_clip': 1., 'log_interval': 100,
    'code_noise_std': .03, 'code_cache_batch_size': 32,
    'sample_steps': 50, 'generation_batch_size': 8, 'max_generated_events': 4096,
    'arms': ['A7_AUX_ON', 'A7_AUX_OFF'], 'aux_weights': {'A7_AUX_ON': 1., 'A7_AUX_OFF': 0.},
    'teacher': {'code_dim': 64, 'width': 128, 'layers': 2, 'heads': 4, 'count_components': 4},
    'model': {'code_dim': 64, 'width': 128, 'layers': 4, 'heads': 4, 'hidden': 128,
              'components': 8, 'diffusion_steps': 100},
    'selection': 'fixed_final_update_no_validation'}


def a7_config(overrides=None):
    config = copy.deepcopy(DEFAULT)
    for key, value in (overrides or {}).items():
        if key in ('teacher', 'model', 'aux_weights'):
            config[key].update(value)
        else:
            config[key] = copy.deepcopy(value)
    if not config['arms'] or not set(config['arms']).issubset({'A7_AUX_ON', 'A7_AUX_OFF'}):
        raise ValueError('Declare only matched A7 auxiliary arms')
    if config['selection'] != 'fixed_final_update_no_validation':
        raise ValueError('Outcome-dependent checkpoint selection is not permitted')
    if config['teacher']['code_dim'] != config['model']['code_dim']:
        raise ValueError('Teacher and student target dimensions differ')
    if config['code_noise_std'] != .03:
        raise ValueError('Keep the predeclared .03 teacher-code smoothing')
    return config


def source_hashes():
    from . import a7_data, a7_model, a7_spacing, a7_teacher, a7_teacher_audit
    from . import a6_data, a6_model, stable_diffusion, paper_data, quantile, data as common_data, experiment
    return {path.name: sha256(path) for path in [Path(__file__), Path(a7_data.__file__),
        Path(a7_model.__file__), Path(a7_spacing.__file__), Path(a7_teacher.__file__),
        Path(a7_teacher_audit.__file__), Path(a6_data.__file__), Path(a6_model.__file__),
        Path(stable_diffusion.__file__), Path(paper_data.__file__), Path(quantile.__file__),
        Path(common_data.__file__), Path(experiment.__file__),
        Path(__file__).resolve().parents[1]/'scripts/run_retail_a7.py']}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def tensor_digest(tensors):
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(name.encode())
        result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_data(data_dir):
    data_dir = Path(data_dir)
    receipt = read_json(data_dir/'manifest.json')
    if sha256(data_dir/'data.pt') != receipt['data_sha256']:
        raise ValueError('A7 training bank changed')
    data = torch.load(data_dir/'data.pt', map_location='cpu', weights_only=False)
    if data['manifest'] != {key: receipt[key] for key in data['manifest']}:
        raise ValueError('A7 training manifest differs from the bank')
    return data


def prepare_run(data_dir, output, config, heldout_queries=None):
    data_dir, output = Path(data_dir), Path(output)
    data = load_data(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'code': source_hashes(), 'data_reference': str(data_dir),
        'data_sha256': sha256(data_dir/'data.pt'), 'data_manifest': data['manifest'],
        'fit_entities': len(data['records']), 'native_teacher_code_input': False,
        'auxiliary_only_difference': 'prediction NLL weight 1 versus 0; common architecture, initialization and diffusion noise',
        'heldout_queries_reference': str(heldout_queries) if heldout_queries else None,
        'heldout_queries_sha256': sha256(heldout_queries) if heldout_queries else None}
    path = output/'manifest.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('A7 experiment identity changed; preserve this run')
    write_json(path, identity)
    return data


def make_teacher(data, config, device='cpu'):
    return A7FutureTeacher(data['state']['token_dim'], data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'], **config['teacher']).to(device)


def make_model(data, config, device='cpu'):
    return A7Generator(data['state']['token_dim'], data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'], time_scale=data['state']['time_scale'],
        **config['model']).to(device)


def _snapshot(model, optimizer, config, step, rng, elapsed, history, path, **extra):
    write_json(path.parent/'history.json', history)
    atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
        'step': step, 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'elapsed_seconds': elapsed, **extra}, path)


def _restore(saved, model, optimizer, rng):
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if torch.cuda.is_available() and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda_rng']])


def train_teacher(data, output, config, device='cuda'):
    output = Path(output)
    directory = output/'teacher'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'complete.json').exists():
        saved = torch.load(directory/'final.pt', map_location=device, weights_only=False)
        if saved['config'] != config or saved['step'] != config['teacher_steps'] or not saved['calibrated_train_only']:
            raise ValueError('Completed teacher identity differs')
        model = make_teacher(data, config, device)
        model.load_state_dict(saved['model'])
        return model.freeze()
    seed_all(config['training_seed'])
    model = make_teacher(data, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['teacher_learning_rate'], weight_decay=config['weight_decay'])
    seed_all(config['training_seed'])
    rng = np.random.default_rng(config['training_seed'])
    history, first, previous_seconds = [], 1, 0.
    latest = directory/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['config'] != config or saved['part'] != 'teacher':
            raise ValueError('Teacher resume belongs to a different run')
        _restore(saved, model, optimizer, rng)
        first, previous_seconds = saved['step']+1, saved['elapsed_seconds']
        history = read_json(directory/'history.json')
    start = time.perf_counter()
    for step in range(first, config['teacher_steps']+1):
        model.train()
        indices = rng.integers(len(data['windows']), size=config['teacher_batch_size'])
        batch = collate_a7([data['windows'][int(index)] for index in indices], device)
        loss, stats = model.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite A7 teacher loss at update {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite A7 teacher gradient')
        optimizer.step()
        if step == 1 or step % config['log_interval'] == 0 or step == config['teacher_steps']:
            elapsed = previous_seconds+time.perf_counter()-start
            history.append({'step': step, 'training_loss': float(loss.detach()), 'diagnostics': stats,
                            'gradient_norm': float(norm), 'elapsed_seconds': elapsed})
            _snapshot(model, optimizer, config, step, rng, elapsed, history, latest, part='teacher')
            print(f'A7 teacher {step}/{config["teacher_steps"]} loss={float(loss):.6f} elapsed={elapsed:.1f}s', flush=True)
    model.freeze()
    with torch.no_grad():
        raw_codes = torch.cat([model.encode(collate_a7(data['windows'][start:start+config['teacher_batch_size']], device)).cpu()
                               for start in range(0, len(data['windows']), config['teacher_batch_size'])])
        model.set_code_calibration(raw_codes.to(device))
    atomic_save({'model': model.state_dict(), 'config': config, 'part': 'teacher',
        'step': config['teacher_steps'], 'training_seed': config['training_seed'],
        'selection': config['selection'], 'calibrated_train_only': True,
        'fit_entities': len(data['records']), 'data_sha256': read_json(output/'manifest.json')['data_sha256']}, directory/'final.pt')
    atomic_save({'raw_codes': raw_codes, 'teacher_checkpoint_sha256': sha256(directory/'final.pt')}, directory/'train_raw_codes.pt')
    write_json(directory/'complete.json', {'updates': config['teacher_steps'], 'frozen': True,
        'calibrated_after_freeze': True, 'calibration_fit': 'training windows only',
        'fit_entities': len(data['records']), 'elapsed_seconds': previous_seconds+time.perf_counter()-start,
        'checkpoint_sha256': sha256(directory/'final.pt'), 'parameters': sum(p.numel() for p in model.parameters())})
    return model


def run_teacher_gate(teacher, data, heldout_queries, output, config, device, prep=None):
    from .a7_teacher_audit import audit_teacher
    if heldout_queries is None:
        raise ValueError('Teacher gate requires a separately encoded held-out query bundle')
    return audit_teacher(teacher, data, heldout_queries, Path(output)/'teacher_audit',
                         device=device, batch_size=config['teacher_batch_size'], prep=prep)


def cache_suffix_codes(teacher, data, output, config, device='cuda'):
    output = Path(output)
    location = output/'teacher/suffix_codes.pt'
    teacher_sha = sha256(output/'teacher/final.pt')
    data_sha = read_json(output/'manifest.json')['data_sha256']
    if location.exists():
        cache = torch.load(location, map_location='cpu', weights_only=False)
        if cache['teacher_checkpoint_sha256'] != teacher_sha or cache['data_sha256'] != data_sha:
            raise ValueError('Frozen teacher suffix cache identity differs')
        return cache['codes']
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Freeze the calibrated teacher before suffix caching')
    codes = [torch.zeros(item['count']+1, config['teacher']['code_dim']) for item in data['windows']]
    jobs = sorted([(item['count']-k, index, k) for index, item in enumerate(data['windows'])
                   for k in range(item['count']+1)], reverse=True)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(jobs), config['code_cache_batch_size']):
            selected = jobs[start:start+config['code_cache_batch_size']]
            windows = [build_suffix_window(data['windows'][index], k) for _, index, k in selected]
            encoded = teacher.encode(collate_a7(windows, device)).cpu()
            for row, (_, index, k) in enumerate(selected):
                codes[index][k] = encoded[row]
            if start % (100*config['code_cache_batch_size']) == 0:
                print(f'A7 teacher suffix cache {start}/{len(jobs)}', flush=True)
    cache = {'codes': codes, 'teacher_checkpoint_sha256': teacher_sha, 'data_sha256': data_sha,
             'targets': 'independently renormalized actual remaining windows, positions0..N inclusive',
             'rows': len(jobs), 'code_dim': config['teacher']['code_dim']}
    atomic_save(cache, location)
    write_json(output/'teacher/suffix_cache.json', {'rows': len(jobs), 'windows': len(codes),
        'elapsed_seconds': time.perf_counter()-started, 'teacher_checkpoint_sha256': teacher_sha,
        'data_sha256': data_sha, 'cache_sha256': sha256(location), 'true_future_input_to_generator': False})
    return codes


def dense_targets(codes, indices, length, device, generator, noise_std):
    values = torch.zeros(len(indices), length, codes[0].shape[-1], device=device)
    for row, index in enumerate(indices):
        value = codes[int(index)]
        values[row, :len(value)] = value.to(device)
    noise = torch.randn(values.shape, device=device, dtype=values.dtype, generator=generator)
    return values+float(noise_std)*noise


def train_generators(data, output, config, codes, device='cuda', arms=None):
    output = Path(output)
    gate = read_json(output/'teacher_audit/teacher_audit.json')
    if gate['decision'] != 'PASS':
        raise RuntimeError('Teacher mark-information gate did not pass; review diagnostics before generator training')
    initial_path = output/'generator_initial.pt'
    if not initial_path.exists():
        seed_all(config['training_seed'])
        initial = make_model(data, config, device)
        atomic_save({'model': initial.state_dict(), 'config': config,
            'initialization': 'from scratch common seed; no true future teacher condition'}, initial_path)
    initial = torch.load(initial_path, map_location=device, weights_only=False)
    if initial['config'] != config:
        raise ValueError('Common generator initialization identity changed')
    initialization_digest = tensor_digest(initial['model'])
    protected = {'teacher_checkpoint_sha256': sha256(output/'teacher/final.pt'),
                 'suffix_cache_sha256': sha256(output/'teacher/suffix_codes.pt')}
    for arm in arms or config['arms']:
        if arm not in config['arms']:
            raise ValueError('Undeclared A7 arm')
        directory = output/arm
        directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():
            done = read_json(directory/'complete.json')
            if done['updates'] != config['generator_steps'] or done['protected_teacher'] != protected:
                raise ValueError('Completed A7 generator identity differs')
            continue
        seed_all(config['training_seed'])
        model = make_model(data, config, device)
        model.load_state_dict(initial['model'])
        if tensor_digest(model.state_dict()) != initialization_digest:
            raise RuntimeError('Auxiliary arms did not start with identical model bytes')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        seed_all(config['training_seed'])
        rng = np.random.default_rng(config['training_seed'])
        aux_rng = torch.Generator(device=device).manual_seed(config['training_seed']+1000)
        history, first, previous_seconds = [], 1, 0.
        latest = directory/'latest.pt'
        if latest.exists():
            saved = torch.load(latest, map_location=device, weights_only=False)
            if saved['config'] != config or saved['arm'] != arm or saved['protected_teacher'] != protected:
                raise ValueError('A7 generator resume identity differs')
            _restore(saved, model, optimizer, rng)
            aux_rng.set_state(saved['aux_rng'].cpu())
            first, previous_seconds = saved['step']+1, saved['elapsed_seconds']
            history = read_json(directory/'history.json')
        started = time.perf_counter()
        for step in range(first, config['generator_steps']+1):
            model.train()
            indices = rng.integers(len(data['windows']), size=config['batch_size'])
            batch = collate_a7([data['windows'][int(index)] for index in indices], device)
            targets = dense_targets(codes, indices, batch['tokens'].shape[1], device, aux_rng, config['code_noise_std'])
            steps = torch.randint(model.diffusion_steps, (len(indices),), device=device)
            noise = torch.randn_like(batch['tokens'])
            loss, stats = model.loss(batch, targets, targets[:, 0], config['aux_weights'][arm], noise=noise, steps=steps)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite A7 loss {arm} update{step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('Nonfinite A7 generator gradient')
            optimizer.step()
            if step == 1 or step % config['log_interval'] == 0 or step == config['generator_steps']:
                elapsed = previous_seconds+time.perf_counter()-started
                signature = tensor_digest({'indices': torch.as_tensor(indices), 'steps': steps, 'noise': noise})
                history.append({'step': step, 'training_loss': float(loss.detach()), 'diagnostics': stats,
                    'gradient_norm': float(norm), 'elapsed_seconds': elapsed, 'diffusion_noise_signature': signature})
                _snapshot(model, optimizer, config, step, rng, elapsed, history, latest,
                    part='generator', arm=arm, protected_teacher=protected, aux_rng=aux_rng.get_state())
                print(f'A7 {arm} {step}/{config["generator_steps"]} loss={float(loss):.6f} elapsed={elapsed:.1f}s', flush=True)
        atomic_save({'model': model.state_dict(), 'config': config, 'part': 'generator', 'arm': arm,
            'step': config['generator_steps'], 'training_seed': config['training_seed'],
            'selection': config['selection'], 'initialization_digest': initialization_digest,
            'protected_teacher': protected}, directory/'final.pt')
        write_json(directory/'complete.json', {'updates': config['generator_steps'], 'arm': arm,
            'aux_weight': config['aux_weights'][arm], 'parameters': model.parameter_counts(),
            'elapsed_seconds': previous_seconds+time.perf_counter()-started,
            'initialization_digest': initialization_digest, 'protected_teacher': protected,
            'checkpoint_sha256': sha256(directory/'final.pt')})
        if protected != {'teacher_checkpoint_sha256': sha256(output/'teacher/final.pt'),
                         'suffix_cache_sha256': sha256(output/'teacher/suffix_codes.pt')}:
            raise RuntimeError('Frozen teacher or its cached targets changed during generator training')
    if all((output/arm/'complete.json').exists() for arm in config['arms']):
        write_json(output/'training_complete.json', {'arms': config['arms'],
            'updates_each': config['generator_steps'], 'common_initialization_digest': initialization_digest,
            'teacher_frozen': True, 'future_codes_used_as_generator_input': False})


@torch.no_grad()
def generate(data, prepared, output, config, device='cuda', arms=None, queries=None,
             population='train_bos'):
    output = Path(output)
    _, prep, _ = load_prepared_paper(prepared)
    if queries is None:
        entity_ids = read_json(Path(prepared)/'parents.json')
        end = data['state']['mark_token_state']['observation_end']
        queries = [{'attrs': attrs, 'anchor': 0., 'delta': end,
            'past_tokens': torch.empty(0, data['state']['token_dim']), 'entity_id': entity,
            'query_id': index} for index, (attrs, entity) in enumerate(zip(data['attrs'], entity_ids))]
    schema = SCHEMAS['online_retail']
    for arm in arms or config['arms']:
        checkpoint = output/arm/'final.pt'
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['config'] != config or saved['step'] != config['generator_steps']:
            raise ValueError('Generation requires the declared fixed final model')
        model = make_model(data, config, device)
        model.load_state_dict(saved['model']); model.eval()
        for seed in config['generation_seeds']:
            location = output/'generating'/arm/population/str(seed)
            location.mkdir(parents=True, exist_ok=True)
            receipt_path = location/'generation.json'
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt['checkpoint_sha256'] != sha256(checkpoint) or receipt['child_sha256'] != sha256(location/'child.csv'):
                    raise ValueError('Existing A7 output identity differs')
                continue
            seed_all(seed)
            frames, counts = [], []
            started = time.perf_counter()
            for start in range(0, len(queries), config['generation_batch_size']):
                chosen = queries[start:start+config['generation_batch_size']]
                attrs = torch.stack([item['attrs'] for item in chosen]).to(device)
                anchors = torch.tensor([item['anchor'] for item in chosen], dtype=torch.float64, device=device)
                horizons = torch.tensor([item['delta'] for item in chosen], dtype=torch.float64, device=device)
                width = max(len(item['past_tokens']) for item in chosen)
                past = torch.zeros(len(chosen), width, data['state']['token_dim'], device=device)
                past_valid = torch.zeros(len(chosen), width, dtype=torch.bool, device=device)
                for row, item in enumerate(chosen):
                    size = len(item['past_tokens']); past[row, :size] = item['past_tokens'].to(device); past_valid[row, :size] = True
                result = model.sample_paths(attrs, anchors, horizons, sample_steps=config['sample_steps'],
                    max_events=config['max_generated_events'], past_tokens=past, past_valid=past_valid)
                if bool(result['cap_hit'].any()):
                    raise RuntimeError('A7 cap failure; no count clipping or resampling')
                for row, item in enumerate(chosen):
                    count = int(result['count'][row])
                    tokens = result['tokens'][row, :count+1]
                    timing = decode_spacing(tokens[:, 0].double()*data['state']['time_scale'],
                        result['active'][row, :count+1], float(item['anchor']), float(item['delta']))
                    marks = inverse_marks(tokens[:count, 1:], prep, data['state']['mark_token_state'])
                    columns = {'entity': np.full(count, item['entity_id']), 't_offset': timing['times'].cpu().numpy()}
                    if population == 'heldout_queries':
                        columns['query_index'] = np.full(count, start+row)
                    for j, name in enumerate(schema['numeric']):
                        columns[name] = np.where(marks['mask'][:, j] >= .5, marks['x_num'][:, j], np.nan)
                    for j, name in enumerate(schema['categorical']):
                        encoded = marks['x_cat'][:, j]
                        columns[name] = np.asarray([float(prep.vocabularies[j][int(value)-1])
                            if 1 <= value <= len(prep.vocabularies[j]) else np.nan for value in encoded])
                    frames.append(pd.DataFrame(columns))
                    counts.append({'query_index': start+row, 'entity': item['entity_id'], 'count': count,
                                   'anchor': item['anchor'], 'delta': item['delta']})
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(location/'child.csv', index=False)
            pd.DataFrame(counts).to_csv(location/'counts.csv', index=False)
            write_json(receipt_path, {'arm': arm, 'population': population, 'generation_seed': seed,
                'queries': len(queries), 'events': len(frame), 'empty_queries': sum(item['count'] == 0 for item in counts),
                'checkpoint_sha256': sha256(checkpoint), 'child_sha256': sha256(location/'child.csv'),
                'elapsed_seconds': time.perf_counter()-started, 'count_or_face_or_teacher_oracle': False,
                'scope': 'conditional heldout future windows' if population == 'heldout_queries' else 'declared static-conditioned BOS population',
                'native_order': True, 'posthoc_time_mark_or_cumulative_correction': False})
            print(f'A7 {arm} {population} seed{seed}: {len(frame)} events', flush=True)
