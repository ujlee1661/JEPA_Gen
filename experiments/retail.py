"""JETS on a prepared temporal-tabular bank.

The bank, frozen teacher caches, preprocessors and checkpoints are external
artifacts. Only load trusted torch files: pickle-based torch.load can run code.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from experiments.data import dense_targets, json_digest, load_bank, load_cache, read_json, sha256, write_json
from jets.data import collate_windows, inverse_marks
from jets.model import JETSGenerator
from jets.preprocessing import QuantilePreprocessor
from jets.spacing import decode_spacing


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(data, config, device):
    state = data['state']
    model = JETSGenerator(state['token_dim'], data['attrs'].shape[1],
        state['mark_token_state']['observation_end'], time_scale=state['time_scale'],
        **config['model']).to(device)
    model.stochastic_condition = True
    return model


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*(root / 'jets').glob('*.py'), *(root / 'experiments').glob('*.py')])
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def atomic_torch_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def train(args):
    bank_path, data = load_bank(args.bank)
    config = read_json(args.config)
    if args.steps is not None:
        config['generator_steps'] = args.steps
    if config['generator_steps'] < 1 or config['batch_size'] < 1:
        raise ValueError('Positive training budget and batch size required')
    time_codes = load_cache(args.time_cache, bank_path, data['windows'], config['model']['code_dim'], args.time_teacher)
    feature_codes = load_cache(args.feature_cache, bank_path, data['windows'], config['model']['code_dim'], args.feature_teacher)
    input_paths = (bank_path, args.config, args.time_cache, args.time_teacher, args.feature_cache, args.feature_teacher, args.initial)
    hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    identity = {'config': config, 'input_sha256': hashes, 'source_sha256': source_hashes(), 'feature_weight': args.feature_weight,
        'stochastic_condition': True, 'native_future_teacher_inputs': False,
        'scope': 'new JETS run with externally prepared windows and frozen teacher caches; not the original run receipt'}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError('Output namespace belongs to a different experiment')
    if not manifest.exists():
        write_json(manifest, identity)
    if (out / 'final.pt').exists():
        raise ValueError('Completed output exists; choose a new output directory')
    seed = int(config['training_seed'])
    seed_all(seed)
    model = build_model(data, config, args.device)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    model.load_state_dict(initial.get('model', initial), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    seed_all(seed)
    rng = np.random.default_rng(seed)
    time_rng = torch.Generator(device=args.device).manual_seed(seed + 1000)
    feature_rng = torch.Generator(device=args.device).manual_seed(seed + 2000)
    latest = out / 'latest.pt'
    first, previous_elapsed = 1, 0.
    if latest.exists():
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        time_rng.set_state(saved['time_rng'].cpu())
        feature_rng.set_state(saved['feature_rng'].cpu())
        first, previous_elapsed = saved['step'] + 1, saved['elapsed_seconds']
    start = time.perf_counter()
    for step in range(first, int(config['generator_steps']) + 1):
        model.train()
        indices = rng.integers(len(data['windows']), size=int(config['batch_size']))
        batch = collate_windows([data['windows'][int(i)] for i in indices], args.device)
        targets = dense_targets(time_codes, indices, batch['tokens'].shape[1], args.device, time_rng, config['code_noise_std'])
        features = dense_targets(feature_codes, indices, batch['tokens'].shape[1], args.device, feature_rng, config['code_noise_std'])
        steps = torch.randint(model.diffusion_steps, (len(indices),), device=args.device)
        noise = torch.randn_like(batch['tokens'])
        loss, stats = model.loss(batch, targets, targets[:, 0], 1., feature_teacher_codes=features,
            feature_aux_weight=args.feature_weight, noise=noise, steps=steps)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Non-finite training gradient')
        optimizer.step()
        if step == 1 or step % int(config['log_interval']) == 0 or step == config['generator_steps']:
            elapsed = previous_elapsed + time.perf_counter() - start
            atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'identity': identity,
                'step': step, 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'time_rng': time_rng.get_state(), 'feature_rng': feature_rng.get_state(), 'elapsed_seconds': elapsed}, latest)
            print(json.dumps({'step': step, 'loss': float(loss.detach()), 'elapsed_seconds': elapsed, **stats}), flush=True)
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Training input changed during execution')
    if source_hashes() != identity['source_sha256']:
        raise ValueError('JETS source changed during execution')
    atomic_torch_save({'model': model.state_dict(), 'config': config, 'step': config['generator_steps'],
        'identity': identity, 'stochastic_condition': True}, out / 'final.pt')
    write_json(out / 'complete.json', {'updates': config['generator_steps'], 'checkpoint_sha256': sha256(out / 'final.pt')})


@torch.no_grad()
def generate(args):
    bank_path, data = load_bank(args.bank)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' not in checkpoint or 'config' not in checkpoint or int(checkpoint.get('step', 0)) < 1:
        raise ValueError('Generation requires a trained model checkpoint')
    config = checkpoint['config']
    model = build_model(data, config, args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    parents = read_json(args.parents)
    prep_state = read_json(args.preprocessor)
    if len(parents) != len(data['attrs']) or len(set(map(str, parents))) != len(parents):
        raise ValueError('One unique parent per static attribute row required')
    if prep_state.get('kind') != 'tabsyn_gaussian_quantile_v1':
        raise ValueError('Expected a saved Gaussian-quantile preprocessor')
    prepared_manifest = args.preprocessor.parent / 'manifest.json'
    if prepared_manifest.exists():
        manifest = read_json(prepared_manifest)
        if (manifest.get('preprocessor_hash') != json_digest(prep_state)
                or manifest.get('parent_hash') != json_digest(parents)):
            raise ValueError('Preprocessor or parent IDs differ from their prepared manifest')
    prep = QuantilePreprocessor.from_state_dict(prep_state)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('child.csv', 'counts.csv', 'generation.json')):
        raise ValueError('Output files already exist; choose a new output directory')
    seed_all(args.seed)
    rows, counts = [], []
    state = data['state']['mark_token_state']
    dim, end = data['state']['token_dim'], float(state['observation_end'])
    for start in range(0, len(parents), args.batch_size):
        stop = min(start + args.batch_size, len(parents))
        attrs = data['attrs'][start:stop].to(args.device)
        size = len(attrs)
        anchors = torch.zeros(size, dtype=torch.float64, device=args.device)
        horizons = torch.full((size,), end, dtype=torch.float64, device=args.device)
        past = torch.empty(size, 0, dim, dtype=torch.float32, device=args.device)
        past_valid = torch.empty(size, 0, dtype=torch.bool, device=args.device)
        paths = model.sample_paths(attrs, anchors, horizons, sample_steps=args.sample_steps,
            max_events=args.max_events, past_tokens=past, past_valid=past_valid)
        for index, entity in enumerate(parents[start:stop]):
            count = int(paths['count'][index])
            token = paths['tokens'][index, :count + 1]
            timing = decode_spacing(token[:, 0].double() * data['state']['time_scale'],
                paths['active'][index, :count + 1], 0., end)
            marks = inverse_marks(token[:count, 1:], prep, state)
            columns = {'entity': np.full(count, entity), 't_offset': timing['times'].cpu().numpy()}
            for j, name in enumerate(prep.numeric_names):
                columns[name] = np.where(marks['mask'][:, j] >= .5, marks['x_num'][:, j], np.nan)
            for j, name in enumerate(prep.categorical_names):
                vocabulary = prep.vocabularies[j]
                columns[name] = [float(vocabulary[int(v) - 1]) if 1 <= int(v) <= len(vocabulary) else np.nan
                                 for v in marks['x_cat'][:, j]]
            rows.append(pd.DataFrame(columns))
            counts.append({'query_index': start + index, 'entity': entity, 'count': count, 'anchor': 0., 'delta': end})
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(out / 'child.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'counts.csv', index=False)
    write_json(out / 'generation.json', {'generation_seed': args.seed, 'events': len(frame),
        'checkpoint_sha256': sha256(args.checkpoint), 'child_sha256': sha256(out / 'child.csv'),
        'bank_sha256': sha256(bank_path), 'preprocessor_sha256': sha256(args.preprocessor),
        'parents_sha256': sha256(args.parents),
        'counts_sha256': sha256(out / 'counts.csv'), 'sample_steps': args.sample_steps,
        'count_or_face_or_teacher_oracle': False, 'scope': 'BOS generation; native count and face',
        'warning': 'External artifacts must be trusted; this new receipt is not a legacy experiment receipt'})
    print(f'Generated {len(frame)} events for {len(parents)} parents: {out}', flush=True)


def evaluate(args):
    """Call a user-supplied trusted scorer without modifying its implementation."""
    spec = importlib.util.spec_from_file_location('jets_external_scorer', args.scorer)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load supplied scorer')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    real, synthetic = pd.read_csv(args.real), pd.read_csv(args.synthetic)
    result = {'ts': scorer.ts(real, synthetic), 'cs': scorer.cs(args.dataset, real, synthetic),
        'sd': scorer.sd(args.dataset, real, synthetic), 'lg': scorer.lg(args.dataset, real, synthetic),
        'joint_ff': scorer.jointff(real, synthetic)}
    clean = {name: float(value) if np.isfinite(value) else None for name, value in result.items()}
    print(json.dumps({'metrics': clean, 'scorer_sha256': sha256(args.scorer)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train', help='Train from an initial checkpoint and existing frozen teacher caches')
    for name in ('bank', 'config', 'time-cache', 'time-teacher', 'feature-cache', 'feature-teacher', 'initial', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, help='Explicit budget override; must not be presented as the original 20k experiment')
    p.add_argument('--feature-weight', type=float, choices=(0., 1.), default=1.)
    p.set_defaults(func=train)
    p = sub.add_parser('generate', help='Generate native paths from an existing trained checkpoint')
    for name in ('bank', 'checkpoint', 'preprocessor', 'parents', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--max-events', type=int, default=4096); p.set_defaults(func=generate)
    p = sub.add_parser('evaluate', help='Evaluate with an explicitly supplied trusted reference scorer')
    for name in ('real', 'synthetic', 'scorer'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dataset', default='online_retail'); p.set_defaults(func=evaluate)
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('batch size must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
