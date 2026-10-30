"""Prepare JETS artifacts from entity/event tables and fit both target encoders.

Input is either pre-aggregated account.csv/trans.csv tables or an explicitly
selected Retail II workbook benchmark conversion. The latter reproduces the
existing local data recipe, including negative invoice-total clipping and cohort
selection; these are source-data rules, not generative post-processing. No future
target is used at native generation. PyTorch artifacts must come from a trusted source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import torch

from experiments.data import json_digest, load_bank, read_json, sha256, write_json
from experiments.data_prep import _prepare_lock, prepare_raw
from experiments.retail import build_model, seed_all, source_hashes
from experiments.target_encoders import (_publish_with_receipt, _recover_publication,
                                         target_encoder_config, train_target_encoder, build_cache)


def _settings(config, target_steps=None):
    """Map user-facing target encoder settings to the compatible training API."""
    if 'model' not in config or 'code_dim' not in config['model']:
        raise ValueError('Generator model configuration and code_dim are required')
    target = config.get('target_encoder', {})
    if not isinstance(target, dict):
        raise ValueError('target_encoder configuration must be an object')
    allowed = {'steps', 'batch_size', 'learning_rate', 'cache_batch_size', 'log_interval',
               'width', 'layers', 'heads', 'decoder_layers'}
    if set(target) - allowed:
        raise ValueError('Unknown target_encoder settings: ' + ', '.join(sorted(set(target) - allowed)))
    overrides = {
        'seed': config['training_seed'],
        'teacher_steps': target.get('steps', 5000) if target_steps is None else target_steps,
        'teacher_batch_size': target.get('batch_size', 32),
        'teacher_learning_rate': target.get('learning_rate', .0003),
        'code_cache_batch_size': target.get('cache_batch_size', 32),
        'log_interval': target.get('log_interval', 100),
        'weight_decay': config.get('weight_decay', .01),
        'gradient_clip': config.get('gradient_clip', 1.),
        'teacher': {'code_dim': config['model']['code_dim'],
                    'width': target.get('width', 128), 'layers': target.get('layers', 2),
                    'heads': target.get('heads', 4), 'decoder_layers': target.get('decoder_layers', 4)},
    }
    return target_encoder_config(overrides)


def _initial_checkpoint(bank, bank_path, config, path):
    """Create or verify a seeded *new* generator initial state; never replace one."""
    seed = int(config['training_seed'])
    seed_all(seed)
    model = build_model(bank, config, 'cpu')
    identity = {'bank_sha256': sha256(bank_path), 'model_config': config['model'],
                'config_sha256': json_digest(config), 'training_seed': seed,
                'initialization': 'fresh JETSGenerator; not an earlier experiment'}
    receipt = path.with_suffix('.json')
    _recover_publication(path, receipt, identity, 'checkpoint_sha256')
    if path.exists():
        if not receipt.exists():
            raise ValueError('Initial generator checkpoint has no receipt; refusing overwrite')
        recorded = read_json(receipt)
        if recorded.get('identity') != identity or recorded.get('checkpoint_sha256') != sha256(path):
            raise ValueError('Existing initial generator checkpoint identity or digest mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('identity') != identity or saved.get('config') != config:
            raise ValueError('Initial generator configuration mismatch')
        current = model.state_dict()
        if saved['model'].keys() != current.keys() or not all(
                torch.equal(saved['model'][key], current[key]) for key in current):
            raise ValueError('Initial generator parameters do not match the seeded architecture')
        return path
    if receipt.exists():
        raise ValueError('Initial generator receipt exists without checkpoint; refusing overwrite')
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_with_receipt(path, receipt, identity, 'checkpoint_sha256',
                          {'model': model.state_dict(), 'config': config, 'identity': identity},
                          {'identity': identity})
    return path


def _snapshot_config(source, destination, expected_digest):
    """Publish exact verified bytes without replacing user files.

    A unique, fully flushed temporary and same-directory exclusive hard link
    make publication atomic. A crash may leave an untrusted temp file; retries
    never remove or reuse it. A published snapshot is always verified on reuse.
    """
    source, destination = Path(source), Path(destination)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError('Config changed before snapshot publication')
    if destination.is_symlink():
        raise ValueError('Config snapshot cannot be a symlink')
    if destination.exists():
        if not destination.is_file() or sha256(destination) != expected_digest:
            raise ValueError('Existing config snapshot does not match source; refusing overwrite')
        return destination
    temporary = destination.with_name('.' + destination.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256(destination) != expected_digest:
        raise ValueError('Published config snapshot failed verification')
    return destination


def _run_locked(raw_dir, output, config_path, *, device='cpu', windows=4096,
                target_steps=None, schema=None, train_ids=None, observation_end=None,
                retail_ii_xlsx=None):
    """Build artifacts while the caller owns the output namespace lock."""
    if (raw_dir is None) == (retail_ii_xlsx is None):
        raise ValueError('Choose exactly one of raw-dir or retail-ii-xlsx')
    output, config_path = Path(output).resolve(), Path(config_path).resolve()
    input_path = Path(raw_dir if raw_dir is not None else retail_ii_xlsx).resolve()
    if raw_dir is not None:
        if output == input_path or output.is_relative_to(input_path):
            raise ValueError('Output must be outside the raw input directory')
        if not (input_path / 'account.csv').is_file() or not (input_path / 'trans.csv').is_file():
            raise ValueError('raw-dir must contain pre-aggregated account.csv and trans.csv')
        input_sha = {name: sha256(input_path / name) for name in ('account.csv', 'trans.csv')}
        input_kind = 'pre-aggregated account.csv and trans.csv'
    else:
        if train_ids is not None:
            raise ValueError('train_ids cannot be used with retail-ii-xlsx: the benchmark recipe fits cohort categories before the split; provide pre-split account.csv/trans.csv via raw-dir instead')
        if not input_path.is_file():
            raise ValueError('retail-ii-xlsx must be an existing workbook')
        if schema is not None:
            raise ValueError('A custom schema is not supported by the fixed Retail II benchmark recipe')
        input_sha = {'workbook': sha256(input_path)}
        input_kind = ('Retail II workbook converted with the existing local benchmark recipe: '
                      'invoice aggregation, 21..800 invoice cohort selection and negative total clipping')
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes.decode('utf-8'))
    settings = _settings(config, target_steps)
    seed = int(config['training_seed'])
    identity = {'input_path': str(input_path), 'input_sha256': input_sha, 'config_path': str(config_path),
                'config_sha256': config_sha, 'windows': windows, 'seed': seed,
                'target_config': settings, 'device': str(device), 'schema': schema,
                'train_ids': train_ids, 'observation_end': observation_end,
                'source_sha256': source_hashes(),
                'input_kind': input_kind,
                'scope': 'full supplied population, not heldout' if train_ids is None else 'selected training entities only'}
    manifest = output / 'pipeline.json'
    if manifest.exists():
        if read_json(manifest) != identity:
            raise ValueError('Pipeline output belongs to different inputs, settings, device, or sources')
    else:
        if output.exists() and any(path.name != '.jets-preparation.lock' for path in output.iterdir()):
            raise FileExistsError('Output directory is not empty; use a new pipeline namespace')
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, identity)

    config_snapshot = _snapshot_config(config_path, output / 'config.snapshot.json', config_sha)
    if retail_ii_xlsx is not None:
        from experiments.retail_source import prepare_retail_ii_benchmark
        table_dir = Path(prepare_retail_ii_benchmark(input_path, output / 'source_tables'))
    else:
        table_dir = input_path
    prepare_raw(table_dir, output / 'input', schema=schema, seed=seed, windows=windows,
                train_ids=train_ids, observation_end=observation_end)
    bank_path, bank = load_bank(output / 'input' / 'bank')
    artifacts = {'config_snapshot': config_snapshot, 'bank': bank_path,
                 'preprocessor': output / 'input/prepared/preprocessor.json',
                 'parents': output / 'input/prepared/parents.json'}
    for family in ('time', 'feature'):
        encoder_dir = output / 'target_encoders' / family
        train_target_encoder(bank_path, encoder_dir, family, settings, device)
        checkpoint = encoder_dir / 'final.pt'
        cache = output / 'target_caches' / f'{family}.pt'
        build_cache(bank_path, checkpoint, cache, family, settings, device)
        artifacts[f'{family}_target_encoder'] = checkpoint
        artifacts[f'{family}_cache'] = cache
    artifacts['initial_generator'] = _initial_checkpoint(bank, bank_path, config, output / 'initial_generator.pt')
    if source_hashes() != identity['source_sha256']:
        raise ValueError('Pipeline source changed during execution')
    if sha256(config_path) != config_sha or sha256(config_snapshot) != config_sha:
        raise ValueError('Config or config snapshot changed during pipeline execution')
    current_sha = ({name: sha256(input_path / name) for name in ('account.csv', 'trans.csv')}
                   if raw_dir is not None else {'workbook': sha256(input_path)})
    if current_sha != input_sha:
        raise ValueError('Raw inputs changed during pipeline execution')
    train_command = [sys.executable, '-m', 'experiments.retail', 'train',
        '--bank', str(artifacts['bank']), '--config', str(config_snapshot),
        '--time-target-encoder', str(artifacts['time_target_encoder']), '--time-cache', str(artifacts['time_cache']),
        '--feature-target-encoder', str(artifacts['feature_target_encoder']), '--feature-cache', str(artifacts['feature_cache']),
        '--initial', str(artifacts['initial_generator']), '--output', str(output / 'generator'), '--device', str(device)]
    result = {'scope': identity['scope'], 'target_steps': settings['teacher_steps'],
              'input_kind': identity['input_kind'],
              'exploratory_budget': settings['teacher_steps'] != 5000 or windows != 4096,
              'artifacts': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in artifacts.items()},
              'generation_future_target_inputs': False, 'generator_train_command': train_command,
              'next_step': 'generator_train_command is an argument array; run it to begin generator training.'}
    complete = output / 'complete.json'
    if complete.exists() and read_json(complete) != result:
        raise ValueError('Pipeline completion receipt or artifacts changed')
    if not complete.exists():
        write_json(complete, result)
    return result


def run(raw_dir, output, config_path, *, device='cpu', windows=4096,
        target_steps=None, schema=None, train_ids=None, observation_end=None,
        retail_ii_xlsx=None):
    """Own the output process lock from source conversion through cache completion."""
    if (raw_dir is None) == (retail_ii_xlsx is None):
        raise ValueError('Choose exactly one of raw-dir or retail-ii-xlsx')
    output_path = Path(output)
    if output_path.is_symlink():
        raise ValueError('Pipeline output directory cannot be a symlink')
    output_path = output_path.resolve()
    input_path = Path(raw_dir if raw_dir is not None else retail_ii_xlsx).resolve()
    # Validate source boundaries before the lock helper creates any directory.
    if output_path == input_path or output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path):
        raise ValueError('Output must be outside the raw input directory and must not contain the source')
    if raw_dir is not None:
        if not (input_path / 'account.csv').is_file() or not (input_path / 'trans.csv').is_file():
            raise ValueError('raw-dir must contain pre-aggregated account.csv and trans.csv')
    else:
        if train_ids is not None:
            raise ValueError('train_ids cannot be used with retail-ii-xlsx: the benchmark recipe fits cohort categories before the split; provide pre-split account.csv/trans.csv via raw-dir instead')
        if schema is not None:
            raise ValueError('A custom schema is not supported by the fixed Retail II benchmark recipe')
        if not input_path.is_file():
            raise ValueError('retail-ii-xlsx must be an existing workbook')
    if not Path(config_path).is_file():
        raise ValueError('A readable config file is required')
    lock_path = output_path / '.jets-preparation.lock'
    if lock_path.is_symlink():
        raise ValueError('Pipeline output lock cannot be a symlink')
    # Do not create a lock in an unrelated, nonempty output directory.
    if (output_path.exists() and not (output_path / 'pipeline.json').exists()
            and any(path.name != lock_path.name for path in output_path.iterdir())):
        raise FileExistsError('Output directory is not empty; use a new pipeline namespace')
    with _prepare_lock(output_path):
        return _run_locked(raw_dir, output_path, config_path, device=device, windows=windows,
                           target_steps=target_steps, schema=schema, train_ids=train_ids,
                           observation_end=observation_end, retail_ii_xlsx=retail_ii_xlsx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--raw-dir', type=Path, help='Directory containing pre-aggregated account.csv and trans.csv')
    source.add_argument('--retail-ii-xlsx', type=Path, help='Explicit opt-in to local Retail II benchmark recipe: aggregates invoices, selects a 21-800 invoice cohort, and clips negative invoice totals to zero')
    parser.add_argument('--output', required=True, type=Path, help='New or matching pipeline output directory')
    parser.add_argument('--config', required=True, type=Path, help='Generator JSON config; target_encoder settings optional')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--windows', type=int, default=4096)
    parser.add_argument('--target-steps', type=int, help='Explicit exploratory override for each target encoder')
    parser.add_argument('--schema', type=Path, help='Optional entity/event schema JSON')
    parser.add_argument('--train-ids', type=Path, help='Optional JSON array of training entity IDs for --raw-dir only; not supported with the workbook benchmark recipe')
    parser.add_argument('--observation-end', type=float, help='Optional known observation horizon')
    args = parser.parse_args()
    if args.windows < 1 or (args.target_steps is not None and args.target_steps < 1):
        parser.error('windows and target-steps must be positive')
    if args.retail_ii_xlsx is not None and args.train_ids is not None:
        parser.error('--train-ids requires pre-split account.csv/trans.csv supplied via --raw-dir; workbook preprocessing fits cohort categories before splitting')
    schema = read_json(args.schema) if args.schema else None
    ids = read_json(args.train_ids) if args.train_ids else None
    if ids is not None and not isinstance(ids, list):
        parser.error('train-ids must be a JSON array')
    print(json.dumps(run(args.raw_dir, args.output, args.config, device=args.device,
                         windows=args.windows, target_steps=args.target_steps,
                         schema=schema, train_ids=ids, observation_end=args.observation_end,
                         retail_ii_xlsx=args.retail_ii_xlsx),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
