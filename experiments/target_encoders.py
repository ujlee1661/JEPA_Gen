"""Train frozen future target encoders and cache suffix representations.

Input banks and checkpoints use PyTorch serialization and must be trusted. The
target encoder sees actual future windows only to build training targets; these codes
are never passed to native generation as conditions.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch

import jets
from experiments.data import load_bank, read_json, sha256, write_json
from jets.data import build_suffix_window, collate_windows
from jets.encoders import FeatureTargetEncoder, TimeTargetEncoder


DEFAULTS = {
    'seed': 9, 'teacher_steps': 5000, 'teacher_batch_size': 32,
    'teacher_learning_rate': 0.0003, 'weight_decay': 0.01,
    'gradient_clip': 1.0, 'log_interval': 100, 'code_cache_batch_size': 32,
    'teacher': {'code_dim': 64, 'width': 128, 'layers': 2, 'heads': 4,
                'decoder_layers': 4},
}


def target_encoder_config(overrides=None):
    """Return explicit target encoder settings; unrelated generator settings ignored."""
    result: dict[str, Any] = copy.deepcopy(DEFAULTS)
    if overrides:
        for key in result:
            if key in overrides:
                if key == 'teacher':
                    result[key].update(overrides[key])
                else:
                    result[key] = overrides[key]
        # A generator predicts one coordinate per target-code coordinate.
        model_dim = overrides.get('model', {}).get('code_dim')
        if model_dim is not None:
            teacher_dim = overrides.get('teacher', {}).get('code_dim')
            if teacher_dim is not None and teacher_dim != model_dim:
                raise ValueError('Target encoder and generator code dimensions must match')
            result['teacher']['code_dim'] = model_dim
    for key in ('teacher_steps', 'teacher_batch_size', 'log_interval', 'code_cache_batch_size'):
        if isinstance(result[key], bool) or int(result[key]) != result[key] or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('teacher_learning_rate', 'gradient_clip'):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(result['weight_decay']) or result['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    return result


def _seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def _pending_paths(path):
    path = Path(path)
    return (path.with_name('.' + path.name + '.pending'),
            path.with_name('.' + path.name + '.journal.json'))


def _publish_json_exclusive(path, value):
    """Atomically publish a JSON file without replacing any existing path."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recover_publication(target, receipt_path, identity, hash_field):
    """Complete only a journaled artifact with matching immutable digests.

    A crash before the journal exists leaves an unverified candidate and is
    deliberately fail-closed; no existing output is overwritten or deleted.
    """
    target, receipt_path = Path(target), Path(receipt_path)
    candidate, journal = _pending_paths(target)
    if not journal.exists():
        if candidate.exists():
            raise ValueError('Unjournaled pending artifact; refusing unverified recovery')
        return False
    info = read_json(journal)
    if (info.get('version') != 1 or info.get('identity') != identity
            or info.get('target_name') != target.name or info.get('receipt_name') != receipt_path.name
            or info.get('hash_field') != hash_field):
        raise ValueError('Pending publication identity mismatch')
    digest, receipt = info.get('artifact_sha256'), info.get('receipt')
    if not isinstance(digest, str) or not isinstance(receipt, dict) or receipt.get(hash_field) != digest:
        raise ValueError('Pending publication receipt is invalid')
    if candidate.exists() and sha256(candidate) != digest:
        raise ValueError('Pending artifact digest mismatch')
    if target.exists() and sha256(target) != digest:
        raise ValueError('Existing artifact differs from pending journal')
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError('Existing receipt differs from pending journal')
    if not target.exists():
        if not candidate.is_file():
            raise ValueError('Pending publication has no verified artifact')
        # Hard-link creation is atomic and fails if another writer owns target.
        os.link(candidate, target)
    if not receipt_path.exists():
        _publish_json_exclusive(receipt_path, receipt)
    # Verify both public artifacts before cleaning up only our journal/candidate.
    if sha256(target) != digest or read_json(receipt_path) != receipt:
        raise ValueError('Published target or receipt failed verification')
    candidate.unlink(missing_ok=True)
    journal.unlink()
    return True


def _publish_with_receipt(target, receipt_path, identity, hash_field, artifact, receipt):
    """Journal artifact digest before publishing either final path."""
    target, receipt_path = Path(target), Path(receipt_path)
    candidate, journal = _pending_paths(target)
    if target.exists() or receipt_path.exists() or candidate.exists() or journal.exists():
        raise ValueError('Output or pending artifact already exists; refusing overwrite')
    with candidate.open('xb') as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256(candidate)
    complete = {**receipt, hash_field: digest}
    _publish_json_exclusive(journal, {'version': 1, 'identity': identity,
        'target_name': target.name, 'receipt_name': receipt_path.name,
        'hash_field': hash_field, 'artifact_sha256': digest, 'receipt': complete})
    _recover_publication(target, receipt_path, identity, hash_field)


def _source_hashes():
    """Record the implementation used by each resumable target encoder."""
    package_dir = Path(jets.__file__).resolve().parent
    paths = [*sorted(package_dir.rglob('*.py')), Path(__file__).resolve(),
             Path(__file__).resolve().with_name('data.py')]
    root = package_dir.parent
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def make_target_encoder(data, family, config, device='cpu'):
    settings = dict(config['teacher'])
    if family == 'time':
        settings.pop('decoder_layers')
        cls = TimeTargetEncoder
    elif family == 'feature':
        cls = FeatureTargetEncoder
    else:
        raise ValueError('target encoder family must be time or feature')
    state = data['state']
    return cls(state['token_dim'], data['attrs'].shape[1],
               state['mark_token_state']['observation_end'], **settings).to(device)


def _batch_inputs(data, config, step, device):
    """Stateless sample schedule and independent Gaussian pair per update."""
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], step, 0]))
    indices = rng.integers(len(data['windows']), size=config['teacher_batch_size'])
    batch = collate_windows([data['windows'][int(i)] for i in indices], device)
    generator = torch.Generator(device=device).manual_seed(config['seed'] + 104729 * step)
    global_noise = torch.randn(2, len(indices), config['teacher']['code_dim'],
                               generator=generator, device=device)
    local_noise = torch.randn(2, *batch['valid'].shape, data['state']['token_dim'],
                              generator=generator, device=device)
    return batch, global_noise, local_noise


def train_target_encoder(bank_path, output_dir, family, config=None, device='cpu', stop_after=None):
    """Fit one target encoder, calibrate on training windows, then freeze it.

    Fixed-final-update selection; a partial call can be resumed using latest.pt.
    ``stop_after`` is intended for tests or explicitly bounded exploratory runs.
    """
    cfg = target_encoder_config(config)
    if stop_after is not None and (stop_after < 1 or int(stop_after) != stop_after):
        raise ValueError('stop_after must be a positive integer')
    bank_path, data = load_bank(Path(bank_path))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if family not in ('time', 'feature'):
        raise ValueError('target encoder family must be time or feature')
    identity = {'family': family, 'config': cfg, 'data_sha256': sha256(bank_path),
                'source_sha256': _source_hashes(), 'numpy_version': str(np.__version__),
                'device': str(device), 'python_version': platform.python_version(),
                'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                'scope': 'training windows only; true future is target supervision only'}
    manifest = out / 'manifest.json'
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError('Target encoder output belongs to a different bank or configuration')
    if not manifest.exists():
        write_json(manifest, identity)
    _seed(cfg['seed'])
    model = make_target_encoder(data, family, cfg, device)
    final = out / 'final.pt'
    complete = out / 'complete.json'
    _recover_publication(final, complete, identity, 'checkpoint_sha256')
    if final.exists():
        if not complete.is_file():
            raise ValueError('Completed target encoder has no receipt; refusing unverified reuse')
        receipt = read_json(complete)
        if (receipt.get('checkpoint_sha256') != sha256(final)
                or receipt.get('data_sha256') != identity['data_sha256']
                or receipt.get('family') != family or receipt.get('updates') != cfg['teacher_steps']):
            raise ValueError('Completed target encoder receipt or artifact digest mismatch')
        saved = torch.load(final, map_location=device, weights_only=False)
        if saved.get('identity') != identity or saved.get('step') != cfg['teacher_steps'] or not saved.get('calibrated_training_only'):
            raise ValueError('Completed target encoder identity or calibration mismatch')
        model.load_state_dict(saved['model'], strict=True)
        return model.freeze()
    if complete.exists():
        raise ValueError('A target encoder receipt exists without its checkpoint; refusing overwrite')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['teacher_learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    first, previous_elapsed, history = 1, 0., []
    latest = out / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get('identity') != identity:
            raise ValueError('Target encoder resume identity mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        first, previous_elapsed, history = saved['step'] + 1, saved['elapsed_seconds'], saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    end = min(cfg['teacher_steps'], int(stop_after)) if stop_after is not None else cfg['teacher_steps']
    started = time.perf_counter()
    for step in range(first, end + 1):
        model.train()
        batch, global_noise, local_noise = _batch_inputs(data, cfg, step, device)
        _seed(cfg['seed'] + 104729 * step + 1)
        if family == 'feature':
            loss, stats = model.loss(batch, objective='energy', global_noise_pair=global_noise,
                                     local_noise_pair=local_noise)
        else:
            loss, stats = model.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite target encoder loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite target encoder gradient')
        optimizer.step()
        if step == 1 or step % cfg['log_interval'] == 0 or step == end:
            elapsed = previous_elapsed + time.perf_counter() - started
            history.append({'step': step, 'loss': float(loss.detach()), 'diagnostics': stats,
                            'elapsed_seconds': elapsed})
            _atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'identity': identity, 'step': step, 'history': history,
                          'elapsed_seconds': elapsed, 'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, latest)
            print(f'{family} target encoder {step}/{cfg["teacher_steps"]} loss={float(loss.detach()):.6f}', flush=True)
    if end < cfg['teacher_steps']:
        return None
    if sha256(bank_path) != identity['data_sha256'] or _source_hashes() != identity['source_sha256']:
        raise ValueError('Target encoder bank or source changed during training')
    model.freeze()
    with torch.no_grad():
        codes = torch.cat([model.encode(collate_windows(data['windows'][i:i+cfg['teacher_batch_size']], device)).cpu()
                           for i in range(0, len(data['windows']), cfg['teacher_batch_size'])])
        if not bool(torch.isfinite(codes).all()):
            raise RuntimeError('Nonfinite training codes for calibration')
        model.set_code_calibration(codes.to(device))
    if sha256(bank_path) != identity['data_sha256'] or _source_hashes() != identity['source_sha256']:
        raise ValueError('Target encoder bank or source changed during calibration')
    _publish_with_receipt(final, complete, identity, 'checkpoint_sha256',
        {'model': model.state_dict(), 'identity': identity, 'family': family,
         'step': cfg['teacher_steps'], 'calibrated_training_only': True},
        {'family': family, 'updates': cfg['teacher_steps'],
         'data_sha256': identity['data_sha256'], 'target_encoder_frozen': True,
         'calibration_scope': 'training windows only'})
    return model.freeze()


def build_cache(bank_path, target_checkpoint, output_path, family, config=None, device='cpu'):
    """Cache independently normalized suffix target codes for every window."""
    cfg = target_encoder_config(config)
    bank_path, data = load_bank(Path(bank_path))
    target_checkpoint, output_path = Path(target_checkpoint).resolve(), Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank_sha, teacher_sha = sha256(bank_path), sha256(target_checkpoint)
    receipt = output_path.with_suffix('.json')
    saved = torch.load(target_checkpoint, map_location=device, weights_only=False)
    if (saved.get('family') != family or saved.get('identity', {}).get('config') != cfg
            or saved.get('identity', {}).get('data_sha256') != bank_sha
            or saved.get('step') != cfg['teacher_steps'] or not saved.get('calibrated_training_only')):
        raise ValueError('Target encoder checkpoint does not match bank, family or configuration')
    if saved['identity'].get('source_sha256') != _source_hashes():
        raise ValueError('Target encoder source differs from training; refusing unverified cache')
    cache_identity = {'family': family, 'config': cfg, 'data_sha256': bank_sha,
                      'target_checkpoint_sha256': teacher_sha,
                      'source_sha256': saved['identity']['source_sha256']}
    _recover_publication(output_path, receipt, cache_identity, 'cache_sha256')
    if output_path.exists():
        if not receipt.is_file():
            raise ValueError('Existing suffix cache has no receipt; refusing unverified reuse')
        cached_receipt = read_json(receipt)
        if (cached_receipt.get('cache_sha256') != sha256(output_path)
                or cached_receipt.get('data_sha256') != bank_sha
                or cached_receipt.get('teacher_checkpoint_sha256') != teacher_sha):
            raise ValueError('Existing suffix cache receipt or artifact digest mismatch')
        cache = torch.load(output_path, map_location='cpu', weights_only=False)
        if cache.get('data_sha256') != bank_sha or cache.get('teacher_checkpoint_sha256') != teacher_sha:
            raise ValueError('Existing suffix cache belongs to a different target encoder or bank')
        if len(cache['codes']) != len(data['windows']) or not all(
            tuple(code.shape) == (int(window['count']) + 1, cfg['teacher']['code_dim'])
            and bool(torch.isfinite(code).all()) for code, window in zip(cache['codes'], data['windows'])):
            raise ValueError('Existing suffix cache has invalid targets')
        return output_path
    if receipt.exists():
        raise ValueError('A cache receipt exists without its target cache; refusing overwrite')
    model = make_target_encoder(data, family, cfg, device)
    model.load_state_dict(saved['model'], strict=True)
    model.freeze()
    codes = [torch.zeros(int(w['count']) + 1, cfg['teacher']['code_dim']) for w in data['windows']]
    jobs = sorted([(int(w['count']) - k, i, k) for i, w in enumerate(data['windows'])
                   for k in range(int(w['count']) + 1)], reverse=True)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(jobs), cfg['code_cache_batch_size']):
            chosen = jobs[start:start+cfg['code_cache_batch_size']]
            suffixes = [build_suffix_window(data['windows'][i], k) for _, i, k in chosen]
            values = model.encode(collate_windows(suffixes, device)).cpu()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite suffix target codes')
            for row, (_, index, k) in enumerate(chosen):
                codes[index][k] = values[row]
            if start % (100 * cfg['code_cache_batch_size']) == 0:
                print(f'{family} suffix cache {start}/{len(jobs)}', flush=True)
    if (sha256(bank_path) != bank_sha or sha256(target_checkpoint) != teacher_sha
            or _source_hashes() != saved['identity']['source_sha256']):
        raise ValueError('Target encoder, source or bank changed during suffix caching')
    _publish_with_receipt(output_path, receipt, cache_identity, 'cache_sha256',
        {'codes': codes, 'teacher_checkpoint_sha256': teacher_sha, 'data_sha256': bank_sha,
         'targets': 'independently normalized remaining windows, positions 0..N',
         'rows': len(jobs), 'code_dim': cfg['teacher']['code_dim'], 'family': family},
        {'rows': len(jobs), 'windows': len(codes), 'data_sha256': bank_sha,
         'teacher_checkpoint_sha256': teacher_sha, 'elapsed_seconds': time.perf_counter() - started,
         'true_future_input_to_generator': False})
    return output_path
