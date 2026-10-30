"""Prepared-window and suffix-target utilities for JETS experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def load_bank(path: Path):
    """Load a trusted serialized bank and validate its adjacent receipt if present."""
    path = path.resolve()
    if path.is_dir():
        path = path / 'data.pt'
    manifest = path.parent / 'manifest.json'
    if manifest.exists():
        expected = read_json(manifest).get('data_sha256')
        if expected is not None and sha256(path) != expected:
            raise ValueError('Prepared window bank does not match its manifest')
    data = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('windows', 'attrs', 'state'):
        if key not in data:
            raise ValueError(f'Missing prepared-bank field: {key}')
    if not data['windows']:
        raise ValueError('Prepared window bank is empty')
    return path, data


def load_cache(path: Path, bank_path: Path, windows, code_dim: int, teacher: Path):
    cache = torch.load(path, map_location='cpu', weights_only=False)
    if cache.get('data_sha256') != sha256(bank_path):
        raise ValueError(f'Teacher cache does not match bank: {path}')
    if cache.get('teacher_checkpoint_sha256') != sha256(teacher):
        raise ValueError(f'Teacher cache does not match teacher checkpoint: {path}')
    codes = cache['codes']
    if len(codes) != len(windows) or not all(
        tuple(code.shape) == (int(window['count']) + 1, code_dim)
        and bool(torch.isfinite(code).all()) for code, window in zip(codes, windows)
    ):
        raise ValueError(f'Teacher cache has incorrect suffix targets: {path}')
    return codes


def dense_targets(codes, indices, length, device, generator, noise_std):
    result = torch.zeros(len(indices), length, codes[0].shape[-1], device=device)
    for row, index in enumerate(indices):
        target = codes[int(index)]
        if len(target) > length:
            raise ValueError('Suffix target exceeds collated window')
        result[row, :len(target)] = target.to(device)
    noise = torch.randn(result.shape, device=device, dtype=result.dtype, generator=generator)
    return result + float(noise_std) * noise
