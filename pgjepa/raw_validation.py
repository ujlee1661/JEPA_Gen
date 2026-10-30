"""Original validation observations for generation-quality selection, never test data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import make_factorial, split_entities
from .experiment import digest, read_json, write_json
from .quantile import QuantilePreprocessor
from .raw_truth import _validate_numeric
from .window_data import truncate_synthetic_windows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_raw_validation_truth(data: str | Path, config: dict, dataset: str) -> list[np.ndarray]:
    """Return raw validation numerics in saved ID order, preserving unseen tails.

    Only the validation tensor and original raw sources are opened. The
    validation-only sidecars leave prepared manifests and training inputs intact.
    Source values are validated with the forward transform; inverse quantiles
    cannot reconstruct genuine observations beyond training support.
    """
    root = Path(data)
    manifest = read_json(root / 'manifest.json')
    if manifest['dataset'] != dataset or manifest['config'] != config:
        raise ValueError('Requested validation truth dataset/config differs from prepared data')
    val_hash = _sha(root / 'val.pt')
    if val_hash != manifest['data_hashes']['val.pt']:
        raise ValueError('Prepared validation tensor integrity check failed')
    state = read_json(root / 'preprocessor.json')
    if digest(state) != manifest['preprocessor_hash']:
        raise ValueError('Prepared quantile state integrity check failed')
    prep = QuantilePreprocessor.from_state_dict(state)
    ids = read_json(root / 'split_ids.json')['val']
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate saved validation entity IDs')
    encoded = torch.load(root / 'val.pt', map_location='cpu')
    package = Path(__file__).resolve().parent
    source_hashes = {f'pgjepa/{name}': _sha(package / name)
                     for name in ('data.py', 'window_data.py')}
    provenance = {
        'kind': 'raw_validation_numeric_v1', 'dataset': dataset,
        'config_digest': digest(config), 'data_manifest_digest': digest(manifest),
        'prepared_validation_sha256': val_hash,
        'preprocessor_hash': manifest['preprocessor_hash'],
        'validation_ids_digest': digest(ids), 'validation_ids': ids,
        'source_hashes': {**source_hashes, 'pgjepa/raw_validation.py': _sha(Path(__file__))},
        'policy': 'validation-only original observations; validate forward transform; never inverse reconstruction',
    }
    for name, current_hash in source_hashes.items():
        original = manifest.get('source_hashes', {}).get(name)
        if original is not None and original != current_hash:
            raise ValueError('Validation truth source differs from prepared data')
    raw_cache = Path(config.get('retail_raw_cache', 'data/PG_RetailII/entities.pt'))
    if dataset == 'retail':
        provenance['retail_raw_cache_sha256'] = _sha(raw_cache)
        original = manifest.get('exposure', {}).get('raw_cache_sha256')
        if original is not None and original != provenance['retail_raw_cache_sha256']:
            raise ValueError('Retail raw cache differs from prepared data')
    elif dataset != 'synthetic':
        raise ValueError(f'Unsupported validation truth dataset {dataset}')
    target, sidecar = root / 'val_raw_numeric.pt', root / 'val_raw_manifest.json'
    if target.exists() and sidecar.exists():
        stored = read_json(sidecar)
        if stored.get('provenance') != provenance:
            raise ValueError('Raw validation provenance changed; refusing stale cache')
        if stored.get('payload_sha256') != _sha(target):
            raise ValueError('Raw validation sidecar integrity check failed')
        raw = [value.numpy() for value in torch.load(target, map_location='cpu')]
        _validate_numeric(raw, encoded, prep)
        return raw
    if dataset == 'synthetic':
        cap = config.get('synthetic_event_cap', 512)
        entities = make_factorial(config.get('entities', 2000), config.get('data_seed', 20261031),
                                  min_events=cap, max_events=cap)
        entities = truncate_synthetic_windows(entities, config.get('prefix_test', 16),
                                              config.get('synthetic_window_days', 7.))
        validation = split_entities(entities, config.get('data_seed', 20261031))['val']
        if [entity['entity_id'] for entity in validation] != ids:
            raise ValueError('Recreated synthetic validation IDs differ from saved split')
    else:
        cached = torch.load(raw_cache, map_location='cpu')
        by_id = {str(entity['entity_id']): entity for entity in cached['entities']}
        if len(by_id) != len(cached['entities']):
            raise ValueError('Duplicate Retail raw-cache entity IDs')
        if any(str(identifier) not in by_id for identifier in ids):
            raise ValueError('Saved validation ID absent from Retail raw cache')
        validation = [by_id[str(identifier)] for identifier in ids]
    raw = [np.asarray(entity['x_num'], dtype=np.float64).copy() for entity in validation]
    _validate_numeric(raw, encoded, prep)
    temporary = target.with_suffix('.pt.tmp')
    torch.save([torch.from_numpy(values) for values in raw], temporary)
    temporary.replace(target)
    write_json(sidecar, {'provenance': provenance, 'payload_sha256': _sha(target),
                         'entities': len(raw), 'rows': sum(len(values) for values in raw)})
    print(f'Preserved original raw {dataset} validation truth: {len(raw)} entities', flush=True)
    return raw
