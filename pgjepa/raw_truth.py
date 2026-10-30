"""Preserve genuine raw test values for evaluating quantile-based generators.

Inverse-transforming encoded test data is not ground truth: train-fitted
quantiles saturate unseen test tails. These evaluation-only sidecars reconstruct
or retrieve raw observations without changing the prepared/training manifest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import make_factorial, split_entities
from .experiment import digest, read_json, write_json
from .quantile import QuantilePreprocessor
from .window_data import truncate_synthetic_windows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_numeric(raw: list[np.ndarray], encoded: list[dict],
                      prep: QuantilePreprocessor) -> None:
    if len(raw) != len(encoded):
        raise ValueError('Raw test entity count differs from encoded test')
    for values, record in zip(raw, encoded):
        expected = record['x_num'].cpu().numpy()
        if values.shape != expected.shape:
            raise ValueError('Raw test row/feature shape differs from encoded test')
        observed = np.isfinite(values)
        mask = record['mask'][..., :prep.n_num].cpu().numpy() >= .5
        if not np.array_equal(observed, mask):
            raise ValueError('Raw test missingness differs from encoded test observation mask')
        for index, model in enumerate(prep.quantile_models):
            valid = observed[:, index]
            if valid.any():
                transformed = model.transform(values[valid, index:index + 1])[:, 0]
                if not np.allclose(transformed, expected[valid, index], rtol=1e-6, atol=1e-6):
                    raise ValueError('Raw test forward quantiles differ from encoded test')


def ensure_raw_test_truth(data: str | Path, config: dict, dataset: str) -> list[np.ndarray]:
    """Return raw numeric arrays in saved test-ID order, with validated provenance.

    This helper is for the evaluation phase. Training never reads its sidecars.
    It writes ``test_raw_numeric.pt`` and ``test_raw_manifest.json`` atomically,
    leaving every original prepared-data artifact unchanged.
    """
    root = Path(data)
    manifest = read_json(root / 'manifest.json')
    if manifest['dataset'] != dataset or manifest['config'] != config:
        raise ValueError('Requested raw truth dataset/config differs from prepared data')
    test_hash = _sha(root / 'test.pt')
    if test_hash != manifest['data_hashes']['test.pt']:
        raise ValueError('Prepared test tensor integrity check failed')
    state = read_json(root / 'preprocessor.json')
    if digest(state) != manifest['preprocessor_hash']:
        raise ValueError('Prepared quantile state integrity check failed')
    prep = QuantilePreprocessor.from_state_dict(state)
    ids = read_json(root / 'split_ids.json')['test']
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate saved test entity IDs')
    encoded = torch.load(root / 'test.pt', map_location='cpu')
    package = Path(__file__).resolve().parent
    provenance = {
        'kind': 'raw_test_numeric_v1', 'dataset': dataset,
        'config_digest': digest(config), 'prepared_test_sha256': test_hash,
        'preprocessor_hash': manifest['preprocessor_hash'], 'test_ids_digest': digest(ids),
        'test_ids': ids, 'source_hashes': {'pgjepa/data.py': _sha(package / 'data.py'),
                                         'pgjepa/window_data.py': _sha(package / 'window_data.py')},
        'policy': 'raw observations; validate by forward transform, never inverse reconstruction',
    }
    prepared_sources = manifest.get('source_hashes', {})
    for name, current_hash in provenance['source_hashes'].items():
        if name in prepared_sources and prepared_sources[name] != current_hash:
            raise ValueError('Raw truth source differs from the prepared-data source')
    raw_cache = Path(config.get('retail_raw_cache', 'data/PG_RetailII/entities.pt'))
    if dataset == 'retail':
        provenance['retail_raw_cache_sha256'] = _sha(raw_cache)
        original_cache = manifest.get('exposure', {}).get('raw_cache_sha256')
        if original_cache is not None and original_cache != provenance['retail_raw_cache_sha256']:
            raise ValueError('Retail raw cache differs from the prepared-data source')
    elif dataset != 'synthetic':
        raise ValueError(f'Unsupported raw truth dataset {dataset}')
    target, sidecar = root / 'test_raw_numeric.pt', root / 'test_raw_manifest.json'
    if target.exists() and sidecar.exists():
        stored = read_json(sidecar)
        if stored.get('provenance') != provenance:
            raise ValueError('Raw test provenance changed; refusing stale ground truth cache')
        if stored.get('payload_sha256') != _sha(target):
            raise ValueError('Raw test numeric sidecar integrity check failed')
        tensors = torch.load(target, map_location='cpu')
        raw = [value.numpy() for value in tensors]
        _validate_numeric(raw, encoded, prep)
        return raw
    if dataset == 'synthetic':
        cap = config.get('synthetic_event_cap', 512)
        entities = make_factorial(config.get('entities', 2000), config.get('data_seed', 20261031),
                                  min_events=cap, max_events=cap)
        entities = truncate_synthetic_windows(entities, config.get('prefix_test', 16),
                                              config.get('synthetic_window_days', 7.))
        test = split_entities(entities, config.get('data_seed', 20261031))['test']
        if [entity['entity_id'] for entity in test] != ids:
            raise ValueError('Recreated synthetic test IDs differ from prepared split')
    else:
        cached = torch.load(raw_cache, map_location='cpu')
        by_id = {str(entity['entity_id']): entity for entity in cached['entities']}
        if len(by_id) != len(cached['entities']):
            raise ValueError('Duplicate Retail raw-cache entity IDs')
        if any(str(identifier) not in by_id for identifier in ids):
            raise ValueError('Saved Retail test ID absent from raw cache')
        test = [by_id[str(identifier)] for identifier in ids]
    raw = [np.asarray(entity['x_num'], dtype=np.float64).copy() for entity in test]
    _validate_numeric(raw, encoded, prep)
    temporary = target.with_suffix('.pt.tmp')
    torch.save([torch.from_numpy(values) for values in raw], temporary)
    temporary.replace(target)
    write_json(sidecar, {'provenance': provenance, 'payload_sha256': _sha(target),
                         'entities': len(raw), 'rows': sum(len(values) for values in raw)})
    print(f'Preserved genuine raw {dataset} test truth: {len(raw)} entities', flush=True)
    return raw
