"""Administrative exposure data for fixed-time, variable-count experiments."""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import _category, _date, _missing, _retail_rows, make_factorial, split_entities
from .experiment import digest, read_json, source_hashes, write_json
from .quantile import QuantilePreprocessor


def truncate_synthetic_windows(entities: list[dict], prefix_events: int,
                               duration_days: float) -> list[dict]:
    """Stop at a clock boundary; fail the whole cohort if a guard is exhausted."""
    if prefix_events < 1 or duration_days <= 0 or not np.isfinite(duration_days):
        raise ValueError('Require a positive prefix length and clock horizon')
    result = []
    for entity in entities:
        times = np.cumsum(entity['delta'])
        if len(times) < prefix_events:
            raise ValueError('Synthetic computational cap cannot supply the prefix')
        end = float(times[prefix_events - 1] + duration_days)
        if times[-1] < end:
            raise ValueError('Synthetic computational event cap exhausted; abort whole cohort')
        length = int(np.searchsorted(times, end, side='left'))
        record = dict(entity)
        for name in ('x_num', 'x_cat', 'delta'):
            record[name] = np.asarray(entity[name])[:length].copy()
        record['exposure_end'] = end
        record['metadata'] = {**entity.get('metadata', {}),
                              'observation_policy': 'prefix_endpoint_plus_fixed_clock_horizon',
                              'exposure_end_days': end, 'window_days': duration_days}
        result.append(record)
    return result


def retail_timestamp_map(workbook: str | Path, destination: str | Path) -> dict:
    """Recover timestamps omitted by the legacy cache; cache by workbook hash."""
    path, cache = Path(workbook), Path(destination)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache.exists():
        state = read_json(cache)
        if state.get('workbook_sha256') == file_hash:
            return state
    first: dict[str, datetime] = {}
    admin_end: datetime | None = None
    rows = 0
    for row in _retail_rows(path):
        rows += 1
        try:
            timestamp = _date(row.get('InvoiceDate'))
        except (ValueError, TypeError):
            continue
        admin_end = timestamp if admin_end is None else max(admin_end, timestamp)
        customer = row.get('CustomerID', row.get('Customer ID'))
        if _missing(customer) or str(customer).strip() == '':
            continue
        try:
            invoice = str(row.get('InvoiceNo', row.get('Invoice'))).strip()
            quantity = float(row['Quantity'])
            price = float(row.get('UnitPrice', row.get('Price')))
            if not invoice or not np.isfinite([quantity, price]).all() or price < 0:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        key = _category(customer)
        first[key] = min(first.get(key, timestamp), timestamp)
    if admin_end is None or not first:
        raise ValueError('No valid Retail administrative exposure could be recovered')
    state = {'source': str(path.resolve()), 'workbook_sha256': file_hash, 'rows': rows,
             'administrative_end': admin_end.isoformat(),
             'first_timestamp_by_customer': {key: value.isoformat() for key, value in first.items()},
             'policy': 'global workbook timestamp end, never customer last invoice'}
    write_json(cache, state)
    return state


def attach_retail_exposure(entities: list[dict], timestamp_state: dict) -> list[dict]:
    end = _date(timestamp_state['administrative_end'])
    first = timestamp_state['first_timestamp_by_customer']
    result = []
    for entity in entities:
        key = str(entity['entity_id'])
        if key not in first:
            raise ValueError(f'Missing first timestamp for Retail entity {key}')
        start = _date(first[key])
        exposure = (end - start).total_seconds() / 86400
        if exposure + 1e-8 < np.sum(entity['delta']):
            raise ValueError('Recovered Retail exposure ends before a recorded event')
        record = dict(entity)
        record['exposure_end'] = float(exposure)
        record['metadata'] = {**entity.get('metadata', {}), 'first_timestamp': start.isoformat(),
                              'administrative_end': end.isoformat(),
                              'exposure_end_days': float(exposure)}
        result.append(record)
    return result


def prepare_window_data(output: str | Path, config: dict, dataset: str = 'synthetic',
                        raw_cache: str | Path | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        existing = read_json(output / 'manifest.json')
        if existing['config'] != config or existing['dataset'] != dataset:
            raise ValueError('Prepared output belongs to a different experiment')
        for name, expected in existing['data_hashes'].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared data integrity check failed')
        if digest(read_json(output / 'preprocessor.json')) != existing['preprocessor_hash']:
            raise ValueError('Prepared quantile state integrity check failed')
        return existing
    seed = config.get('data_seed', 20261031)
    if dataset == 'synthetic':
        cap = config.get('synthetic_event_cap', 512)
        entities = make_factorial(config.get('entities', 2000), seed,
                                  min_events=cap, max_events=cap)
        entities = truncate_synthetic_windows(entities, config.get('prefix_test', 16),
                                              config.get('synthetic_window_days', 7.))
        provenance: dict[str, Any] = {'dgp_event_cap': cap, 'guard_policy': 'whole_cohort_abort',
                                      'observation_policy': 'fixed_clock_after_prefix'}
    elif dataset == 'retail':
        cache_path = Path(raw_cache or 'data/PG_RetailII/entities.pt')
        cached = torch.load(cache_path, map_location='cpu')
        workbook = config.get('retail_path', 'data/raw/online_retail_II.xlsx')
        timestamp_state = retail_timestamp_map(workbook, output / 'retail_exposure.json')
        entities = attach_retail_exposure(cached['entities'], timestamp_state)
        provenance = {'workbook_sha256': timestamp_state['workbook_sha256'],
                      'raw_cache_sha256': hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                      'administrative_end': timestamp_state['administrative_end']}
    else:
        raise ValueError(f'Unsupported fixed-time dataset {dataset}')
    splits = split_entities(entities, seed)
    prep = QuantilePreprocessor(seed=seed,
                                quantile_grid=config.get('quantile_grid', 'legacy_1000')).fit(splits['train'])
    write_json(output / 'preprocessor.json', prep.state_dict())
    split_ids = {}
    audit = {}
    for name, records in splits.items():
        encoded = [prep.transform(entity) for entity in records]
        target = output / f'{name}.pt'
        temporary = target.with_suffix('.pt.tmp')
        torch.save(encoded, temporary)
        temporary.replace(target)
        split_ids[name] = [entity['entity_id'] for entity in records]
        audit[name] = {'entities': len(records), 'events': sum(len(e['delta']) for e in records),
                       'zero_tail_entities': sum(float(e['exposure_end']) > float(np.sum(e['delta']))
                                                 for e in records)}
        write_json(output / f'{name}_metadata.json',
                   [{'entity_id': e['entity_id'], 'metadata': e['metadata']} for e in records])
    write_json(output / 'split_ids.json', split_ids)
    write_json(output / 'audit.json', {'dataset': dataset, 'splits': audit, **provenance})
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ('train.pt', 'val.pt', 'test.pt')}
    manifest = {'dataset': dataset, 'config': config, 'data_hashes': hashes,
                'preprocessor_hash': digest(prep.state_dict()), 'source_hashes': source_hashes(),
                'numeric_transform': 'train_only_gaussian_quantile', 'exposure': provenance}
    write_json(output / 'manifest.json', manifest)
    print(f'Prepared fixed-time {dataset}: {audit}', flush=True)
    return manifest
