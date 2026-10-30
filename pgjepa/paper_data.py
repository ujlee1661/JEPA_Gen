"""Frozen full-parent TimeFIRST populations for our fidelity model variant."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .experiment import digest, read_json, write_json
from .quantile import QuantilePreprocessor


SCHEMAS: dict[str, dict[str, list[str]]] = {
    'berka': {'numeric': ['log_amount', 'log_balance'], 'categorical': ['cat_code'],
              'parent_numeric': [], 'parent_categorical': ['frequency', 'district_id_attr']},
    'coupon': {'numeric': ['log_page', 'log_cumpage'], 'categorical': ['cat_code'],
               'parent_numeric': ['AGE'], 'parent_categorical': ['SEX_ID', 'PREF_NAME']},
    'online_retail': {'numeric': ['log_amount', 'log_cumamt'], 'categorical': [],
                      'parent_numeric': [], 'parent_categorical': ['Country']},
    'sparkov': {'numeric': ['log_amount', 'log_cumamt'], 'categorical': ['cat_code'],
                'parent_numeric': ['age'], 'parent_categorical': ['gender', 'state']},
}
EXPECTED = {'berka': (500, 114255), 'coupon': (500, 87322),
            'online_retail': (417, 18284), 'sparkov': (500, 96245)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_paper_raw(raw_root, dataset, *, enforce_paper_counts=True):
    """Read source values without recomputing any supplied mark column."""
    schema = SCHEMAS[dataset]
    raw = Path(raw_root) / dataset / 'raw'
    parent = pd.read_csv(raw / 'account.csv')
    child = pd.read_csv(raw / 'trans.csv')
    parent_columns = ['account_id', *schema['parent_numeric'], *schema['parent_categorical']]
    child_columns = ['trans_id', 'account_id', 't_offset', *schema['numeric'], *schema['categorical']]
    if set(parent.columns) != set(parent_columns) or set(child.columns) != set(child_columns):
        raise ValueError('Unexpected paper schema; utility labels or missing fields are not accepted')
    if parent.account_id.duplicated().any() or child.trans_id.duplicated().any():
        raise ValueError('Duplicate paper keys')
    if not set(child.account_id).issubset(set(parent.account_id)):
        raise ValueError('Child foreign key is missing from fixed parent population')
    if enforce_paper_counts and (len(parent), len(child)) != EXPECTED[dataset]:
        raise ValueError('Paper population counts differ')
    if not np.isfinite(child.t_offset).all() or (child.t_offset < 0).any():
        raise ValueError('Calendar offsets must be finite and nonnegative')
    end = float(child.t_offset.max())
    groups = {key: group.sort_values(['t_offset', 'trans_id'], kind='stable')
              for key, group in child.groupby('account_id', sort=False)}
    entities = []
    for _, row in parent.iterrows():
        key = row['account_id']
        group = groups.get(key)
        if group is None:
            raise ValueError('Frozen paper parent has no source child events')
        times = group.t_offset.to_numpy(dtype=np.float64)
        attrs: dict[str, Any] = {name: float(row[name]) for name in schema['parent_numeric']}
        for name in schema['parent_categorical']:
            value = row[name]
            attrs[name] = None if pd.isna(value) else str(value)
        entities.append({'entity_id': int(key),
                         'x_num': group[schema['numeric']].to_numpy(dtype=np.float64),
                         'x_cat': group[schema['categorical']].to_numpy(dtype=object),
                         'delta': np.diff(np.r_[0., times]), 'calendar_times': times,
                         'exposure_end': end, 'attrs': attrs,
                         'numeric_names': schema['numeric'], 'categorical_names': schema['categorical']})
    return entities, {'dataset': dataset, 'entities': len(parent), 'events': len(child),
        'source_files': {name: sha256(raw / name) for name in ('account.csv', 'trans.csv')},
        'schema': schema, 'source_column_order': {'parent': list(parent.columns), 'child': list(child.columns)},
        'observation_end': end,
        'observation_end_assumption': 'own native process uses maximum frozen calendar offset, not target-specific last time',
        'scope': 'same-parent full-population fidelity; no held-out generalization claim'}


def prepare_paper_population(raw_root, dataset, output, *, enforce_paper_counts=True, seed=9):
    output = Path(output)
    entities, provenance = load_paper_raw(raw_root, dataset, enforce_paper_counts=enforce_paper_counts)
    if (output / 'manifest.json').exists():
        saved = read_json(output / 'manifest.json')
        if saved['source'] != provenance or sha256(output / 'population.pt') != saved['population_sha256']:
            raise ValueError('Prepared paper population identity changed')
        if digest(read_json(output / 'preprocessor.json')) != saved['preprocessor_hash']:
            raise ValueError('Prepared quantile state changed')
        return saved
    output.mkdir(parents=True, exist_ok=True)
    prep = QuantilePreprocessor(max_categories=4096, seed=seed, quantile_grid='all_observed').fit(entities)
    records = []
    for entity in entities:
        record = prep.transform(entity)
        record['calendar_times'] = torch.as_tensor(entity['calendar_times'], dtype=torch.float64)
        records.append(record)
    target = output / 'population.pt'
    temporary = target.with_suffix('.pt.tmp')
    torch.save(records, temporary)
    temporary.replace(target)
    write_json(output / 'preprocessor.json', prep.state_dict())
    write_json(output / 'parents.json', [entity['entity_id'] for entity in entities])
    manifest = {'source': provenance, 'population_sha256': sha256(target),
                'preprocessor_hash': digest(prep.state_dict()),
                'parent_hash': digest([entity['entity_id'] for entity in entities]),
                'numeric_policy': 'all observed training quantiles; existing log marks are independent targets',
                'static_types': {'numeric': SCHEMAS[dataset]['parent_numeric'],
                                 'categorical': SCHEMAS[dataset]['parent_categorical']}}
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_prepared_paper(output):
    root = Path(output)
    manifest = read_json(root / 'manifest.json')
    if sha256(root / 'population.pt') != manifest['population_sha256']:
        raise ValueError('Population payload changed')
    state = read_json(root / 'preprocessor.json')
    if digest(state) != manifest['preprocessor_hash']:
        raise ValueError('Quantile state changed')
    return (torch.load(root / 'population.pt', map_location='cpu'),
            QuantilePreprocessor.from_state_dict(state), manifest)
