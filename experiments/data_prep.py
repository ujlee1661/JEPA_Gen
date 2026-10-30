"""Prepare a JETS window bank from entity/event CSV tables.

The default columns are the already aggregated Online Retail experiment
tables, not the original transaction-line spreadsheet. Input feature values
are kept as supplied; this module does not infer or repair data semantics.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import importlib
import math
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.special import ndtri
import torch

from experiments.data import json_digest, read_json, sha256, write_json
from jets.data import QuantilePreprocessor, _complete_window
from jets.layers.spacing import encode_spacing


RETAIL_SCHEMA = {
    'numeric': ['log_amount', 'log_cumamt'], 'categorical': [],
    'parent_numeric': [], 'parent_categorical': ['Country'],
}


def _read_entities(raw_dir: Path, schema: dict, train_ids=None, observation_end=None):
    required = {'numeric', 'categorical', 'parent_numeric', 'parent_categorical'}
    if not required.issubset(schema) or set(schema) - (required | {'id_dtype'}) or any(
            not isinstance(schema[k], list) for k in required):
        raise ValueError('Schema must define numeric/categorical parent and event column lists')
    id_dtype = schema.get('id_dtype', 'infer')
    if id_dtype not in ('infer', 'string'):
        raise ValueError('Schema id_dtype must be infer or string')
    # Explicit string identifiers are parsed by converters to preserve leading
    # zeros and legitimate values such as "NA". Other columns retain pandas'
    # normal missing-value handling. The default is unchanged numeric inference.
    parent = pd.read_csv(raw_dir / 'account.csv', **(
        {'converters': {'account_id': str}} if id_dtype == 'string' else {}))
    child = pd.read_csv(raw_dir / 'trans.csv', **(
        {'converters': {'account_id': str, 'trans_id': str}} if id_dtype == 'string' else {}))
    parent_columns = ['account_id', *schema['parent_numeric'], *schema['parent_categorical']]
    child_columns = ['trans_id', 'account_id', 't_offset', *schema['numeric'], *schema['categorical']]
    if len(set(parent_columns)) != len(parent_columns) or len(set(child_columns)) != len(child_columns):
        raise ValueError('Schema columns must be unique')
    if set(parent.columns) != set(parent_columns) or set(child.columns) != set(child_columns):
        raise ValueError('Input tables do not match the selected entity/event schema')
    if parent.account_id.isna().any() or child.account_id.isna().any() or child.trans_id.isna().any():
        raise ValueError('Entity and event identifiers cannot be missing')
    if id_dtype == 'string' and any(
            (series.str.strip() == '').any() for series in (parent.account_id, child.account_id, child.trans_id)):
        raise ValueError('Entity and event identifiers cannot be blank')
    if parent.account_id.duplicated().any() or child.trans_id.duplicated().any():
        raise ValueError('Duplicate entity or event identifiers')
    if not set(child.account_id).issubset(set(parent.account_id)):
        raise ValueError('An event references an unknown entity')
    if train_ids is not None:
        requested = list(train_ids)
        if not requested or len(requested) != len(set(requested)) or not set(requested).issubset(set(parent.account_id)):
            raise ValueError('Training IDs must be unique, nonempty, and present in account.csv')
        parent = parent[parent.account_id.isin(requested)].copy()
        child = child[child.account_id.isin(requested)].copy()
    times = pd.to_numeric(child.t_offset, errors='raise').to_numpy(dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('Training events need finite nonnegative time offsets')
    end = float(np.max(times)) if observation_end is None else float(observation_end)
    if not math.isfinite(end) or end <= 0 or end < float(np.max(times)):
        raise ValueError('Observation end must be positive and cover all training events')
    child = child.copy()
    child['t_offset'] = times
    groups = {key: group.sort_values(['t_offset', 'trans_id'], kind='stable')
              for key, group in child.groupby('account_id', sort=False)}
    entities = []
    for index, row in parent.iterrows():
        # iterrows coerces an all-numeric row to a common dtype; reading an
        # integer ID from that row can round distinct IDs above 2**53.
        key = parent.at[index, 'account_id']
        group = groups.get(key, child.iloc[:0])
        ts = group.t_offset.to_numpy(dtype=np.float64)
        attrs = {name: float(row[name]) for name in schema['parent_numeric']}
        attrs.update({name: None if pd.isna(row[name]) else str(row[name])
                      for name in schema['parent_categorical']})
        key = key.item() if isinstance(key, np.generic) else key
        if isinstance(key, float) and math.isfinite(key) and key.is_integer():
            key = int(key)
        entities.append({'entity_id': key, 'x_num': group[schema['numeric']].to_numpy(dtype=np.float64),
            'x_cat': group[schema['categorical']].to_numpy(dtype=object),
            'delta': np.diff(np.r_[0., ts]), 'calendar_times': ts, 'exposure_end': end,
            'attrs': attrs, 'numeric_names': schema['numeric'], 'categorical_names': schema['categorical']})
    return entities, end


def _tokens(record, state, anchor, horizon, indices):
    indices = np.asarray(indices, dtype=np.int64)
    times = record['calendar_times'].numpy()[indices]
    if len(times) and ((times < anchor - 1e-8).any() or (times > anchor + horizon + 1e-8).any()):
        raise ValueError('Event falls outside query window')
    fraction = np.full(len(indices), .5) if horizon == 0 else (times - anchor) / horizon
    values = np.zeros((len(indices), state['token_dim']), dtype=np.float64)
    values[:, 0] = ndtri(np.clip(fraction, state['time_epsilon'], 1 - state['time_epsilon']))
    lo, hi = state['numeric_slice']
    values[:, lo:hi] = record['x_num'].numpy()[indices]
    for column, (start, stop) in enumerate(state['category_slices']):
        encoded = record['x_cat'].numpy()[indices, column].astype(np.int64)
        if ((encoded < 0) | (encoded >= stop - start)).any():
            raise ValueError('Invalid encoded category')
        values[np.arange(len(indices)), start + encoded] = 1.
    lo, hi = state['mask_slice']
    values[:, lo:hi] = record['mask'].numpy()[indices]
    if not np.isfinite(values).all():
        raise ValueError('Token coordinates must be finite')
    return values


def _token_state(records, prep, end):
    offset = 1 + prep.n_num
    slices = []
    for size in prep.cat_sizes:
        slices.append((offset, offset + size))
        offset += size
    state = {'version': 'jets_tokens_v1', 'n_num': prep.n_num, 'cat_sizes': list(prep.cat_sizes),
        'attrs_dim': prep.attrs_dim, 'numeric_slice': (1, 1 + prep.n_num),
        'category_slices': slices, 'mask_slice': (offset, offset + prep.n_num + len(prep.cat_sizes)),
        'token_dim': offset + prep.n_num + len(prep.cat_sizes),
        'continuous_indices': list(range(1 + prep.n_num)), 'observation_end': end, 'time_epsilon': 1e-7,
        'time_coordinate': 'probit(relative calendar fraction); epsilon only at numerical endpoints',
        'generation_order': 'jointly sort time and every corresponding mark/mask row'}
    all_values = np.concatenate([_tokens(record, state, 0., end, np.arange(len(record['delta'])))
                                 for record in records])
    mean, std = all_values.mean(0), all_values.std(0)
    state['constant_token_channels'] = np.flatnonzero(std < 1e-7).tolist()
    state['mean'], state['std'] = mean, np.where(std < 1e-7, 1., std)
    return state


def _encoded(record, state, anchor, horizon, indices):
    values = _tokens(record, state, anchor, horizon, indices)
    return torch.as_tensor((values - state['mean']) / state['std'], dtype=torch.float32)


def _raw_window(record, cut, horizon, token_state):
    times = record['calendar_times'].numpy()
    anchor = float(times[cut - 1]) if cut else 0.
    finish = max(cut, int(np.searchsorted(times, anchor + horizon, side='right')))
    selected = torch.as_tensor(times[cut:finish], dtype=torch.float64)
    timing = encode_spacing(selected, anchor, horizon)
    return {'count': finish - cut, 'anchor': anchor, 'delta': horizon, 'times': selected,
        'marks': _encoded(record, token_state, anchor, horizon, np.arange(cut, finish))[:, 1:],
        'past_tokens': _encoded(record, token_state, 0., token_state['observation_end'], np.arange(cut)),
        'attrs': record['attrs'].clone(), **timing}


def _window_bank(records, prep, end, count, seed):
    if count < 2 * len(records):
        raise ValueError('Window count must cover each entity at least once in the BOS half')
    token_state = _token_state(records, prep, end)
    rng = np.random.default_rng(seed)
    bos_count = count // 2
    endpoint_count = (count - bos_count) // 2
    bos_entities = np.concatenate([rng.permutation(len(records))
        for _ in range((bos_count + len(records) - 1) // len(records))])[:bos_count]
    raw, metadata = [], []
    for index in range(count):
        kind = 'bos_full' if index < bos_count else ('prefix_endpoint' if index < bos_count + endpoint_count else 'prefix_random')
        entity = int(bos_entities[index]) if kind == 'bos_full' else int(rng.integers(len(records)))
        record = records[entity]
        cut = 0 if kind == 'bos_full' else int(rng.integers(len(record['delta']) + 1))
        anchor = float(record['calendar_times'][cut - 1]) if cut else 0.
        remaining = end - anchor
        horizon = float(rng.random() * remaining) if kind == 'prefix_random' else remaining
        raw.append(_raw_window(record, cut, horizon, token_state))
        metadata.append({'entity': entity, 'cut': cut, 'kind': kind,
            'query_kind': {'bos_full': 'BOS_full', 'prefix_endpoint': 'prefix_endpoint', 'prefix_random': 'prefix_uniform'}[kind]})
    coordinates = sum(int(w['active'].sum()) for w in raw)
    if not coordinates:
        raise ValueError('Window bank has no active timing coordinates')
    rms = float(np.sqrt(sum(float(w['clr'][w['active']].square().sum()) for w in raw) / coordinates))
    scale = rms if rms > np.finfo(np.float64).eps else 1.
    windows = []
    for item, meta in zip(raw, metadata):
        window = _complete_window(item, scale, meta)
        window['past_extension_tokens'] = _encoded(records[meta['entity']], token_state, 0., end,
            np.arange(meta['cut'], meta['cut'] + item['count']))
        windows.append(window)
    state = {'version': 'jets_window_bank_v1', 'time_scale': scale, 'mark_token_state': token_state,
        'token_dim': token_state['token_dim'], 'mark_dim': token_state['token_dim'] - 1,
        'time_scale_fit': 'RMS of active CLR on training windows', 'active_CLR_coordinates': coordinates,
        'unscaled_active_CLR_RMS': rms, 'zero_timing_variance_fallback': 'scale 1',
        'timing_coordinate': 'native CLR/scalar_scale', 'mark_order': 'native stable order'}
    return {'version': 'jets_window_bank_v1', 'records': records,
        'attrs': torch.stack([r['attrs'] for r in records]), 'windows': windows, 'state': state,
        'manifest': {'seed': seed, 'window_bank_size': count, 'entities': len(records),
            'query_measure': 'half shuffled-entity BOS; remaining endpoint/random half split',
            'scope': 'training entities only; no held-out generalization claim'}}


def _save_torch(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


@contextmanager
def _prepare_lock(output_dir: Path):
    """Hold a process-scoped lock, automatically released on interruption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = (output_dir / '.jets-preparation.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b'\0'); lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another preparation is using this output') from exc
        else:
            fcntl = importlib.import_module('fcntl')
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another preparation is using this output') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()


def _verify_prepared_files(base: Path, files: dict):
    expected = {'prepared/population.pt', 'prepared/preprocessor.json',
                'prepared/parents.json', 'prepared/manifest.json',
                'bank/data.pt', 'bank/manifest.json'}
    if set(files) != expected:
        raise ValueError('Unexpected preparation receipt contents')
    if any((base / name).is_symlink() for name in ('prepared', 'bank')):
        raise ValueError('Preparation output directory cannot be a symlink')
    for relative, digest in files.items():
        path = base / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError('Prepared output changed after creation')


def _publish_transaction(output_dir: Path, identity: dict, journal: dict):
    """Recover a fully staged transaction without overwriting existing files."""
    if journal.get('identity') != identity or journal.get('receipt', {}).get('identity') != identity:
        raise ValueError('Pending preparation belongs to different inputs or parameters')
    name = journal.get('staging_dir')
    if not isinstance(name, str) or not name.startswith('.jets-prepare-') or Path(name).name != name:
        raise ValueError('Invalid preparation staging path')
    staging = output_dir / name
    if staging.is_symlink():
        raise ValueError('Preparation staging directory cannot be a symlink')
    files = journal['receipt']['files']
    if set(files) != {'prepared/population.pt', 'prepared/preprocessor.json',
                      'prepared/parents.json', 'prepared/manifest.json',
                      'bank/data.pt', 'bank/manifest.json'}:
        raise ValueError('Unexpected preparation receipt contents')
    for name in ('prepared', 'bank'):
        target = output_dir / name
        if target.is_symlink():
            raise ValueError('Preparation output directory cannot be a symlink')
        source = staging / name
        # The half already committed by a previous process is verified in
        # place. A missing half must match the staged receipt before rename.
        base = target if target.exists() else source
        if base.is_symlink() or not base.is_dir():
            raise ValueError('Preparation transaction directory is missing or invalid')
        for relative, digest in files.items():
            if relative.startswith(name + '/'):
                path = base / Path(relative).name
                if path.is_symlink() or not path.is_file() or sha256(path) != digest:
                    raise ValueError('Prepared output changed after creation')
        if not target.exists():
            os.rename(source, target)
    _verify_prepared_files(output_dir, files)
    complete = output_dir / 'preparation.json'
    if complete.exists():
        if read_json(complete) != journal['receipt']:
            raise ValueError('Existing preparation receipt differs')
    else:
        write_json(complete, journal['receipt'])
    # Remove only our own empty staging directory; never recursively delete.
    if staging.is_dir():
        try:
            staging.rmdir()
        except OSError:
            pass
    return journal['receipt']


def prepare_raw(raw_dir, output_dir, *, schema=None, seed=9, windows=4096,
                train_ids=None, observation_end=None):
    """Fit on selected entities and create a prepared population and window bank.

    ``raw_dir`` must contain aggregated ``account.csv`` and ``trans.csv``.
    Set schema ``id_dtype`` to ``string`` to preserve identifiers such as 001;
    the default ``infer`` retains the original numeric pandas inference. When
    using string IDs, train_ids must also use exactly matching strings.
    ``train_ids`` is an optional iterable of entity IDs; heldout entities are
    excluded from *all* fitted statistics, outputs and observation-end inference.
    Omission uses the complete population and is not heldout evaluation.
    Outputs are immutable once complete; identity mismatch fails closed.
    """
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if output_dir == raw_dir or output_dir.is_relative_to(raw_dir):
        raise ValueError('Preparation output must be outside the raw input directory')
    schema = RETAIL_SCHEMA if schema is None else schema
    if not isinstance(seed, int) or not isinstance(windows, int):
        raise ValueError('Seed and window count must be integers')
    identity = {'input_sha256': {n: sha256(raw_dir / n) for n in ('account.csv', 'trans.csv')},
        'schema': schema, 'seed': seed, 'windows': windows, 'train_ids': None if train_ids is None else list(train_ids),
        'observation_end': observation_end}
    with _prepare_lock(output_dir):
        complete = output_dir / 'preparation.json'
        if complete.exists():
            result = read_json(complete)
            if result['identity'] != identity:
                raise ValueError('Prepared output belongs to different inputs or parameters')
            _verify_prepared_files(output_dir, result['files'])
            return result
        journal_path = output_dir / '.jets-preparation-transaction.json'
        if journal_path.exists():
            return _publish_transaction(output_dir, identity, read_json(journal_path))
        if (output_dir / 'prepared').exists() or (output_dir / 'bank').exists():
            raise FileExistsError('Existing preparation without a verified transaction; use a new output directory')
        entities, end = _read_entities(raw_dir, schema, identity['train_ids'], observation_end)
        prep = QuantilePreprocessor(max_categories=4096, seed=seed, quantile_grid='all_observed').fit(entities)
        records = []
        for entity in entities:
            record = prep.transform(entity)
            record['calendar_times'] = torch.as_tensor(entity['calendar_times'], dtype=torch.float64)
            records.append(record)
        result = _window_bank(records, prep, end, windows, seed)
        staging = Path(tempfile.mkdtemp(prefix='.jets-prepare-', dir=output_dir))
        prepared, bank = staging / 'prepared', staging / 'bank'
        prepared.mkdir(); bank.mkdir()
        _save_torch(prepared / 'population.pt', records)
        write_json(prepared / 'preprocessor.json', prep.state_dict())
        parents = [e['entity_id'] for e in entities]
        write_json(prepared / 'parents.json', parents)
        write_json(prepared / 'manifest.json', {'population_sha256': sha256(prepared / 'population.pt'),
            'preprocessor_hash': json_digest(prep.state_dict()), 'parent_hash': json_digest(parents),
            'source': {'input_sha256': identity['input_sha256'], 'schema': schema, 'observation_end': end,
                       'scope': 'all supplied entities' if train_ids is None else 'selected training entities only'}})
        _save_torch(bank / 'data.pt', result)
        write_json(bank / 'manifest.json', {**result['manifest'], 'data_sha256': sha256(bank / 'data.pt'),
            'preprocessor_hash': json_digest(prep.state_dict()), 'parent_hash': json_digest(parents)})
        if any(sha256(raw_dir / name) != digest for name, digest in identity['input_sha256'].items()):
            raise RuntimeError('Source input changed during preparation')
        paths = [prepared / n for n in ('population.pt', 'preprocessor.json', 'parents.json', 'manifest.json')]
        paths += [bank / n for n in ('data.pt', 'manifest.json')]
        receipt = {'identity': identity, 'files': {str(path.relative_to(staging)).replace('\\', '/'): sha256(path) for path in paths},
            'entities': len(entities), 'events': sum(len(e['delta']) for e in entities),
            'observation_end': end, 'scope': 'full supplied population, not heldout' if train_ids is None else 'training subset only'}
        journal = {'identity': identity, 'staging_dir': staging.name, 'receipt': receipt}
        write_json(journal_path, journal)
        return _publish_transaction(output_dir, identity, journal)
