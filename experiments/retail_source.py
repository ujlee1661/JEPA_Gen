"""Explicit Online Retail II benchmark-table conversion.

This optional adapter reproduces the archived benchmark input contract; it is
not a model rule or a generated-value correction. The source transaction rows
are aggregated by customer and invoice, invoice amounts below zero are set to
zero before log1p, and customers with 21--800 invoices are eligible. At most
500 eligible customers are selected with seed 9; country codes are factorized
on that selected population. These choices are part of this source dataset's
benchmark definition and should not be silently applied to other datasets.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import read_json, sha256, write_json


SHEETS = ('Year 2009-2010', 'Year 2010-2011')
COLUMNS = ('Invoice', 'InvoiceDate', 'Quantity', 'Price', 'Customer ID', 'Country')
RECIPE = 'online_retail_ii_benchmark_v1'


def _read_workbook(source: Path) -> pd.DataFrame:
    try:
        with pd.ExcelFile(source, engine='openpyxl') as workbook:
            if tuple(workbook.sheet_names) != SHEETS:
                raise ValueError(f'Expected workbook sheets in this order: {SHEETS}')
            return pd.concat(
                [pd.read_excel(workbook, sheet_name=name, usecols=list(COLUMNS)) for name in SHEETS],
                ignore_index=True,
            )
    except ImportError as error:
        raise ImportError('Online Retail II XLSX input requires the optional openpyxl package') from error


def build_benchmark_tables(source_rows: pd.DataFrame):
    """Apply the established source-table recipe, not post-generation clipping."""
    missing = set(COLUMNS).difference(source_rows.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    frame = source_rows.loc[:, COLUMNS].dropna(subset=['Customer ID']).copy()
    frame['InvoiceDate'] = pd.to_datetime(frame['InvoiceDate'], errors='coerce')
    frame = frame.dropna(subset=['InvoiceDate'])
    frame['amt'] = frame['Quantity'] * frame['Price']
    invoices = (frame.groupby(['Customer ID', 'Invoice'])
                .agg(Date=('InvoiceDate', 'first'), amt=('amt', 'sum'), Country=('Country', 'first'))
                .reset_index().sort_values(['Customer ID', 'Date']))
    eligible = [key for key, group in invoices.groupby('Customer ID') if 21 <= len(group) <= 800]
    if not eligible:
        raise ValueError('No customer satisfies the benchmark eligibility contract (21--800 invoices)')
    rng = np.random.default_rng(9)
    indices = np.sort(rng.choice(len(eligible), size=min(500, len(eligible)), replace=False))
    anchor = pd.Timestamp('2009-12-01')
    parents: list[tuple] = []
    events: list[tuple] = []
    for new_id, index in enumerate(indices):
        group = invoices[invoices['Customer ID'] == eligible[index]].sort_values('Date')
        parents.append((new_id, group.Country.iloc[0]))
        amount = np.clip(group.amt.to_numpy(float), 0, None)
        times = ((group.Date - anchor).dt.total_seconds() / 86400.).to_numpy(float)
        events.extend(zip(np.full(len(group), new_id), times, np.log1p(amount), np.log1p(np.cumsum(amount))))
    account = pd.DataFrame(parents, columns=['account_id', 'Country'])
    account['Country'] = pd.factorize(account.Country, sort=True)[0].astype(float)
    trans = pd.DataFrame(events, columns=['account_id', 't_offset', 'log_amount', 'log_cumamt'])
    trans = trans[trans.t_offset >= 0].sort_values(['account_id', 't_offset']).reset_index(drop=True)
    trans.insert(0, 'trans_id', range(len(trans)))
    return account, trans


def prepare_retail_ii_benchmark(source, output_dir):
    """Convert a trusted original XLSX to immutable benchmark entity/event CSVs."""
    source, out = Path(source).resolve(), Path(output_dir).resolve()
    if source.suffix.lower() != '.xlsx':
        raise ValueError('Online Retail II benchmark input must be an XLSX workbook')
    receipt_path = out / 'source_manifest.json'
    if not receipt_path.exists() and any((out / name).exists() for name in ('account.csv', 'trans.csv')):
        raise FileExistsError('Incomplete or existing source tables; choose a new output directory')
    try:
        engine_version = version('openpyxl')
    except PackageNotFoundError as error:
        raise ImportError('Online Retail II XLSX input requires the optional openpyxl package') from error
    identity = {'recipe': RECIPE, 'source_sha256': sha256(source), 'sheets': list(SHEETS),
                'converter_sha256': sha256(Path(__file__)), 'openpyxl_version': engine_version,
                'numpy_version': np.__version__, 'pandas_version': pd.__version__}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get('identity') != identity:
            raise ValueError('Existing benchmark tables belong to a different source or recipe')
        for name, expected in receipt['files'].items():
            if sha256(out / name) != expected:
                raise ValueError('Existing benchmark tables changed after creation')
        return out
    account, trans = build_benchmark_tables(_read_workbook(source))
    if sha256(source) != identity['source_sha256']:
        raise RuntimeError('Source workbook changed during conversion')
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (('account.csv', account), ('trans.csv', trans)):
        temp = out / (name + '.tmp')
        table.to_csv(temp, index=False)
        temp.replace(out / name)
    write_json(receipt_path, {
        'identity': identity,
        'files': {name: sha256(out / name) for name in ('account.csv', 'trans.csv')},
        'entities': len(account), 'events': len(trans),
        'definition': 'two workbook sheets; per-customer invoice aggregation; '
                      '21--800 invoices; at most 500 customers, seed 9; '
                      'negative invoice totals clipped before log1p; calendar anchor 2009-12-01',
        'scope': 'benchmark source preprocessing, not synthetic-output adjustment',
    })
    return out
