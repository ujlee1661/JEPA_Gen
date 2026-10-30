"""Tests for explicit source-table conversion, never synthetic-data repair."""
from importlib.metadata import version
import json

import numpy as np
import pandas as pd
import pytest

from experiments.retail_source import build_benchmark_tables, prepare_retail_ii_benchmark


def sample_rows():
    rows = []
    for customer, country in ((101, 'UK'), (202, 'FR')):
        for invoice in range(21):
            rows.append({'Invoice': f'{customer}-{invoice}',
                         'InvoiceDate': pd.Timestamp('2010-01-01') + pd.Timedelta(days=invoice),
                         'Quantity': -2 if invoice == 0 else 2,
                         'Price': 3., 'Customer ID': customer, 'Country': country})
    # A customer with fewer invoices is outside the explicit benchmark cohort.
    rows.append({'Invoice': 'small', 'InvoiceDate': pd.Timestamp('2010-01-01'),
                 'Quantity': 1, 'Price': 1., 'Customer ID': 303, 'Country': 'US'})
    return pd.DataFrame(rows)


def test_benchmark_contract_aggregates_and_documents_source_rule():
    account, trans = build_benchmark_tables(sample_rows())
    assert len(account) == 2 and len(trans) == 42
    assert account['account_id'].tolist() == [0, 1]
    assert account['Country'].tolist() == [1., 0.]
    assert trans['trans_id'].tolist() == list(range(42))
    first = trans[trans.account_id == 0].reset_index(drop=True)
    assert first.log_amount.iloc[0] == 0.
    assert first.log_cumamt.iloc[0] == 0.
    np.testing.assert_allclose(first.log_amount.iloc[1:], np.log1p(6.))
    assert first.t_offset.iloc[0] == 31.


def test_missing_columns_and_no_eligible_customers_fail():
    with pytest.raises(ValueError, match='Missing input columns'):
        build_benchmark_tables(pd.DataFrame({'Invoice': [1]}))
    with pytest.raises(ValueError, match='No customer'):
        build_benchmark_tables(sample_rows().iloc[:2])


def test_workbook_conversion_is_idempotent_and_checks_hashes(tmp_path):
    pytest.importorskip('openpyxl')
    source, out = tmp_path / 'source.xlsx', tmp_path / 'tables'
    rows = sample_rows()
    with pd.ExcelWriter(source, engine='openpyxl') as writer:
        rows.iloc[:22].to_excel(writer, sheet_name='Year 2009-2010', index=False)
        rows.iloc[22:].to_excel(writer, sheet_name='Year 2010-2011', index=False)
    assert prepare_retail_ii_benchmark(source, out) == out.resolve()
    assert prepare_retail_ii_benchmark(source, out) == out.resolve()
    assert len(pd.read_csv(out / 'account.csv')) == 2
    receipt_path = out / 'source_manifest.json'
    original_receipt = receipt_path.read_text(encoding='utf-8')
    receipt = json.loads(original_receipt)
    assert receipt['identity']['openpyxl_version'] == version('openpyxl')
    receipt['identity']['openpyxl_version'] = 'different-version'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='different source or recipe'):
        prepare_retail_ii_benchmark(source, out)
    receipt_path.write_text(original_receipt, encoding='utf-8')
    original = source.read_bytes()
    source.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='different source'):
        prepare_retail_ii_benchmark(source, out)
    source.write_bytes(original)
    (out / 'trans.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        prepare_retail_ii_benchmark(source, out)


def test_partial_output_is_never_overwritten(tmp_path):
    source, out = tmp_path / 'source.xlsx', tmp_path / 'tables'
    source.write_bytes(b'not a workbook')
    out.mkdir()
    (out / 'account.csv').write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        prepare_retail_ii_benchmark(source, out)
    assert (out / 'account.csv').read_text(encoding='utf-8') == 'keep'
