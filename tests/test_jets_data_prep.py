import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.data import load_bank, read_json
import experiments.data_prep as data_prep
from experiments.data_prep import prepare_raw
from jets.data import build_suffix_window
from jets.data import QuantilePreprocessor


def raw_tables(path):
    path.mkdir()
    # Numeric categories force pandas iterrows to promote integer IDs to float.
    pd.DataFrame({'account_id': [1, 2], 'Country': [0., 1.]}).to_csv(path/'account.csv', index=False)
    pd.DataFrame({'trans_id': [2, 1, 3], 'account_id': [1, 1, 2],
        't_offset': [2., 1., 9.], 'log_amount': [1., 0., 99.],
        'log_cumamt': [1., 0., 99.]}).to_csv(path/'trans.csv', index=False)


def test_train_only_preparation_and_suffix(tmp_path):
    raw = tmp_path/'raw'; raw_tables(raw)
    output = tmp_path/'out'
    receipt = prepare_raw(raw, output, train_ids=[1], windows=4, seed=9)
    assert receipt['entities'] == 1 and receipt['events'] == 2
    assert receipt['observation_end'] == 2.  # heldout event at t=9 does not set the training horizon
    assert read_json(output/'prepared/parents.json') == [1]
    state = read_json(output/'prepared/preprocessor.json')
    prep = QuantilePreprocessor.from_state_dict(state)
    assert prep.quantile_models[0].quantiles_[-1, 0] == 1.  # heldout 99 excluded
    _, bank = load_bank(output/'bank')
    assert len(bank['windows']) == 4
    bos = next(w for w in bank['windows'] if w['kind'] == 'bos_full')
    assert bos['count'] == 2 and torch.equal(bos['times'], torch.tensor([1., 2.], dtype=torch.float64))
    suffix = build_suffix_window(bos, 1)
    assert suffix['count'] == 1 and len(suffix['past_tokens']) == 1
    np.testing.assert_allclose(suffix['gaps'].numpy(), [1., 0.])
    assert prepare_raw(raw, output, train_ids=[1], windows=4, seed=9) == receipt
    with pytest.raises(ValueError, match='different inputs'):
        prepare_raw(raw, output, train_ids=[1, 2], windows=4, seed=9)


def test_full_population_and_changed_output_fail_closed(tmp_path):
    raw = tmp_path/'raw'; raw_tables(raw)
    out = tmp_path/'out'
    receipt = prepare_raw(raw, out, windows=4, seed=12)
    assert receipt['entities'] == 2 and receipt['observation_end'] == 9.
    assert receipt['scope'] == 'full supplied population, not heldout'
    parents = read_json(out/'prepared/parents.json')
    assert parents == [1, 2]
    assert all(type(value) is int for value in parents)
    prep_manifest = read_json(out/'prepared/manifest.json')
    assert len(prep_manifest['preprocessor_hash']) == 64
    (out/'prepared/parents.json').write_text(json.dumps([2, 1]), encoding='utf8')
    with pytest.raises(ValueError, match='changed after creation'):
        prepare_raw(raw, out, windows=4, seed=12)


def test_reject_unknown_train_ids_and_schema(tmp_path):
    raw = tmp_path/'raw'; raw_tables(raw)
    with pytest.raises(ValueError, match='Training IDs'):
        prepare_raw(raw, tmp_path/'out1', windows=4, train_ids=[999])
    with pytest.raises(ValueError, match='Schema'):
        prepare_raw(raw, tmp_path/'out2', windows=4, schema={'numeric': ['log_amount']})


def test_large_integer_ids_not_rounded_by_numeric_parent_rows(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    ids = [9007199254740992, 9007199254740993]
    pd.DataFrame({'account_id': ids, 'Country': [0., 1.]}).to_csv(raw/'account.csv', index=False)
    pd.DataFrame({'trans_id': [1, 2], 'account_id': ids, 't_offset': [1., 2.],
                  'log_amount': [0., 1.], 'log_cumamt': [0., 1.]}).to_csv(raw/'trans.csv', index=False)
    prepare_raw(raw, tmp_path/'out', windows=4)
    assert read_json(tmp_path/'out/prepared/parents.json') == ids
    _, bank = load_bank(tmp_path/'out/bank')
    assert len(bank['records']) == 2
    assert bank['records'][0]['x_num'][0, 0] != bank['records'][1]['x_num'][0, 0]


def test_explicit_string_ids_preserve_leading_zero_and_na(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    pd.DataFrame({'account_id': ['001', '1', 'NA'], 'Country': [0., 1., 2.]}).to_csv(raw/'account.csv', index=False)
    pd.DataFrame({'trans_id': ['2', '010', '3', '4'], 'account_id': ['001', '001', '1', 'NA'],
                  't_offset': [1., 1., 2., 3.], 'log_amount': [2., 1., 4., 5.],
                  'log_cumamt': [2., 1., 4., 5.]}).to_csv(raw/'trans.csv', index=False)
    schema = {**data_prep.RETAIL_SCHEMA, 'id_dtype': 'string'}
    out = tmp_path/'out'
    prepare_raw(raw, out, windows=4, schema=schema, train_ids=['001', 'NA'])
    assert read_json(out/'prepared/parents.json') == ['001', 'NA']
    _, bank = load_bank(out/'bank')
    # Tied timestamps use lexical event IDs: '010' sorts before '2'.
    values = bank['records'][0]['x_num'][:, 0].numpy()
    assert len(values) == 2 and values[0] < values[1]
    assert len(bank['records'][1]['calendar_times']) == 1
    assert data_prep.prepare_raw(raw, out, windows=4, schema=schema, train_ids=['001', 'NA'])['entities'] == 2
    with pytest.raises(ValueError, match='Training IDs'):
        prepare_raw(raw, tmp_path/'wrong', windows=4, schema=schema, train_ids=[1])


def test_explicit_string_ids_reject_blank_and_unknown_mode(tmp_path):
    raw = tmp_path/'raw'; raw_tables(raw)
    (raw/'account.csv').write_text('account_id,Country\n,0\n1,1\n', encoding='utf8')
    with pytest.raises(ValueError, match='identifiers cannot be blank'):
        prepare_raw(raw, tmp_path/'out1', windows=4, schema={**data_prep.RETAIL_SCHEMA, 'id_dtype': 'string'})
    with pytest.raises(ValueError, match='id_dtype'):
        prepare_raw(raw, tmp_path/'out2', windows=4, schema={**data_prep.RETAIL_SCHEMA, 'id_dtype': 'float'})


def test_interruption_during_publish_recovers_without_overwrite(tmp_path, monkeypatch):
    raw = tmp_path/'raw'; raw_tables(raw)
    out = tmp_path/'out'
    rename = data_prep.os.rename
    def fail_bank_once(src, dst):
        if Path(src).name == 'bank':
            raise RuntimeError('simulated interruption while publishing bank')
        return rename(src, dst)
    monkeypatch.setattr(data_prep.os, 'rename', fail_bank_once)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        prepare_raw(raw, out, windows=4)
    assert (out/'prepared/parents.json').exists() and not (out/'bank').exists()
    prior_parent_bytes = (out/'prepared/parents.json').read_bytes()
    with pytest.raises(ValueError, match='different inputs'):
        prepare_raw(raw, out, windows=4, seed=123)
    journal_path = out/'.jets-preparation-transaction.json'
    original_journal = journal_path.read_text(encoding='utf8')
    altered = json.loads(original_journal)
    altered['receipt']['identity']['seed'] = 999
    journal_path.write_text(json.dumps(altered), encoding='utf8')
    with pytest.raises(ValueError, match='different inputs'):
        prepare_raw(raw, out, windows=4)
    journal_path.write_text(original_journal, encoding='utf8')
    monkeypatch.setattr(data_prep.os, 'rename', rename)
    result = prepare_raw(raw, out, windows=4)
    assert (out/'prepared/parents.json').read_bytes() == prior_parent_bytes
    assert (out/'bank/data.pt').is_file() and prepare_raw(raw, out, windows=4) == result


def test_interruption_before_journal_keeps_unfinished_staging(tmp_path, monkeypatch):
    raw = tmp_path/'raw'; raw_tables(raw)
    out = tmp_path/'out'
    save = data_prep._save_torch
    def fail_bank(path, value):
        if path.parent.name == 'bank':
            raise RuntimeError('simulated failure before journal')
        return save(path, value)
    monkeypatch.setattr(data_prep, '_save_torch', fail_bank)
    with pytest.raises(RuntimeError, match='simulated failure'):
        prepare_raw(raw, out, windows=4)
    stale = list(out.glob('.jets-prepare-*'))
    assert len(stale) == 1 and (stale[0]/'prepared/parents.json').is_file()
    assert not (out/'prepared').exists() and not (out/'bank').exists()
    monkeypatch.setattr(data_prep, '_save_torch', save)
    prepare_raw(raw, out, windows=4)
    assert (stale[0]/'prepared/parents.json').is_file()  # no recursive cleanup
    assert (out/'bank/data.pt').is_file()


def test_preexisting_unowned_output_is_not_replaced(tmp_path):
    raw = tmp_path/'raw'; raw_tables(raw)
    out = tmp_path/'out'; (out/'bank').mkdir(parents=True)
    marker = out/'bank/user-file.txt'; marker.write_text('keep', encoding='utf8')
    with pytest.raises(FileExistsError, match='without a verified transaction'):
        prepare_raw(raw, out, windows=4)
    assert marker.read_text(encoding='utf8') == 'keep'


def test_preparation_lock_rejects_concurrent_owner_and_releases(tmp_path):
    out = tmp_path/'out'
    with data_prep._prepare_lock(out):
        with pytest.raises(RuntimeError, match='Another preparation'):
            with data_prep._prepare_lock(out):
                pass
    with data_prep._prepare_lock(out):
        assert (out/'.jets-preparation.lock').is_file()
