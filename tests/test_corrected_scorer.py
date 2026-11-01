import numpy as np
import pandas as pd
import pytest
import hashlib
from pathlib import Path
from types import SimpleNamespace

import analysis.corrected_scorer as corrected_scorer
from analysis.corrected_scorer import category_key, categorical_tv, default_scorer, evaluate_paths


def test_numeric_scalar_equivalence_preserves_text_labels_and_missing():
    assert category_key(0) == category_key(0.0) == category_key(np.int64(0))
    assert category_key(2) == category_key(np.float64(2.0))
    assert category_key('0') != category_key('00')
    assert category_key('0') != category_key(0)
    assert category_key('1.0') != category_key('1')
    assert category_key(np.nan) == category_key(None)


def test_tv_counts_numeric_equivalence_and_missing():
    real = pd.DataFrame({'day': [0, 0, 0], 'cat': [0, 1, None]})
    syn = pd.DataFrame({'day': [0, 0, 0], 'cat': [0.0, 1.0, np.nan]})
    assert categorical_tv(real.groupby('day'), syn.groupby('day'), [0], 'cat') == 1.0


def test_tv_does_not_conflate_actual_text_categories_and_ignores_unshared_days():
    real = pd.DataFrame({'day': [0, 1], 'cat': ['0', '01']})
    syn = pd.DataFrame({'day': [0, 2], 'cat': ['00', '01']})
    assert categorical_tv(real.groupby('day'), syn.groupby('day'), [0, 1], 'cat') == 0.0


def test_tv_drops_missing_instead_of_counting_it_as_a_category():
    real = pd.DataFrame({'day': [0, 0], 'cat': [0.0, np.nan]})
    syn = pd.DataFrame({'day': [0, 0], 'cat': [0.0, 0.0]})
    assert categorical_tv(real.groupby('day'), syn.groupby('day'), [0], 'cat') == 1.0


def test_tv_empty_after_dropping_missing_is_not_scored():
    real = pd.DataFrame({'day': [0], 'cat': [None]})
    syn = pd.DataFrame({'day': [0], 'cat': [0]})
    assert np.isnan(categorical_tv(real.groupby('day'), syn.groupby('day'), [0], 'cat'))


def test_explicit_real_path_and_string_identifiers_are_used(tmp_path):
    real = tmp_path / 'test.csv'
    child = tmp_path / 'generated.csv'
    real.write_text('account_id,t_offset\n001,0\n', encoding='utf-8')
    child.write_text('entity,t_offset\n001,0\n', encoding='utf-8')
    seen = []

    def paired(r, s):
        seen.append((r.account_id.tolist(), s.entity.tolist()))
        return 0.5

    def dataset_paired(dataset, r, s):
        assert dataset == 'online_retail'
        return paired(r, s)

    scorer = SimpleNamespace(ts=paired, cs=dataset_paired, sd=dataset_paired,
                             lg=dataset_paired, jointff=paired)
    result = evaluate_paths(scorer, 'online_retail', real, child, id_dtype='string')
    assert result == dict.fromkeys(('ts', 'cs', 'sd', 'lg', 'joint_ff'), 0.5)
    assert len(seen) == 5
    assert all(r == ['001'] and s == ['001'] for r, s in seen)


def test_missing_or_empty_identifiers_are_rejected(tmp_path):
    real = tmp_path / 'test.csv'
    child = tmp_path / 'generated.csv'
    real.write_text('account_id,t_offset\n1,0\n', encoding='utf-8')
    child.write_text('not_entity,t_offset\n1,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Missing required identifier'):
        evaluate_paths(None, 'online_retail', real, child)
    child.write_text('entity,t_offset\n,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid empty identifier'):
        evaluate_paths(None, 'online_retail', real, child, id_dtype='string')


def test_reference_location_is_configurable(monkeypatch, tmp_path):
    selected = tmp_path / 'score.py'
    monkeypatch.setenv('JETS_REFERENCE_SCORER', str(selected))
    assert default_scorer() == selected


def test_automatic_reference_discovery_requires_a_unique_pinned_file(monkeypatch, tmp_path):
    monkeypatch.delenv('JETS_REFERENCE_SCORER', raising=False)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(corrected_scorer, '__file__', str(tmp_path / 'project/analysis/corrected_scorer.py'))
    content = b'pinned scorer'
    monkeypatch.setattr(corrected_scorer, 'SCORER_SHA256', hashlib.sha256(content).hexdigest())
    first = tmp_path / 'run1/reference/eval/score.py'
    first.parent.mkdir(parents=True)
    first.write_bytes(content)
    assert default_scorer() == first.resolve()
    second = tmp_path / 'run2/reference/eval/score.py'
    second.parent.mkdir(parents=True)
    second.write_bytes(content)
    with pytest.raises(FileNotFoundError, match='one pinned reference scorer'):
        default_scorer()
