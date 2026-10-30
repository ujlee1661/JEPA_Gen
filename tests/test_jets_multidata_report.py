"""Scientific-report boundary tests on fabricated tiny native paths."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis.jets_multidata_report import SCORER_SHA256, aggregates, compare, run
from analysis.collect_jets_cases import build


def test_jets_case_manifest_covers_training_and_generation_seeds(tmp_path):
    data = build(tmp_path)
    assert len(data['samples']) == 36
    assert {case['dataset'] for case in data['samples']} == {'online_retail', 'berka', 'coupon', 'sparkov'}
    assert {case['training_seed'] for case in data['samples']} == {1, 2, 3}
    assert {case['generation_seed'] for case in data['samples']} == {1, 2, 3}
    assert all('generated N' in case['count_source'] for case in data['samples'])


def test_native_count_gaps_and_mixed_column_distributions(tmp_path):
    parent = tmp_path/'account.csv'; real = tmp_path/'trans.csv'; child = tmp_path/'child.csv'
    counts = tmp_path/'counts.csv'
    pd.DataFrame({'account_id': [1, 2, 3]}).to_csv(parent, index=False)
    pd.DataFrame({'account_id': [1, 1, 2], 't_offset': [1., 3., 2.],
                  'log_amount': np.log1p([0., 10., 5.]), 'cat_code': [1, 2, 1]}).to_csv(real, index=False)
    pd.DataFrame({'entity': [1, 1, 3], 't_offset': [1., 4., 1.],
                  'log_amount': np.log1p([0., 8., 6.]), 'cat_code': [1, 2, 2]}).to_csv(child, index=False)
    pd.DataFrame({'entity': [1, 2, 3], 'count': [2, 0, 1]}).to_csv(counts, index=False)
    result = compare(parent, real, child, ['log_amount'], ['cat_code'], counts)
    assert result['parents'] == 3 and result['events'] == 3
    assert result['count_ks_all_parents'] == pytest.approx(0.)
    assert result['internal_gap_ks'] == pytest.approx(1.)
    assert result['numeric']['log_amount']['expm1_generated']['mean'] == pytest.approx(14/3)
    assert result['numeric']['log_amount']['expm1_generated']['zero_fraction'] == pytest.approx(1/3)
    assert result['categorical']['cat_code']['observed_total_variation'] == pytest.approx(1/3)
    pd.DataFrame({'entity': [1, 2, 3], 'count': [1, 0, 1]}).to_csv(counts, index=False)
    with pytest.raises(ValueError, match='differs'):
        compare(parent, real, child, ['log_amount'], ['cat_code'], counts)


def test_generation_replicates_are_nested_in_train_seed():
    rows = []
    for seed, vals in [(1, [1., 3., 5.]), (2, [3., 5., 7.]), (3, [5., 7., 9.])]:
        for gen, value in enumerate(vals, 1):
            rows.append({'dataset': 'sample', 'model': 'JETS', 'training_seed': seed,
                'generation_seed': gen, 'count_source': 'generated',
                'metrics': {'ts': value, 'cs': value, 'sd': value, 'lg': value, 'joint_ff': value},
                'distribution': {'events': 1, 'count_ks_all_parents': value,
                    'internal_gap_ks': value, 'active_span_ks': value,
                    'numeric': {'log_amount': {'expm1_generated': {'mean': value, 'std': 1.,
                        'p99': value, 'zero_fraction': 0., 'missing': 0}}}, 'categorical': {}}})
    train, overall = aggregates(rows, [1, 2, 3], [1, 2, 3])
    assert [r['ts_mean'] for r in train] == [3., 5., 7.]
    assert overall[0]['complete'] and overall[0]['ts_mean'] == 5.
    assert overall[0]['ts_train_seed_sd'] == pytest.approx(2.)
    assert overall[0]['numeric.log_amount.expm1_generated.mean_mean'] == 5.
    assert overall[0]['numeric.log_amount.expm1_generated.mean_train_seed_sd'] == pytest.approx(2.)


def test_partial_outputs_and_original_metric_digest(tmp_path):
    raw = tmp_path/'data'/'dataset'/'raw'; raw.mkdir(parents=True)
    pd.DataFrame({'account_id': [1]}).to_csv(raw/'account.csv', index=False)
    pd.DataFrame({'account_id': [1], 't_offset': [1.], 'log_amount': [0.]}).to_csv(raw/'trans.csv', index=False)
    child=tmp_path/'child.csv'; pd.DataFrame({'entity': [1], 't_offset': [1.], 'log_amount': [0.]}).to_csv(child,index=False)
    metrics=tmp_path/'metrics.json';metrics.write_text(json.dumps({'scorer_sha256': SCORER_SHA256, 'metrics': {'ts': .9}}))
    cases={'schemas': {'dataset': {'numeric': ['log_amount'], 'categorical': []}}, 'samples': [
        {'dataset':'dataset','model':'JETS','training_seed':1,'generation_seed':1,'count_source':'generated',
         'child':str(child),'metrics':str(metrics)},
        {'dataset':'dataset','model':'JETS','training_seed':1,'generation_seed':2,'count_source':'generated',
         'child':str(tmp_path/'pending.csv')} ]}
    source=tmp_path/'cases.json';source.write_text(json.dumps(cases))
    with pytest.raises(FileNotFoundError):
        run(tmp_path/'data', source, tmp_path/'strict.json')
    result=run(tmp_path/'data',source,tmp_path/'partial.json',allow_partial=True)
    assert result['partial'] and len(result['pending'])==1 and not result['overall'][0]['complete']
    assert result['per_generation'][0]['metric_link'].startswith('scorer verified')
    with pytest.raises(FileExistsError):
        run(tmp_path/'data',source,tmp_path/'partial.json',allow_partial=True)
