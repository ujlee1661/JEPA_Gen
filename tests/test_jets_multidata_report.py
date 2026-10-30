"""Scientific-report boundary tests on fabricated tiny native paths."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis.jets_multidata_report import SCORER_SHA256, aggregates, compare, run, verify_jets_pipeline
from analysis.collect_jets_cases import build
from analysis.collect_tabargn_cases import build as build_baseline_cases


def test_jets_case_manifest_covers_training_and_generation_seeds(tmp_path):
    data = build(tmp_path)
    assert len(data['samples']) == 36
    assert {case['dataset'] for case in data['samples']} == {'online_retail', 'berka', 'coupon', 'sparkov'}
    assert {case['training_seed'] for case in data['samples']} == {1, 2, 3}
    assert {case['generation_seed'] for case in data['samples']} == {1, 2, 3}
    assert all('generated N' in case['count_source'] for case in data['samples'])
    baseline = build_baseline_cases(tmp_path)
    assert len(baseline['samples']) == 36
    assert all(case['counts_value_column'] == 'N' for case in baseline['samples'])


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
    pd.DataFrame({'entity': [1, 2, 3], 'N': [2, 0, 1]}).to_csv(counts, index=False)
    assert compare(parent, real, child, ['log_amount'], ['cat_code'], counts,
                   counts_value_column='N')['count_ks_all_parents'] == 0.
    pd.DataFrame({'entity': [1, 2, 3], 'count': [1, 0, 1]}).to_csv(counts, index=False)
    with pytest.raises(ValueError, match='differs'):
        compare(parent, real, child, ['log_amount'], ['cat_code'], counts)


def test_generation_replicates_are_nested_in_train_seed():
    rows = []
    for seed, vals in [(1, [1., 3., 5.]), (2, [3., 5., 7.]), (3, [5., 7., 9.])]:
        for gen, value in enumerate(vals, 1):
            rows.append({'dataset': 'sample', 'model': 'JETS', 'training_seed': seed,
                'generation_seed': gen, 'count_source': 'generated', 'training_provenance_verified': True,
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


def test_generated_parent_population_is_not_forced_to_real_ids(tmp_path):
    parent=tmp_path/'real_parents.csv'; real=tmp_path/'real.csv'; child=tmp_path/'generated.csv'
    generated_parents=tmp_path/'generated_parents.csv'; counts=tmp_path/'counts.csv'
    pd.DataFrame({'account_id':[1,2,3]}).to_csv(parent,index=False)
    pd.DataFrame({'account_id':[1,1,2], 't_offset':[1.,2.,3.], 'x':[1.,2.,3.]}).to_csv(real,index=False)
    pd.DataFrame({'entity':['x','x','y'], 't_offset':[1.,2.,3.], 'x':[1.,2.,3.]}).to_csv(child,index=False)
    pd.DataFrame({'account_id':['x','y','z']}).to_csv(generated_parents,index=False)
    pd.DataFrame({'entity':['x','y','z'], 'count':[2,1,0]}).to_csv(counts,index=False)
    with pytest.raises(ValueError, match='identifiers do not match'):
        compare(parent,real,child,['x'],[])
    got=compare(parent,real,child,['x'],[],counts,parent_identity='generated',generated_parent_file=generated_parents)
    assert got['zero_event_parent_count_known'] and got['count_ks_all_parents']==0.
    via_counts=compare(parent,real,child,['x'],[],counts,parent_identity='generated')
    assert via_counts['count_ks_all_parents']==0.
    unknown=compare(parent,real,child,['x'],[],parent_identity='generated')
    assert not unknown['zero_event_parent_count_known'] and unknown['count_ks_all_parents'] is None
    assert unknown['count_ks_nonempty_parents']==0.


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


def test_in_progress_generation_waits_for_verified_receipt(tmp_path):
    raw=tmp_path/'data'/'dataset'/'raw';raw.mkdir(parents=True)
    pd.DataFrame({'account_id':[1]}).to_csv(raw/'account.csv',index=False)
    pd.DataFrame({'account_id':[1],'t_offset':[1.],'x':[2.]}).to_csv(raw/'trans.csv',index=False)
    child=tmp_path/'child.csv';pd.DataFrame({'entity':[1],'t_offset':[1.],'x':[2.]}).to_csv(child,index=False)
    counts=tmp_path/'counts.csv';receipt=tmp_path/'generation.json'
    case={'dataset':'dataset','model':'JETS','training_seed':1,'generation_seed':1,
          'count_source':'generated','child':str(child),'counts':str(counts),'receipt':str(receipt)}
    source=tmp_path/'cases.json';source.write_text(json.dumps({'schemas':{'dataset':{'numeric':['x'],'categorical':[]}},'samples':[case]}))
    result=run(tmp_path/'data',source,tmp_path/'waiting.json',allow_partial=True)
    assert not result['per_generation'] and len(result['pending'])==1
    pd.DataFrame({'entity':[1],'count':[1]}).to_csv(counts,index=False)
    receipt.write_text(json.dumps({'child_sha256':'incorrect','generation_seed':1}))
    with pytest.raises(ValueError, match='receipt does not match'):
        run(tmp_path/'data',source,tmp_path/'bad.json',allow_partial=True)


def test_training_provenance_checks_checkpoint_seed_and_budget(tmp_path):
    from experiments.data import sha256
    raw=tmp_path/'data'/'dataset'/'raw';raw.mkdir(parents=True)
    pd.DataFrame({'account_id':[1]}).to_csv(raw/'account.csv',index=False)
    pd.DataFrame({'account_id':[1],'t_offset':[1.],'x':[2.]}).to_csv(raw/'trans.csv',index=False)
    child=tmp_path/'child.csv';pd.DataFrame({'entity':[1],'t_offset':[1.],'x':[2.]}).to_csv(child,index=False)
    ckpt=tmp_path/'final.pt';ckpt.write_bytes(b'model')
    train=tmp_path/'complete.json';train.write_text(json.dumps({'checkpoint_sha256':sha256(ckpt),'updates':20000}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'config':{'training_seed':2}}))
    receipt=tmp_path/'generation.json';receipt.write_text(json.dumps({'child_sha256':sha256(child),
        'checkpoint_sha256':sha256(ckpt),'generation_seed':1}))
    case={'dataset':'dataset','model':'JETS','training_seed':1,'generation_seed':1,'count_source':'generated',
          'child':str(child),'receipt':str(receipt),'checkpoint':str(ckpt),'training_receipt':str(train),
          'training_manifest':str(manifest),'expected_updates':20000}
    source=tmp_path/'cases.json';source.write_text(json.dumps({'schemas':{'dataset':{'numeric':['x'],'categorical':[]}},'samples':[case]}))
    with pytest.raises(ValueError,match='different training seed'):
        run(tmp_path/'data',source,tmp_path/'wrong.json',allow_partial=True)
    manifest.write_text(json.dumps({'config':{'training_seed':1}}))
    result=run(tmp_path/'data',source,tmp_path/'valid.json',allow_partial=True)
    assert result['per_generation'][0]['training_provenance_verified']
    assert result['per_generation'][0]['training_checkpoint_sha256']==sha256(ckpt)
    train.write_text(json.dumps({'checkpoint_sha256':sha256(ckpt),'updates':10}))
    with pytest.raises(ValueError,match='update count'):
        run(tmp_path/'data',source,tmp_path/'budget.json',allow_partial=True)


def test_baseline_receipt_checks_all_weights_but_does_not_claim_generation_link(tmp_path):
    from experiments.data import sha256
    raw=tmp_path/'data'/'dataset'/'raw';raw.mkdir(parents=True)
    pd.DataFrame({'account_id':[1]}).to_csv(raw/'account.csv',index=False)
    pd.DataFrame({'account_id':[1],'t_offset':[1.],'x':[2.]}).to_csv(raw/'trans.csv',index=False)
    child=tmp_path/'child.csv';pd.DataFrame({'entity':[1],'t_offset':[1.],'x':[2.]}).to_csv(child,index=False)
    weights=tmp_path/'model.pt';weights.write_bytes(b'weights')
    evidence=tmp_path/'training_complete.json';evidence.write_text(json.dumps({'training_seed':1,
        'max_epochs':100, 'training_exit_code':0, 'weights':{'model.pt':sha256(weights)}}))
    receipt=tmp_path/'generation.json';receipt.write_text(json.dumps({'child_sha256':sha256(child),'generation_seed':1}))
    case={'dataset':'dataset','model':'TabARGN','training_seed':1,'generation_seed':1,
          'count_source':'generated','child':str(child),'receipt':str(receipt),'training_evidence':str(evidence),
          'training_budget_field':'max_epochs','expected_training_budget':100}
    source=tmp_path/'cases.json';source.write_text(json.dumps({'schemas':{'dataset':{'numeric':['x'],'categorical':[]}},'samples':[case]}))
    result=run(tmp_path/'data',source,tmp_path/'valid.json',allow_partial=True)
    assert result['per_generation'][0]['training_provenance_verified']
    assert not result['per_generation'][0]['generation_checkpoint_link_verified']
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='weight path or digest'):
        run(tmp_path/'data',source,tmp_path/'wrong.json',allow_partial=True)


def test_jets_pipeline_checks_both_encoder_seeds_and_cache_links(tmp_path):
    from experiments.data import sha256
    root=tmp_path/'prepared';root.mkdir()
    config={'training_seed':1,'generator_steps':20000,'batch_size':16}
    cfg={'seed':1,'teacher_steps':5000,'teacher_batch_size':32}
    source={'jets/model.py':'fixed'}; raw={'account.csv':'a','trans.csv':'b'}
    paths={'config_snapshot':root/'config.snapshot.json', 'bank':root/'input/bank/data.pt',
           'preprocessor':root/'input/prepared/preprocessor.json', 'parents':root/'input/prepared/parents.json',
           'time_target_encoder':root/'target_encoders/time/final.pt', 'time_cache':root/'target_caches/time.pt',
           'feature_target_encoder':root/'target_encoders/feature/final.pt', 'feature_cache':root/'target_caches/feature.pt',
           'initial_generator':root/'initial_generator.pt'}
    for name,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(config) if name=='config_snapshot' else name)
    artifact={name:{'path':str(path),'sha256':sha256(path)} for name,path in paths.items()}
    pipe={'seed':1,'windows':4096,'target_config':cfg,'train_ids':None,'input_sha256':raw,
          'config_sha256':artifact['config_snapshot']['sha256'],'source_sha256':source}
    (root/'pipeline.json').write_text(json.dumps(pipe))
    (root/'complete.json').write_text(json.dumps({'artifacts':artifact,'target_steps':5000,
        'exploratory_budget':False,'generation_future_target_inputs':False}))
    train={'config':config,'input_sha256':{str(paths[name].resolve()): artifact[name]['sha256'] for name in
        ('config_snapshot','bank','time_target_encoder','time_cache','feature_target_encoder','feature_cache','initial_generator')},
        'source_sha256':source,'feature_weight':1.,'stochastic_condition':False,'native_future_teacher_inputs':False}
    for family in ('time','feature'):
        checkpoint=artifact[family+'_target_encoder']['sha256']; cache=artifact[family+'_cache']['sha256']; bank=artifact['bank']['sha256']
        (root/'target_encoders'/family/'manifest.json').write_text(json.dumps({'family':family,'data_sha256':bank,
            'config':cfg,'source_sha256':source}))
        (root/'target_encoders'/family/'complete.json').write_text(json.dumps({'family':family,'updates':5000,
            'data_sha256':bank,'checkpoint_sha256':checkpoint,'target_encoder_frozen':True,'calibration_scope':'training windows only'}))
        (root/'target_caches'/f'{family}.json').write_text(json.dumps({'data_sha256':bank,
            'teacher_checkpoint_sha256':checkpoint,'cache_sha256':cache,'windows':4096,'rows':5000,
            'true_future_input_to_generator':False}))
    evidence=verify_jets_pipeline(root,1,raw,train)
    assert len(evidence)==17
    bad=json.loads((root/'target_encoders/feature/manifest.json').read_text());bad['config']['seed']=2
    (root/'target_encoders/feature/manifest.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='target encoder identity'):
        verify_jets_pipeline(root,1,raw,train)


def test_one_training_seed_cannot_mix_checkpoint_hashes():
    rows=[]
    for gen,digest in [(1,'a'),(2,'b')]:
        rows.append({'dataset':'dataset','model':'JETS','training_seed':1,'generation_seed':gen,
            'count_source':'generated','training_checkpoint_sha256':digest,'metrics':None,
            'distribution':{'events':1,'count_ks_all_parents':0.,'internal_gap_ks':0.,
                            'active_span_ks':0.,'numeric':{},'categorical':{}}})
    with pytest.raises(ValueError,match='same checkpoint'):
        aggregates(rows,[1],[1,2])
