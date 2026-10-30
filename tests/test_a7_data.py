from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pgjepa.a6_data import fit_token_state, encode_window
from pgjepa.a7_data import prepare_a7, collate_a7, inverse_marks, build_suffix_window, encode_a7_window
from pgjepa.a7_spacing import decode_spacing


def fixture_bank(tmp_path):
    prep=SimpleNamespace(n_num=2,cat_sizes=[3],attrs_dim=2,inverse_numeric=lambda x:np.asarray(x).copy())
    record={'x_num':torch.tensor([[0.,1.],[2.,3.],[4.,5.],[6.,7.]]),
        'x_cat':torch.tensor([[1],[2],[1],[0]]),'mask':torch.ones(4,3),
        'delta':torch.tensor([[0.],[0.],[2.],[3.]]),
        'calendar_times':torch.tensor([0.,0.,2.,5.],dtype=torch.float64),
        'exposure_end':torch.tensor(10.,dtype=torch.float64),'attrs':torch.ones(2)}
    state=fit_token_state([record],prep)
    windows=[]
    for cut,horizon in [(0,10.),(1,0.),(4,5.)]:
        item=encode_window(record,cut,horizon,state)
        item.update(entity=0,kind='bos_full' if cut==0 else 'prefix_endpoint')
        windows.append(item)
    source=tmp_path/'a6.pt'
    torch.save({'records':[record],'attrs':record['attrs'][None],
        'token_state':state,'windows':windows,'profile_state':{'DO_NOT_USE':'profile'},
        'manifest':{'test':'fixed_query_bank','query_measure':'fixture'}},source)
    return source,prep,record


def test_exact_queries_native_ties_terminal_and_zero_duration(tmp_path):
    source,prep,record=fixture_bank(tmp_path)
    data=prepare_a7(source,tmp_path/'a7')
    assert data['manifest']['profile_codes_used'] is False
    assert 'profile_state' not in data and 'code' not in data['windows'][0]
    full,tied,empty=data['windows']
    assert full['count']==4 and full['tokens'].shape==(5,9)
    assert full['active'].tolist()==[False,False,True,True,True]
    assert full['event_valid'].tolist()==[True,True,True,True,False]
    assert torch.count_nonzero(full['tokens'][-1,1:])==0
    decoded=decode_spacing(full['tokens'][:,0].double()*data['state']['time_scale'],full['active'],0.,10.)
    torch.testing.assert_close(decoded['times'],record['calendar_times'],rtol=1e-7,atol=1e-7)
    assert tied['count']==1 and tied['delta']==0 and not tied['active'].any()
    assert empty['count']==0 and empty['tokens'].shape==(1,9) and empty['active'].tolist()==[True]
    assert empty['tokens'][0,0]==0
    assert prepare_a7(source,tmp_path/'a7')['manifest']==data['manifest']


def test_collate_padding_and_native_mark_order(tmp_path):
    source,prep,record=fixture_bank(tmp_path)
    data=prepare_a7(source,tmp_path/'a7')
    batch=collate_a7(data['windows'])
    assert batch['tokens'].shape==(3,5,9)
    assert batch['valid'].sum(1).tolist()==[5,2,1]
    assert batch['event_valid'].sum(1).tolist()==[4,1,0]
    assert not batch['active'][1:].any(1)[0]  # δ0 tied future
    assert not batch['valid'][2,1:].any()
    assert batch['past_tokens'].shape==(3,4,9)
    assert batch['past_valid'].sum(1).tolist()==[0,1,4]
    assert not batch['past_tokens'][0].any()
    empty_past=collate_a7([data['windows'][0]])
    assert empty_past['past_tokens'].shape==(1,0,9)
    assert empty_past['past_valid'].shape==(1,0)
    decoded=inverse_marks(data['windows'][0]['marks'],prep,data['state']['mark_token_state'])
    np.testing.assert_allclose(decoded['x_num'],record['x_num'],atol=1e-6)
    np.testing.assert_array_equal(decoded['x_cat'],record['x_cat'])
    assert 'order' not in decoded  # No sort or permutation operation exists here.


def test_suffix_recomputes_intrinsic_coordinates_and_extends_true_prefix(tmp_path):
    source,_,_=fixture_bank(tmp_path)
    data=prepare_a7(source,tmp_path/'a7')
    original=data['windows'][0]
    suffix=build_suffix_window(original,2)
    assert suffix['anchor']==0 and suffix['delta']==10 and suffix['count']==2
    assert suffix['past_tokens'].shape[0]==2
    assert suffix['times'].tolist()==[2.,5.]
    # Removing two zero entries may leave active CLR unchanged, but a positive
    # first interval removed changes the geometric mean and all surviving CLRs.
    deeper=build_suffix_window(original,3)
    assert deeper['anchor']==2. and deeper['delta']==8. and deeper['count']==1
    assert not torch.equal(deeper['clr'],original['clr'][3:])
    decoded=decode_spacing(deeper['clr'],deeper['active'],deeper['anchor'],deeper['delta'])
    torch.testing.assert_close(decoded['times'],torch.tensor([5.],dtype=torch.float64))
    final=build_suffix_window(original,4)
    assert final['count']==0 and final['active'].tolist()==[True] and final['delta']==5.
    with pytest.raises(ValueError):build_suffix_window(original,5)


def test_timing_scale_is_single_rms_over_active_training_coordinates(tmp_path):
    source,_,_=fixture_bank(tmp_path)
    data=prepare_a7(source,tmp_path/'a7')
    active=torch.cat([w['clr'][w['active']] for w in data['windows']])
    expected=float(active.square().mean().sqrt())
    assert data['state']['time_scale']==pytest.approx(expected)
    for w in data['windows']:
        assert abs(float(w['tokens'][w['active'],0].sum()))<1e-6


def test_train_and_other_namespaces_keep_independent_time_calibration(tmp_path):
    source,_,_=fixture_bank(tmp_path)
    first=prepare_a7(source,tmp_path/'a7')
    changed=torch.load(source,weights_only=False)
    changed['windows']=changed['windows'][1:]
    other=tmp_path/'other_a6.pt'
    torch.save(changed,other)
    second=prepare_a7(other,tmp_path/'a7_other')
    assert second['state']['time_scale']==1.  # only singleton/empty faces remain
    assert first['state']['time_scale']!=second['state']['time_scale']
    with pytest.raises(ValueError,match='identity changed'):
        prepare_a7(other,tmp_path/'a7')


def test_heldout_query_uses_frozen_supplied_train_scale(tmp_path):
    source,_,record=fixture_bank(tmp_path)
    data=prepare_a7(source,tmp_path/'a7')
    fixed=data['state']['time_scale']
    changed={**record,'calendar_times':torch.tensor([0.,1.,4.,9.],dtype=torch.float64)}
    encoded=encode_a7_window(changed,1,10.,data['state'],{'entity_id':99,'split':'heldout'})
    assert encoded['time_scale']==fixed and data['state']['time_scale']==fixed
    assert encoded['entity_id']==99 and encoded['count']==3
    assert encoded['past_tokens'].shape[0]==1


def test_actual_heldout_adapter_never_fits_test_timing_scale(tmp_path):
    from pgjepa.a7_data import prepare_a7_heldout
    original,prep,record=fixture_bank(tmp_path)
    source=tmp_path/'source'
    (source/'train_a6').mkdir(parents=True)
    train=torch.load(original,weights_only=False)
    torch.save(train,source/'train_a6/data.pt')
    changed={**record,'calendar_times':torch.tensor([0.,1e-8,9.,9.999],dtype=torch.float64)}
    q=encode_window(changed,1,10.,train['token_state'])
    q.update(entity=0,entity_id=99,prefix_fraction=.25,horizon_fraction=1.,split='heldout')
    torch.save({'records':[changed],'entity_ids':[99],'attrs':changed['attrs'][None],
                'queries':[q],'preprocessor_hash':'frozen_train_prep'},source/'heldout_queries.pt')
    (source/'split_manifest.json').write_text('{}')
    manifest=prepare_a7_heldout(source,tmp_path/'heldout_a7')
    training=torch.load(tmp_path/'heldout_a7/train_a7/data.pt',weights_only=False)
    heldout=torch.load(tmp_path/'heldout_a7/heldout_queries.pt',weights_only=False)
    assert manifest['time_scale_fit_split']=='train bank only'
    assert heldout['state']['time_scale']==training['state']['time_scale']
    assert heldout['queries'][0]['time_scale']==training['state']['time_scale']
    assert heldout['queries'][0]['entity_id']==99
    assert manifest['heldout_queries']==1 and heldout['queries'][0]['count']==3
    reread=prepare_a7_heldout(source,tmp_path/'heldout_a7')
    assert manifest==reread
