"""A7 native spacing/mark windows adapted from an immutable A6 query bank.

Timing uses N+1 simplex gaps and a train-fitted scalar CLR scale. Mark values
retain the exact old quantile/type normalization. No profile code is consumed,
and neither event order nor generated values are repaired by this adapter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .a6_data import tokenize_sequence
from .a7_spacing import encode_record_window, encode_spacing
from .experiment import digest, write_json
from .paper_data import sha256
from .a6_heldout_data import calibration_digest


def _complete_window(raw, time_scale, metadata=None):
    count=int(raw['count'])
    marks=torch.as_tensor(raw['marks'],dtype=torch.float32).detach().cpu().clone()
    if marks.ndim!=2 or len(marks)!=count:
        raise ValueError('One native-order mark row per event required')
    clr=torch.as_tensor(raw['clr'],dtype=torch.float64).detach().cpu().clone()
    active=torch.as_tensor(raw['active'],dtype=torch.bool).detach().cpu().clone()
    if len(clr)!=count+1 or active.shape!=clr.shape or not np.isfinite(time_scale) or time_scale<=0:
        raise ValueError('Valid N+1 CLR/face and positive scalar time scale required')
    tokens=torch.zeros(count+1,1+marks.shape[1],dtype=torch.float32)
    tokens[:,0]=(clr/time_scale).float()
    tokens[:count,1:]=marks
    event_valid=torch.arange(count+1)<count
    result={**(metadata or {}),'count':count,'anchor':float(raw['anchor']),'delta':float(raw['delta']),
        'attrs':torch.as_tensor(raw['attrs']).detach().cpu().clone(),
        'tokens':tokens,'valid':torch.ones(count+1,dtype=torch.bool),
        'event_valid':event_valid,'active':active,'marks':marks,
        'times':torch.as_tensor(raw['times'],dtype=torch.float64).detach().cpu().clone(),
        'gaps':torch.as_tensor(raw['gaps'],dtype=torch.float64).detach().cpu().clone(),
        'clr':clr,'time_scale':float(time_scale),
        'past_tokens':torch.as_tensor(raw['past_tokens']).detach().cpu().clone()}
    if count==0 and (len(active)!=1 or bool(active[0]) != (result['delta']>0)):
        raise ValueError('Empty paths have one terminal gap, active exactly for positive duration')
    if result['delta']==0 and bool(active.any()):
        raise ValueError('Zero-duration paths have no active time degrees of freedom')
    return result


def encode_a7_window(record, cut, horizon, state, metadata=None):
    """Encode train/test queries using a supplied already-fitted training scale."""
    token_state=state['mark_token_state']
    raw=encode_record_window(record,cut,horizon,token_state)
    result=_complete_window(raw,state['time_scale'],{'cut':int(cut),**(metadata or {})})
    result['past_extension_tokens']=tokenize_sequence(record,token_state,0.,
        token_state['observation_end'],np.arange(cut,cut+raw['count']))
    return result


def prepare_a7(source_a6data, output):
    """Preserve every entity/cut/horizon query, replacing only time coordinates.

    Fit the one scalar timing scale on the provided TRAINING bank. For strict
    held-out work supply the335-entity training A6 namespace, never full417 data.
    """
    source=Path(source_a6data)
    source_file=source/'data.pt' if source.is_dir() else source
    source_hash=sha256(source_file)
    data=torch.load(source_file,map_location='cpu',weights_only=False)
    token_state=data['token_state']
    records=data['records']
    raw_windows=[]
    square_sum=0.
    active_count=0
    query_rows=[]
    for item in data['windows']:
        raw=encode_record_window(records[item['entity']],item['cut'],item['delta'],token_state)
        if raw['count']!=item['count'] or raw['anchor']!=item['anchor']:
            raise ValueError('Native A7 window changed the original A6 query selection')
        selected=raw['clr'][raw['active']]
        square_sum+=float(selected.square().sum())
        active_count+=len(selected)
        query_rows.append((int(item['entity']),int(item['cut']),float(item['anchor']),float(item['delta']),int(item['count'])))
        raw_windows.append(raw)
    if not raw_windows or not active_count:
        raise ValueError('Training bank needs positive-duration spacing coordinates')
    rms=float(np.sqrt(square_sum/active_count))
    time_scale=rms if rms>np.finfo(np.float64).eps else 1.
    windows=[]
    for item,raw in zip(data['windows'],raw_windows):
        metadata={key:item[key] for key in ('entity','cut','kind','query_kind') if key in item}
        window=_complete_window(raw,time_scale,metadata)
        window['past_extension_tokens']=tokenize_sequence(records[item['entity']],token_state,0.,
            token_state['observation_end'],np.arange(item['cut'],item['cut']+item['count']))
        windows.append(window)
    state={'version':'a7_native_spacing_marks_v1','time_scale':time_scale,
        'mark_token_state':token_state,'token_dim':token_state['token_dim'],
        'mark_dim':token_state['token_dim']-1,'time_scale_fit':'RMS of active CLR over supplied training bank only',
        'active_CLR_coordinates':active_count,'unscaled_active_CLR_RMS':rms,
        'zero_timing_variance_fallback':'scale1 only if all active CLR are machine-zero',
        'timing_coordinate':'native CLR/scalar_scale; inactive and zero-duration coordinates zero',
        'mark_order':'native stable order, no sorting'}
    manifest={'version':'a7_prepared_v1','source_a6_data_sha256':source_hash,
        'source_a6_manifest_digest':digest(data['manifest']),'queries_digest':digest(query_rows),
        'window_bank_size':len(windows),'entities':len(records),'token_dim':state['token_dim'],
        'time_scale':time_scale,'active_CLR_coordinates':active_count,'profile_codes_used':False,
        'query_measure':data['manifest'].get('query_measure'),
        'source_query_counts':data['manifest'].get('query_counts'),
        'source_query_proportions':data['manifest'].get('query_proportions'),
        'scope':data['manifest'].get('scope','provided training namespace'),
        'train_only_calibration_required':True}
    result={'version':'a7_prepared_v1','records':records,'attrs':data['attrs'],
        'windows':windows,'state':state,'manifest':manifest}
    if sha256(source_file)!=source_hash:
        raise RuntimeError('Protected A6 data changed during A7 preparation')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    target=output/'data.pt'
    if target.exists():
        previous=torch.load(target,map_location='cpu',weights_only=False)
        if previous['manifest']!=manifest:
            raise ValueError('A7 preparation identity changed; preserve it and use a new namespace')
        return previous
    temporary=output/'data.pt.tmp'
    torch.save(result,temporary)
    temporary.replace(target)
    write_json(output/'manifest.json',{**manifest,'data_sha256':sha256(target)})
    return result


def collate_a7(windows, device='cpu'):
    if not windows:
        raise ValueError('Cannot collate an empty A7 batch')
    width=max(item['count']+1 for item in windows)
    dim=windows[0]['tokens'].shape[1]
    tokens=torch.zeros(len(windows),width,dim,dtype=torch.float32)
    valid=torch.zeros(len(windows),width,dtype=torch.bool)
    events=valid.clone()
    active=valid.clone()
    for row,item in enumerate(windows):
        length=item['count']+1
        if item['tokens'].shape!=(length,dim) or item['active'].shape!=(length,):
            raise ValueError('Incompatible A7 window dimensions')
        tokens[row,:length]=item['tokens']
        valid[row,:length]=True
        events[row,:length]=item['event_valid']
        active[row,:length]=item['active']
    result={'tokens':tokens,'valid':valid,'event_valid':events,'active':active,
        'attrs':torch.stack([item['attrs'] for item in windows]),
        'count':torch.tensor([item['count'] for item in windows],dtype=torch.long),
        'anchor':torch.tensor([item['anchor'] for item in windows],dtype=torch.float64),
        'delta':torch.tensor([item['delta'] for item in windows],dtype=torch.float64)}
    past_width=max(len(item['past_tokens']) for item in windows)
    past_dim=windows[0]['past_tokens'].shape[1]
    past_tokens=torch.zeros(len(windows),past_width,past_dim,dtype=torch.float32)
    past_valid=torch.zeros(len(windows),past_width,dtype=torch.bool)
    for row,item in enumerate(windows):
        count=len(item['past_tokens'])
        if item['past_tokens'].ndim!=2 or item['past_tokens'].shape[1]!=past_dim:
            raise ValueError('Incompatible observed-history token schema')
        past_tokens[row,:count]=item['past_tokens']
        past_valid[row,:count]=True
    result.update(past_tokens=past_tokens,past_valid=past_valid)
    if all('teacher_code' in item for item in windows):
        result['teacher_codes']=torch.stack([item['teacher_code'] for item in windows])
    return {key:value.to(device) for key,value in result.items()}


def inverse_marks(mark_tokens, prep, old_token_state):
    """Decode only mark columns, retaining input order exactly; no time channel."""
    marks=mark_tokens.detach().cpu().numpy() if torch.is_tensor(mark_tokens) else np.asarray(mark_tokens)
    if marks.ndim!=2 or marks.shape[1]!=old_token_state['token_dim']-1 or not np.isfinite(marks).all():
        raise ValueError('Finite native-order mark matrix required')
    raw=marks.astype(np.float64)*old_token_state['std'][1:]+old_token_state['mean'][1:]
    lo,hi=old_token_state['numeric_slice']
    q=raw[:,lo-1:hi-1]
    numeric=prep.inverse_numeric(q.copy())
    lo,hi=old_token_state['mask_slice']
    mask=raw[:,lo-1:hi-1]>=.5
    categories=[raw[:,start-1:stop-1].argmax(1) for start,stop in old_token_state['category_slices']]
    cat=np.column_stack(categories) if categories else np.empty((len(raw),0),dtype=np.int64)
    return {'x_num':numeric,'x_num_quantile':q,'x_cat':cat,'mask':mask.astype(np.float32)}


def build_suffix_window(cleanwindow, k):
    """A genuine remaining-window query; CLR must be recomputed on its new face."""
    count=int(cleanwindow['count'])
    if not isinstance(k,(int,np.integer)) or not 0<=k<=count:
        raise ValueError('Observed suffix prefix length must be in0..N')
    end=float(cleanwindow['anchor'])+float(cleanwindow['delta'])
    anchor=float(cleanwindow['times'][k-1]) if k else float(cleanwindow['anchor'])
    times=cleanwindow['times'][k:].detach().cpu().clone()
    spacing=encode_spacing(times,anchor,end-anchor)
    raw={'count':count-k,'anchor':anchor,'delta':end-anchor,'times':times,
        'marks':cleanwindow['marks'][k:].clone(),'attrs':cleanwindow['attrs'],
        'past_tokens':cleanwindow['past_tokens'],**spacing}
    metadata={key:cleanwindow[key] for key in ('entity','kind','query_kind') if key in cleanwindow}
    metadata['cut']=int(cleanwindow.get('cut',0))+int(k)
    metadata['suffix_observed_events']=int(k)
    result=_complete_window(raw,cleanwindow['time_scale'],metadata)
    if 'past_extension_tokens' in cleanwindow:
        extension=cleanwindow['past_extension_tokens']
        result['past_tokens']=torch.cat([cleanwindow['past_tokens'],extension[:k]],0)
        result['past_extension_tokens']=extension[k:].clone()
    elif k:
        raise ValueError('Absolute observed-prefix extension tokens needed for a nonempty suffix query')
    return result


def prepare_a7_heldout(source_a6_namespace, output):
    """Adapt held-out queries with the one timing scale fitted on train335 only."""
    source=Path(source_a6_namespace)
    output=Path(output)
    protected=[source/'train_a6/data.pt',source/'heldout_queries.pt',source/'split_manifest.json']
    before={str(path):sha256(path) for path in protected}
    training=prepare_a7(source/'train_a6/data.pt',output/'train_a7')
    test=torch.load(source/'heldout_queries.pt',map_location='cpu',weights_only=False)
    queries=[]
    for query in test['queries']:
        record=test['records'][query['entity']]
        metadata={key:query[key] for key in ('entity','entity_id','prefix_fraction','horizon_fraction','split','target_use')
                  if key in query}
        encoded=encode_a7_window(record,query['cut'],query['delta'],training['state'],metadata)
        if encoded['count']!=query['count'] or encoded['anchor']!=query['anchor']:
            raise ValueError('A7 held-out adaptation changed the supplied evaluation query')
        for key in ('target_calendar_times','target_raw_numeric'):
            if key in query:
                encoded[key]=query[key].clone()
        queries.append(encoded)
    state_hash=calibration_digest(training['state'])
    bundle={'version':'a7_heldout_queries_v1','records':test['records'],
        'entity_ids':test['entity_ids'],'attrs':test['attrs'],'queries':queries,
        'state':training['state'],'training_state_digest':state_hash,
        'preprocessor_hash':test['preprocessor_hash'],'fit_split':'train only',
        'evaluation_unit':'entity, not independent overlapping query windows'}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'heldout_queries.pt'
    manifest={'version':'a7_entity_heldout_v1','train_entities':len(training['records']),
        'heldout_entities':len(test['entity_ids']),'heldout_queries':len(queries),
        'source_a6_artifact_sha256':before,'training_state_digest':state_hash,
        'time_scale':training['state']['time_scale'],'time_scale_fit_split':'train bank only',
        'source_preprocessor_hash':test['preprocessor_hash'],
        'old_A6_profile_used':False,'queries_preserve_entity_cut_anchor_horizon_count':True}
    manifest_path=output/'split_manifest.json'
    if manifest_path.exists():
        from .experiment import read_json
        previous=read_json(manifest_path)
        for key,value in manifest.items():
            if previous[key]!=value:
                raise ValueError('A7 held-out adaptation identity changed')
        if sha256(path)!=previous['heldout_queries_sha256']:
            raise ValueError('A7 held-out query bundle changed')
        return previous
    torch.save(bundle,path)
    if calibration_digest(training['state'])!=state_hash or any(sha256(Path(p))!=h for p,h in before.items()):
        raise RuntimeError('Training calibration or a protected A6 source changed')
    manifest['heldout_queries_sha256']=sha256(path)
    write_json(manifest_path,manifest)
    return manifest


collate=collate_a7
adapt_window=encode_a7_window
