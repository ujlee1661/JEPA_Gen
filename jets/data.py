"""Window collation, suffix construction, and native mark decoding."""
from __future__ import annotations
import numpy as np
import torch
from .spacing import encode_spacing

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


def collate_windows(windows, device='cpu'):
    if not windows:
        raise ValueError('Cannot collate an empty event batch')
    width=max(item['count']+1 for item in windows)
    dim=windows[0]['tokens'].shape[1]
    tokens=torch.zeros(len(windows),width,dim,dtype=torch.float32)
    valid=torch.zeros(len(windows),width,dtype=torch.bool)
    events=valid.clone()
    active=valid.clone()
    for row,item in enumerate(windows):
        length=item['count']+1
        if item['tokens'].shape!=(length,dim) or item['active'].shape!=(length,):
            raise ValueError('Incompatible event window dimensions')
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
