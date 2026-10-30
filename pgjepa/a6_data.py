"""Schema-generic windows and fixed global profiles for joint-path A6.

All numeric mark coordinates reuse the frozen training quantile transform.
Profiles describe a regime; they are not lossless representations of paths.
No real count, timeline, or target profile is needed by inverse_decode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
import torch

from .experiment import digest, write_json
from .paper_data import load_prepared_paper, sha256


PROFILE_QUANTILES = (.1, .5, .9, .99, .999)


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _times(record):
    if 'calendar_times' in record:
        return _array(record['calendar_times']).astype(np.float64)
    return _array(record['delta']).reshape(-1).astype(np.float64).cumsum()


def _unscaled_tokens(record, state, anchor, horizon, indices):
    indices = np.asarray(indices, dtype=np.int64)
    size = len(indices)
    result = np.zeros((size, state['token_dim']), dtype=np.float64)
    times = _times(record)[indices]
    if horizon < 0 or not np.isfinite(horizon) or not np.isfinite(anchor):
        raise ValueError('Finite anchor and nonnegative horizon required')
    if size and ((times < anchor-1e-8).any() or (times > anchor+horizon+1e-8).any()):
        raise ValueError('Requested event lies outside the declared time window')
    fraction = np.full(size, .5) if horizon == 0 else (times-anchor)/horizon
    result[:, 0] = ndtri(np.clip(fraction, state['time_epsilon'], 1-state['time_epsilon']))
    lo, hi = state['numeric_slice']
    result[:, lo:hi] = _array(record['x_num'])[indices]
    for column, (start, stop) in enumerate(state['category_slices']):
        encoded = _array(record['x_cat'])[indices, column].astype(np.int64)
        if ((encoded < 0)|(encoded >= stop-start)).any():
            raise ValueError('Category code outside the fitted schema')
        result[np.arange(size), start+encoded] = 1.
    lo, hi = state['mask_slice']
    result[:, lo:hi] = _array(record['mask'])[indices]
    if not np.isfinite(result).all():
        raise ValueError('Token coordinates must be finite')
    return result


def fit_token_state(records, prep, *, time_epsilon=1e-7):
    if not records or not 0 < time_epsilon < .5:
        raise ValueError('Require records and a valid probit endpoint epsilon')
    ends = [float(torch.as_tensor(r['exposure_end'])) for r in records]
    if min(ends) <= 0 or not np.allclose(ends, ends[0], rtol=0, atol=1e-8):
        raise ValueError('One positive administrative end is required')
    offset = 1+prep.n_num
    category_slices = []
    for size in prep.cat_sizes:
        category_slices.append((offset, offset+size))
        offset += size
    mask_slice = (offset, offset+prep.n_num+len(prep.cat_sizes))
    state = {'version':'a6_joint_tokens_v1', 'n_num':prep.n_num, 'cat_sizes':list(prep.cat_sizes),
        'attrs_dim':prep.attrs_dim, 'numeric_slice':(1, 1+prep.n_num),
        'category_slices':category_slices, 'mask_slice':mask_slice,
        'token_dim':mask_slice[1], 'continuous_indices':list(range(1+prep.n_num)),
        'observation_end':ends[0], 'time_epsilon':float(time_epsilon),
        'time_coordinate':'probit(relative calendar fraction); epsilon only at numerical endpoints',
        'generation_order':'jointly sort time and every corresponding mark/mask row'}
    values = np.concatenate([_unscaled_tokens(r,state,0.,ends[0],np.arange(len(r['delta']))) for r in records])
    mean, std = values.mean(0), values.std(0)
    state['constant_token_channels'] = np.flatnonzero(std < 1e-7).tolist()
    state['mean'] = mean
    state['std'] = np.where(std < 1e-7, 1., std)
    return state


def tokenize_sequence(record, token_state, anchor=0., horizon=None, indices=None):
    horizon = token_state['observation_end'] if horizon is None else float(horizon)
    indices = np.arange(len(record['delta'])) if indices is None else indices
    raw = _unscaled_tokens(record,token_state,float(anchor),horizon,indices)
    return torch.as_tensor((raw-token_state['mean'])/token_state['std'], dtype=torch.float32)


def encode_window(record, cut, horizon, token_state, profile_state=None):
    size = len(record['delta'])
    if not isinstance(cut, (int,np.integer)) or not 0 <= cut <= size:
        raise ValueError('Cut must be an observed prefix length')
    times = _times(record)
    anchor = float(times[cut-1]) if cut else 0.
    end = token_state['observation_end']
    horizon = float(horizon)
    if not np.isfinite(horizon) or not 0 <= horizon <= end-anchor+1e-8:
        raise ValueError('Horizon extends beyond observable calendar')
    finish = max(cut,int(np.searchsorted(times,anchor+horizon,side='right')))
    future = tokenize_sequence(record,token_state,anchor,horizon,np.arange(cut,finish))
    result = {'cut':int(cut),'anchor':anchor,'delta':horizon,'count':finish-cut,
        'attrs':record['attrs'].detach().cpu().clone(),
        'past_tokens':tokenize_sequence(record,token_state,0.,end,np.arange(cut)),
        'future_tokens':future}
    if profile_state is not None:
        result['code'] = profile_features(future,token_state,profile_state)
    return result


def _raw_profile(tokens, token_state, feature_state):
    x = _array(tokens).astype(np.float64)
    dim = len(token_state['continuous_indices'])
    if not len(x):
        return np.zeros(7*dim), np.zeros(feature_state['residual_dim']), 1.
    continuous = x[:,token_state['continuous_indices']]
    stats = np.stack([continuous.mean(0),continuous.std(0),
                      *[np.quantile(continuous,q,axis=0) for q in PROFILE_QUANTILES]],axis=1).reshape(-1)
    rff = np.sqrt(2/feature_state['rff_dim'])*np.cos(x@feature_state['rff_weight']+feature_state['rff_phase'])
    covariance = (continuous-continuous.mean(0)).T@(continuous-continuous.mean(0))/len(x)
    covariance = covariance[np.triu_indices(dim,k=1)]
    typed_freq = x[:,1+token_state['n_num']:].mean(0)
    residual = np.r_[rff.mean(0),covariance,typed_freq]
    return stats,residual,0.


def fit_profile_state(window_tokens, token_state, *, seed=9, rff_dim=32, pca_dim=14):
    if not window_tokens or pca_dim < 1 or rff_dim < 1:
        raise ValueError('Positive feature sizes and a nonempty training window bank required')
    rng = np.random.default_rng(seed)
    d = token_state['token_dim']
    c = len(token_state['continuous_indices'])
    residual_dim = rff_dim+c*(c-1)//2+d-1-token_state['n_num']
    state = {'version':'a6_fixed_profile_v1','seed':int(seed),'rff_dim':rff_dim,'pca_dim':pca_dim,
        'residual_dim':residual_dim,'rff_weight':rng.standard_normal((d,rff_dim)),
        'rff_phase':rng.uniform(0,2*np.pi,rff_dim),'quantiles':list(PROFILE_QUANTILES)}
    raw = [_raw_profile(x,token_state,state) for x in window_tokens]
    residual = np.stack([entry[1] for entry in raw])
    state['residual_mean'] = residual.mean(0)
    centered = residual-state['residual_mean']
    covariance = centered.T@centered/max(1,len(centered)-1)
    _, basis = np.linalg.eigh(covariance)
    basis = basis[:,::-1][:,:min(pca_dim,residual_dim)]
    for column in range(basis.shape[1]):
        if basis[np.abs(basis[:,column]).argmax(),column] < 0:
            basis[:,column] *= -1
    state['pca_basis'] = np.pad(basis,((0,0),(0,pca_dim-basis.shape[1])))
    values = np.stack([np.r_[entry[0],(entry[1]-state['residual_mean'])@state['pca_basis'],entry[2]] for entry in raw])
    state['code_mean'] = values.mean(0)
    std = values.std(0)
    state['constant_code_channels'] = np.flatnonzero(std < 1e-5).tolist()
    state['code_std'] = np.where(std < 1e-5,1.,std)
    state['code_dim'] = values.shape[1]
    state['definition'] = 'generic continuous mean/std/q10/q50/q90/q99/q999 + PCA of joint RFF/covariance/type frequencies + empty; regime descriptor, not lossless'
    return state


def profile_features(tokens, token_state, profile_state):
    stats,residual,empty = _raw_profile(tokens,token_state,profile_state)
    value = np.r_[stats,(residual-profile_state['residual_mean'])@profile_state['pca_basis'],empty]
    return torch.as_tensor((value-profile_state['code_mean'])/profile_state['code_std'],dtype=torch.float32)


def inverse_decode(tokens, prep, token_state, anchor=0., horizon=None):
    """Decode a sampled marked point set; sort entire rows by sampled time."""
    horizon = token_state['observation_end'] if horizon is None else float(horizon)
    if horizon < 0 or not np.isfinite(horizon) or not np.isfinite(anchor):
        raise ValueError('Finite window required')
    normalized = _array(tokens).astype(np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != token_state['token_dim'] or not np.isfinite(normalized).all():
        raise ValueError('Finite sampled token matrix with fitted schema required')
    value = normalized*token_state['std']+token_state['mean']
    times = float(anchor)+horizon*ndtr(value[:,0])
    order = np.argsort(times,kind='stable')
    value, times = value[order],times[order]
    lo,hi = token_state['numeric_slice']
    q = value[:,lo:hi]
    raw = prep.inverse_numeric(q.copy())
    lo,hi = token_state['mask_slice']
    mask = value[:,lo:hi] >= .5
    categories = [value[:,start:stop].argmax(1) for start,stop in token_state['category_slices']]
    cat = np.column_stack(categories) if categories else np.empty((len(value),0),dtype=np.int64)
    return {'t_offset':times,'x_num':raw,'x_num_quantile':q,'x_cat':cat,
            'mask':mask.astype(np.float32),'order':order}


def prepare_a6(preparedpath, output, windows=4096, seed=9):
    if windows < 1:
        raise ValueError('Positive training bank size required')
    records,prep,source = load_prepared_paper(preparedpath)
    token_state = fit_token_state(records,prep)
    rng = np.random.default_rng(seed)
    bos_count = windows//2
    if bos_count < len(records):
        raise ValueError('Window bank needs at least twice as many windows as entities for complete BOS coverage')
    endpoint_count = (windows-bos_count)//2
    random_count = windows-bos_count-endpoint_count
    bos_entities = np.concatenate([rng.permutation(len(records))
        for _ in range((bos_count+len(records)-1)//len(records))])[:bos_count]
    bank = []
    for index in range(windows):
        kind = ('bos_full' if index < bos_count else
                'prefix_endpoint' if index < bos_count+endpoint_count else 'prefix_random')
        entity = int(bos_entities[index]) if kind == 'bos_full' else int(rng.integers(len(records)))
        record = records[entity]
        cut = 0 if kind == 'bos_full' else int(rng.integers(len(record['delta'])+1))
        anchor = float(_times(record)[cut-1]) if cut else 0.
        remaining = token_state['observation_end']-anchor
        delta = float(rng.random()*remaining) if kind == 'prefix_random' else remaining
        item = encode_window(record,cut,delta,token_state)
        item['entity'] = entity
        item['kind'] = kind
        item['query_kind'] = {'bos_full':'BOS_full','prefix_endpoint':'prefix_endpoint',
                              'prefix_random':'prefix_uniform'}[kind]
        bank.append(item)
    profile_state = fit_profile_state([item['future_tokens'] for item in bank],token_state,seed=seed)
    for item in bank:
        item['code'] = profile_features(item['future_tokens'],token_state,profile_state)
    result = {'version':'a6_prepared_v2','records':records,'attrs':torch.stack([r['attrs'] for r in records]),
        'token_state':token_state,'profile_state':profile_state,'windows':bank,
        'manifest':{'source_prepared_manifest_digest':digest(source),'seed':seed,'window_bank_size':windows,
            'query_measure':'exact half BOS-full with shuffled entity cycles; remaining endpoint/random split',
            'query_counts':{'bos_full':bos_count,'prefix_endpoint':endpoint_count,'prefix_random':random_count},
            'query_proportions':{'bos_full':bos_count/windows,'prefix_endpoint':endpoint_count/windows,
                                 'prefix_random':random_count/windows},
            'bos_entities_covered':len(set(int(i) for i in bos_entities)),
            'bos_sampling':'choose entity uniformly then its representative BOS window; do not weight by duplicate-window frequency',
            'scope':'training-population window bank; no held-out generalization claim',
            'old_quantile_state_digest':digest(prep.state_dict())}}
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    target = output/'data.pt'
    if target.exists():
        existing = torch.load(target,map_location='cpu',weights_only=False)
        if existing['manifest'] != result['manifest']:
            raise ValueError('Prepared A6 identity differs; preserve prior data and use a new folder')
        return existing
    temporary = output/'data.pt.tmp'
    torch.save(result,temporary)
    temporary.replace(target)
    write_json(output/'manifest.json',{**result['manifest'],'token_dim':token_state['token_dim'],
        'code_dim':profile_state['code_dim'],'data_sha256':sha256(target)})
    return result
