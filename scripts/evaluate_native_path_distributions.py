"""Read-only full-population native path distributions and customer patterns.

This is not the original paper scorer. Overlapping forecast windows are rejected.
Source numeric columns are retained; expm1 is applied to each observed log value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from pgjepa.experiment import write_json
from pgjepa.paper_data import SCHEMAS, EXPECTED, sha256


def identifier(value):
    if pd.isna(value):
        raise ValueError('Missing parent ID')
    if isinstance(value,(int,float,np.number)):
        if not np.isfinite(value):
            raise ValueError('Nonfinite parent ID')
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


def describe(values):
    values = np.asarray(values,dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite numeric value; no silent deletion')
    observed = values[np.isfinite(values)]
    n,total = len(observed),len(values)
    result = {'total':total,'observed':n,'missing_or_undefined':total-n,
        'missing_or_undefined_fraction':float((total-n)/total) if total else None,
        'mean':float(observed.mean()) if n else None,
        'std_ddof0':float(observed.std(ddof=0)) if n else None,
        'std_ddof1':float(observed.std(ddof=1)) if n>1 else None,
        'p95':float(np.quantile(observed,.95)) if n else None,
        'p99':float(np.quantile(observed,.99)) if n else None,
        'p999':float(np.quantile(observed,.999)) if n else None,
        'min':float(observed.min()) if n else None,'max':float(observed.max()) if n else None,
        'zero_fraction_observed':float((observed==0).mean()) if n else None,
        'zero_fraction_all':float((observed==0).sum()/total) if total else None,
        'negative_fraction_observed':float((observed<0).mean()) if n else None}
    if any(isinstance(value,float) and not np.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite computed statistic; no silent deletion or clipping')
    return result


def ks_distance(real,sample):
    a,b = np.asarray(real,dtype=float),np.asarray(sample,dtype=float)
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError('Infinite comparison value')
    aa,bb = a[np.isfinite(a)],b[np.isfinite(b)]
    distance = float(ks_2samp(aa,bb).statistic) if len(aa) and len(bb) else None
    return {'KS_distance':distance,'KS_similarity':1-distance if distance is not None else None,
            'real_defined':len(aa),'sample_defined':len(bb),
            'real_undefined':len(a)-len(aa),'sample_undefined':len(b)-len(bb)}


def correlation(x,y):
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    valid = np.isfinite(x)&np.isfinite(y)
    x,y = x[valid],y[valid]
    value = None
    if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0:
        value = float(spearmanr(x,y).statistic)
    return {'spearman':value,'defined_pairs':len(x),'undefined_pairs':int((~valid).sum()),
            'constant_or_insufficient':value is None}


def category_probabilities(values):
    counts: dict[str,int] = {}
    for value in values:
        if pd.isna(value):
            key = 'missing:'
        else:
            if isinstance(value,(int,float,np.number)):
                if not np.isfinite(value):
                    raise ValueError('Infinite categorical value')
                value = format(float(value),'.17g')
            key = 'category:'+str(value)
        counts[key] = counts.get(key,0)+1
    return {key:count/len(values) for key,count in sorted(counts.items())} if len(values) else {}


def validate_frame(frame,parents,dataset):
    frame = frame.copy()
    key = 'account_id' if 'account_id' in frame else 'entity' if 'entity' in frame else None
    required = {'t_offset',*SCHEMAS[dataset]['numeric'],*SCHEMAS[dataset]['categorical']}
    if key is None or not required.issubset(frame.columns):
        raise ValueError('Expected native parent/time/mark schema')
    frame['_parent'] = frame[key].map(identifier)
    if set(frame['_parent'])-set(parents):
        raise ValueError('Generated rows include unknown parent IDs')
    # A unique BOS query index is allowed; repeated windows per parent are not.
    for query_key in ('query_index','query_id'):
        if query_key in frame and (frame.groupby('_parent')[query_key].nunique()>1).any():
            raise ValueError('Overlapping forecast windows are not a full-native population')
    for name in ['t_offset',*SCHEMAS[dataset]['numeric']]:
        frame[name] = pd.to_numeric(frame[name],errors='raise')
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError('Infinite numeric/time value in '+name)
    if not np.isfinite(frame.t_offset).all():
        raise ValueError('Missing or nonfinite event time')
    ordering = ['_parent','t_offset']+(['trans_id'] if 'trans_id' in frame else [])
    return frame.sort_values(ordering,kind='stable').reset_index(drop=True)


def analyze(frame,parents,dataset,*,retail_cumulative_error=False):
    """All-parent patterns include zero-event parents; marks retain row alignment."""
    parents = [identifier(value) for value in parents]
    if len(set(parents))!=len(parents):
        raise ValueError('Duplicate population parent IDs')
    frame = validate_frame(frame,parents,dataset)
    schema = SCHEMAS[dataset]
    groups = {key:group for key,group in frame.groupby('_parent',sort=False)}
    arrays: dict[str,Any] = {}
    patterns: dict[str,Any] = {}
    columns: dict[str,Any] = {}
    categories: dict[str,Any] = {}
    count_rows,span_rows,first,last,gap_rows = [],[],[],[],[]
    for entity in parents:
        group = groups.get(entity)
        times = group.t_offset.to_numpy(dtype=float) if group is not None else np.empty(0)
        count_rows.append(len(times))
        span_rows.append(float(times[-1]-times[0]) if len(times) else 0.)
        first.append(times[0] if len(times) else np.nan)
        last.append(times[-1] if len(times) else np.nan)
        gap_rows.extend(np.diff(times).round(9).tolist())
    counts,spans,gaps = np.asarray(count_rows),np.asarray(span_rows),np.asarray(gap_rows)
    arrays.update({'parent.N_all':counts,'parent.span_all_empty0':spans,'time.internal_gap_round9':gaps,
        'time.internal_positive_gap_round9':gaps[gaps>0],'time.span_active_round9':spans[counts>0].round(9),
        'time.first_active':np.asarray(first),'time.last_active':np.asarray(last)})
    temporal = {key:describe(value) for key,value in arrays.items()}
    temporal['active_N_span_original_FF_inputs'] = np.column_stack([counts[counts>0],spans[counts>0]]).tolist()
    temporal['empty_parents'] = int((counts==0).sum())
    temporal['nonempty_parents'] = int((counts>0).sum())
    temporal['gap_zero_atom'] = float((gaps==0).mean()) if len(gaps) else None
    temporal['negative_event_time_fraction'] = float((frame.t_offset<0).mean()) if len(frame) else None
    patterns['N'] = counts
    patterns['span_empty0'] = spans
    patterns['N_per_active_span'] = np.divide(counts,spans,out=np.full(len(parents),np.nan),where=spans>0)
    gap_cv,positive_gap_cv = [],[]
    for entity in parents:
        group = groups.get(entity)
        delta = np.diff(group.t_offset.to_numpy(dtype=float)).round(9) if group is not None else np.empty(0)
        positive = delta[delta>0]
        gap_cv.append(float(delta.std(ddof=0)/delta.mean()) if len(delta) and delta.mean()>0 else np.nan)
        positive_gap_cv.append(float(positive.std(ddof=0)/positive.mean()) if len(positive) else np.nan)
    patterns['internal_gap_CV_ddof0'] = np.asarray(gap_cv)
    patterns['internal_positive_gap_CV_ddof0'] = np.asarray(positive_gap_cv)
    for name in schema['numeric']:
        raw = frame[name].to_numpy(dtype=float)
        modes = {'source':raw}
        if name.startswith('log_'):
            with np.errstate(over='raise',invalid='raise'):
                try:
                    modes['expm1_per_row'] = np.expm1(raw)
                except FloatingPointError as error:
                    raise ValueError('Log back-transform overflow in '+name) from error
        columns[name] = {}
        for mode,values in modes.items():
            prefix = name+'.'+mode
            arrays['column.'+prefix] = values
            columns[name][mode] = describe(values)
            # Index-aligned whole-row sorting above avoids pairing unrelated marks/times.
            temporary = frame[['_parent','t_offset']].copy()
            temporary['_value'] = values
            by_parent = {entity:g for entity,g in temporary.groupby('_parent',sort=False)}
            rows: dict[str,list[Any]] = {key:[] for key in ('mean','std_ddof0','std_ddof1','total_missing_as_unobserved',
                                      'final_observed','time_mark_spearman','lag1_spearman')}
            differences = []
            for entity in parents:
                group = by_parent.get(entity)
                v = group['_value'].to_numpy() if group is not None else np.empty(0)
                t = group.t_offset.to_numpy() if group is not None else np.empty(0)
                observed = v[np.isfinite(v)]
                rows['mean'].append(observed.mean() if len(observed) else np.nan)
                rows['std_ddof0'].append(observed.std(ddof=0) if len(observed) else np.nan)
                rows['std_ddof1'].append(observed.std(ddof=1) if len(observed)>1 else np.nan)
                # N0 total is0; a nonempty path with all marks missing has unknown total.
                rows['total_missing_as_unobserved'].append(observed.sum() if len(observed) else 0. if not len(v) else np.nan)
                rows['final_observed'].append(v[-1] if len(v) and np.isfinite(v[-1]) else np.nan)
                rows['time_mark_spearman'].append(correlation(t,v)['spearman'] if len(v) else None)
                rows['lag1_spearman'].append(correlation(v[:-1],v[1:])['spearman'] if len(v)>1 else None)
                if len(v)>1:
                    paired = np.isfinite(v[:-1])&np.isfinite(v[1:])
                    differences.extend(np.diff(v)[paired].tolist())
            for metric,values_by_entity in rows.items():
                patterns[prefix+'.'+metric] = np.asarray(values_by_entity,dtype=float)
            arrays['first_difference.'+prefix] = np.asarray(differences)
            columns[name][mode]['adjacent_first_difference'] = describe(differences)
    for name in schema['categorical']:
        probabilities = category_probabilities(frame[name].to_numpy())
        categories[name] = {'probabilities_including_missing':probabilities,
            'missing_fraction':float(frame[name].isna().mean()) if len(frame) else None,
            'observed_categories':sum(key!='missing:' for key in probabilities)}
    arrays.update({'customer.'+key:value for key,value in patterns.items()})
    primary = 'log_amount' if 'log_amount' in schema['numeric'] else schema['numeric'][0]
    mode = 'expm1_per_row' if primary.startswith('log_') else 'source'
    average = patterns[primary+'.'+mode+'.mean']
    total = patterns[primary+'.'+mode+'.total_missing_as_unobserved']
    coupling = {'N_vs_mean_primary_mark':correlation(counts,average),'N_vs_span':correlation(counts,spans),
                'primary_mark_total_vs_span':correlation(total,spans),'primary_mark':primary,'mark_mode':mode}
    second = schema['numeric'][1] if len(schema['numeric'])>1 else None
    descriptive = {'ordered_second_numeric_column':second,
        'interpretation':'Descriptive changes only; decreases may be legitimate (including Berka balance).'}
    if second:
        diff = arrays['first_difference.'+second+'.source']
        descriptive['decrease_fraction_observed_adjacent_pairs'] = float((diff<0).mean()) if len(diff) else None
        descriptive['observed_adjacent_pairs'] = len(diff)
    if retail_cumulative_error:
        if dataset!='online_retail':
            raise ValueError('Retail cumulative reconstruction diagnostic is only defined for online_retail')
        errors,tolerances = [],[]
        for group in groups.values():
            amount,cumulative = np.expm1(group.log_amount.to_numpy()),np.expm1(group.log_cumamt.to_numpy())
            if len(group)>1:
                delta = np.diff(cumulative)
                valid = np.isfinite(delta)&np.isfinite(amount[1:])
                errors.extend((delta[valid]-amount[1:][valid]).tolist())
                tolerances.extend((1e-6+1e-9*np.maximum(np.abs(delta[valid]),np.abs(amount[1:][valid]))).tolist())
        errors = np.asarray(errors)
        descriptive['retail_delta_cumulative_minus_current_amount'] = {'error':describe(errors),
            'mean_absolute_error':float(np.abs(errors).mean()) if len(errors) else None,
            'within_floating_tolerance_fraction':float((np.abs(errors)<=tolerances).mean()) if len(errors) else None,
            'tolerance':'abs(error) <= 1e-6 + 1e-9*max(abs(delta cumulative), abs(current amount)); evaluation only',
            'source_real_baseline_required':True}
    summary = {'population_parents':len(parents),'events':len(frame),'columns':columns,'categories':categories,
        'temporal':temporal,'customer_patterns':{key:describe(value) for key,value in patterns.items()},
        'customer_coupling':coupling,'ordered_change_diagnostics':descriptive}
    table = pd.DataFrame({'parent':parents,**patterns})
    return summary,arrays,table


def compare(real_arrays,sample_arrays,real_summary,sample_summary):
    result = {'distributions':{key:ks_distance(value,sample_arrays[key]) for key,value in real_arrays.items()},
              'categorical_total_variation':{}}
    for name,item in real_summary['categories'].items():
        a,b = item['probabilities_including_missing'],sample_summary['categories'][name]['probabilities_including_missing']
        result['categorical_total_variation'][name] = {
            'TV_including_missing':float(.5*sum(abs(a.get(key,0.)-b.get(key,0.)) for key in set(a)|set(b))) if a and b else None,
            'real_missing_fraction':real_summary['categories'][name]['missing_fraction'],
            'sample_missing_fraction':sample_summary['categories'][name]['missing_fraction']}
    return result


def validate_counts(path,parents,table):
    path = Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as saved:
            ids,counts = saved['parent_ids'],saved['counts']
        frame = pd.DataFrame({'entity':ids,'count':counts})
    else:
        frame = pd.read_csv(path)
    key = 'entity' if 'entity' in frame else 'account_id' if 'account_id' in frame else 'parent'
    if key not in frame or 'count' not in frame:
        raise ValueError('Counts require explicit parent IDs and count column')
    ids = frame[key].map(identifier)
    values = pd.to_numeric(frame['count'],errors='raise').to_numpy()
    if (ids.duplicated().any() or set(ids)!=set(parents) or not np.isfinite(values).all() or
            (values<0).any() or (values!=np.floor(values)).any()):
        raise ValueError('Counts must cover every source parent exactly once with nonnegative integer N')
    declared = dict(zip(ids,values))
    if any(declared[row.parent]!=row.N for row in table.itertuples()):
        raise ValueError('Declared sampled counts disagree with actual CSV rows')


def evaluate(dataset,raw_root,samples,output,*,parents_path=None,counts=None,count_sources=None,retail_cumulative_error=False):
    raw_root,output = Path(raw_root),Path(output)
    if any((output/name).exists() for name in ('distributions.json','report.md','real_customer_patterns.csv')):
        raise ValueError('Preserve existing evaluation output; choose a new namespace')
    raw = raw_root/dataset/'raw'
    source_parents,source_child = raw/'account.csv',raw/'trans.csv'
    parents_path = Path(parents_path) if parents_path else source_parents
    paths = {source_parents,source_child,parents_path,Path(__file__),*[Path(p) for p in samples.values()],
             *[Path(p) for p in (counts or {}).values()]}
    before = {str(path):sha256(path) for path in paths}
    parent_frame,real = pd.read_csv(source_parents),pd.read_csv(source_child)
    parents = parent_frame.account_id.map(identifier).tolist()
    if len(set(parents))!=len(parents) or (len(parents),len(real))!=EXPECTED[dataset]:
        raise ValueError('Original source paper population counts or parent identities differ')
    supplied = pd.read_csv(parents_path)
    supplied_key = 'account_id' if 'account_id' in supplied else 'entity'
    supplied_ids = supplied[supplied_key].map(identifier)
    if supplied_ids.duplicated().any() or set(supplied_ids)!=set(parents):
        raise ValueError('Supplied parents must match the full original population')
    real_summary,real_arrays,real_table = analyze(real,parents,dataset,retail_cumulative_error=retail_cumulative_error)
    self_comparison = compare(real_arrays,real_arrays,real_summary,real_summary)
    if any(value['KS_distance'] not in (0.,None) for value in self_comparison['distributions'].values()):
        raise RuntimeError('Real-self distribution baseline failed')
    result = {'dataset':dataset,'scope':'full-native same-parent population distributions; not paper five scores',
        'numeric_policy':'Original raw CSV source values; source log columns and expm1 of EACH row; no QT roundtrip.',
        'timing_policy':'Stable whole-row parent/time sort; internal differences round9 as original scorer; active ND retains unrounded spans.',
        'zero_count_policy':'All raw account IDs included in N and span(empty=0); undefined customer means explicitly counted.',
        'count_source_policy':'Diagnostics do not verify whether N was free; given-N/graph baselines cannot establish free-count quality.',
        'ordered_change_policy':'Decreases are descriptive; Berka balance decreases are legitimate, not labeled violations.',
        'input_scope_assumption':'Caller declares complete native paths. Multiple query IDs per parent are rejected; this does not certify the generation protocol.',
        'real':real_summary,'real_self_comparison':self_comparison,'samples':{}}
    output.mkdir(parents=True,exist_ok=True)
    real_table.to_csv(output/'real_customer_patterns.csv',index=False)
    for index,(label,path) in enumerate(samples.items()):
        frame = pd.read_csv(path)
        summary,arrays,table = analyze(frame,parents,dataset,retail_cumulative_error=retail_cumulative_error)
        if label in (counts or {}):
            validate_counts(counts[label],parents,table)
        comparison = compare(real_arrays,arrays,real_summary,summary)
        result['samples'][label] = {'count_source':(count_sources or {}).get(label,'unknown; count-free claim not established'),
            'summary':summary,'comparison':comparison,'source_csv_sha256':before[str(Path(path))]}
        table.to_csv(output/f'sample_{index}_customer_patterns.csv',index=False)
    if any(sha256(Path(path))!=expected for path,expected in before.items()):
        raise RuntimeError('Protected source changed during evaluation')
    result['protected_sha256'] = before
    result['protected_sources_unchanged'] = True
    write_json(output/'distributions.json',result)
    lines = ['# Full-native path distribution audit','',result['scope']+'.','',
        'Each customer-pattern CSV retains all original parent IDs, including generated N=0. '
        'Missing or undefined pattern values are reported explicitly. All KS values below are distances; lower is better.','',
        '| Sample | N KS (all parents) | Gap KS | Active span KS | Count source |',
        '|---|---:|---:|---:|---|']
    for label,item in result['samples'].items():
        dist = item['comparison']['distributions']
        values = [dist[key]['KS_distance'] for key in ('parent.N_all','time.internal_gap_round9','time.span_active_round9')]
        formatted = ['undefined' if value is None else f'{value:.5f}' for value in values]
        lines.append('| '+label+' | '+' | '.join(formatted)+' | '+item['count_source']+' |')
    lines += ['','Detailed global columns, both standard-deviation conventions, customer mark mean/std/total/final distributions, '
        'within-customer time/mark and lag-one correlations, and between-customer coupling are in distributions.json. '
        'Signed first differences and ordered decreases are descriptive. Original LG and Joint FF are evaluated separately.','',
        'The real-self baseline is exact for all defined distributions. Optional Retail cumulative reconstruction error '
        'is reported for the physical source and samples with the same explicit floating tolerance; no values are corrected.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def labelled(arguments):
    result = {}
    for argument in arguments or []:
        if '=' not in argument:
            raise ValueError('Expected LABEL=VALUE')
        name,value = argument.split('=',1)
        if not name or name in result:
            raise ValueError('Unique nonempty sample labels required')
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=sorted(SCHEMAS),required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--parents',type=Path)
    parser.add_argument('--sample',action='append',required=True,help='LABEL=full-native-child.csv')
    parser.add_argument('--counts',action='append',help='LABEL=counts.csv or counts.npz; validates all N including zeros')
    parser.add_argument('--count-source',action='append',help='LABEL=description, e.g. given-N plus graph or freely sampled')
    parser.add_argument('--retail-cumulative-error',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    samples,counts,sources = labelled(args.sample),labelled(args.counts),labelled(args.count_source)
    if (set(counts)|set(sources))-set(samples):
        raise ValueError('Count files/source annotations require a declared sample label')
    try:
        evaluate(args.dataset,args.raw_root,samples,args.output,parents_path=args.parents,counts=counts,
                 count_sources=sources,retail_cumulative_error=args.retail_cumulative_error)
    except (ValueError,RuntimeError,FloatingPointError) as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'invalid.json',{'status':'INVALID','reason':str(error),'no_source_values_modified':True})
        raise
    print(args.output/'report.md')


if __name__=='__main__':
    main()
