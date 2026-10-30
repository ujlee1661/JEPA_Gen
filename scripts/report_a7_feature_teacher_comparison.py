"""Reproducible full417 A7 feature-teacher/native baseline comparison.

Read-only scientific inputs. New native paths are arguments, never inferred
from latest checkpoints. Oracle/overlapping forecasts are not five-score rows.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from pgjepa.experiment import read_json,write_json
from pgjepa.paper_data import EXPECTED,sha256
from scripts.evaluate_native_path_distributions import analyze,compare


SCORER_SHA='f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'
PROJECT=Path(__file__).resolve().parents[1]


def baseline_cases(runtime):
    runtime=Path(runtime);result=[]
    for label,folder,arm in [('A7_POINT_ON','retail_a7_full_fidelity_v1','A7_AUX_ON'),
                             ('A7_POINT_OFF','retail_a7_full_fidelity_v1','A7_AUX_OFF'),
                             ('A7_ENERGY','retail_a7_full_distribution_student_v1','A7_AUX_ON')]:
        for seed in (1,2,3):
            result.append({'label':label,'path':str(runtime/folder/'online_retail/generating'/arm/'native'/str(seed)/'child.csv'),
                'generation_seed':seed,'training_seed':9,'kind':'preserved_baseline','count_source':'Generated-N',
                'completion_required':True})
    for seed in (1,2,3):
        result.append({'label':'TabARGN','path':str(PROJECT/'outputs/paper_reproduction/jobs/tabularargn/online_retail'/f'native_s{seed}'/'child.csv'),
            'generation_seed':seed,'training_seed':9,'kind':'preserved_baseline','count_source':'Generated-N',
            'completion_required':False,'author_primary_generation_seed':1})
    result.append({'label':'RDBDiff_GivenN_graph','path':str(PROJECT/'outputs/paper_reproduction/jobs/rdbdiff/online_retail/native_s1/child.csv'),
        'generation_seed':1,'training_seed':1,'kind':'given_N_fixed_graph_contrast',
        'count_source':'Given real N + fixed structure; NOT count-free','completion_required':False})
    return result


def new_cases(arguments):
    result=[]
    for arg in arguments or []:
        if '=' not in arg:raise ValueError('New native inputs require LABEL=child.csv')
        label,path=arg.split('=',1)
        if not label:raise ValueError('Nonempty model label required')
        result.append({'label':label,'path':path,'kind':'new_model','completion_required':True,
                       'count_source':'unverified until native receipt checked'})
    return result


def score_module(path):
    path=Path(path)
    if sha256(path)!=SCORER_SHA:raise ValueError('Original f254 scorer bytes differ')
    spec=importlib.util.spec_from_file_location('feature_teacher_f254',path)
    if spec is None or spec.loader is None:raise ImportError('Cannot load frozen evaluator')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_case(case,raw,protected):
    """Artifact provenance only, not a proof of all runtime information flows."""
    case=dict(case);path=Path(case['path']);folder=path.parent
    if not path.exists():return {**case,'status':'PENDING_SOURCE','reason':'Native CSV not yet available'}
    protected[str(path)]=sha256(path)
    receipt_path=folder/'generation.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path);protected[str(receipt_path)]=sha256(receipt_path)
        if receipt['child_sha256']!=sha256(path):raise ValueError('Native CSV receipt SHA differs: '+str(path))
        if receipt.get('count_or_face_or_teacher_oracle') or receipt.get('population')=='heldout_queries':
            return {**case,'status':'EXCLUDED_PRIVILEGED_OR_FORECAST','reason':'Oracle/overlapping future-window output is not full-native fidelity'}
        case['generation_seed']=receipt.get('generation_seed',case.get('generation_seed'))
        case['training_seed']=receipt.get('training_seed',case.get('training_seed'))
        case['scope']=receipt.get('scope','full-parent native artifact; provenance scope not declared')
        case['count_source']=receipt.get('count_regime',case.get('count_source','Generated-N per nonoracle receipt'))
        if case['completion_required']:
            if len(folder.parents)<4 or folder.parent.name!='native':
                return {**case,'status':'PENDING_COMPLETION_PROOF','reason':'Require standard generating/ARM/native/SEED/final artifact layout'}
            run=folder.parents[3];arm=folder.parents[1].name
            complete_path=run/arm/'complete.json';checkpoint=run/arm/'final.pt'
            if not complete_path.exists() or not checkpoint.exists():
                return {**case,'status':'PENDING_COMPLETION_PROOF','reason':'Fixed final completion/weights absent; latest model never scored'}
            complete=read_json(complete_path)
            if complete.get('checkpoint_sha256')!=sha256(checkpoint) or receipt.get('checkpoint_sha256')!=sha256(checkpoint):
                raise ValueError('Final completion/checkpoint identity differs')
            protected[str(complete_path)]=sha256(complete_path);protected[str(checkpoint)]=sha256(checkpoint)
            manifest=run/'manifest.json'
            if manifest.exists():
                declaration=read_json(manifest);protected[str(manifest)]=sha256(manifest)
                if declaration.get('fit_entities',417)!=417:raise ValueError('Require full417 model; strict335 forecast is separate')
                case['training_seed']=declaration.get('config',{}).get('training_seed',case.get('training_seed'))
                case['expected_generation_seeds']=declaration.get('config',{}).get('generation_seeds')
                declared_updates=declaration.get('config',{}).get('generator_steps')
                if declared_updates is not None and complete.get('updates')!=declared_updates:
                    raise ValueError('Completed updates differ from prospective generator budget')
                case['declared_generator_updates']=declared_updates
        case['receipt_scope']='Declared nonoracle generation receipt + fixed-final checkpoint verified; not full access trace'
    else:
        meta=folder/'author_output.meta.json'
        if not meta.exists():return {**case,'status':'PENDING_NATIVE_PROVENANCE','reason':'No completed native generation receipt'}
        author=read_json(meta);protected[str(meta)]=sha256(meta)
        if author['output']['sha256']!=sha256(path):raise ValueError('Author baseline output SHA differs')
        for name,expected in author['inputs'].items():
            if name.endswith('raw/trans.csv') and expected!=sha256(raw/'trans.csv'):raise ValueError('Baseline real-child cohort differs')
            if name.endswith('raw/account.csv') and expected!=sha256(raw/'account.csv'):raise ValueError('Baseline parent cohort differs')
        if case['kind']=='new_model':return {**case,'status':'PENDING_COMPLETION_PROOF','reason':'New model needs fixed-final native generation metadata, not author-baseline exemption'}
        if case['label'].startswith('RDBDiff'):
            complete_path=folder/'complete.json';complete=read_json(complete_path)
            protected[str(complete_path)]=sha256(complete_path)
            if not complete.get('Given_N') or not complete.get('fixed_structure'):raise ValueError('RDBDiff declared count/graph regime differs')
        case['scope']='Original full417 same-parent baseline fidelity; original author protocol'
    return {**case,'status':'VERIFIED_NATIVE_INPUT'}


def flat_stats(summary,comparison):
    amount=summary['columns']['log_amount']['expm1_per_row'];cum=summary['columns']['log_cumamt']['expm1_per_row']
    distribution=comparison['distributions'];change=summary['ordered_change_diagnostics']
    return {'events':summary['events'],'empty_parents':summary['temporal']['empty_parents'],
        'amount_mean':amount['mean'],'amount_std_ddof1':amount['std_ddof1'],'amount_p99':amount['p99'],
        'amount_zero_percent':100*amount['zero_fraction_observed'] if amount['zero_fraction_observed'] is not None else None,
        'cumulative_mean':cum['mean'],'cumulative_std_ddof1':cum['std_ddof1'],'cumulative_p99':cum['p99'],
        'customer_mean_amount_SD':summary['customer_patterns']['log_amount.expm1_per_row.mean']['std_ddof0'],
        'cumulative_decrease_percent':100*change['decrease_fraction_observed_adjacent_pairs'] if change['decrease_fraction_observed_adjacent_pairs'] is not None else None,
        'cumulative_increment_MAE':change['retail_delta_cumulative_minus_current_amount']['mean_absolute_error'],
        'N_KS_distance_ALL417':distribution['parent.N_all']['KS_distance'],
        'gap_KS_distance_round9':distribution['time.internal_gap_round9']['KS_distance'],
        'span_KS_distance_active':distribution['time.span_active_round9']['KS_distance']}


def aggregate(rows):
    result=[]
    keys=sorted({(r['label'],str(r.get('training_seed','unknown'))) for r in rows})
    for label,training in keys:
        selected=[r for r in rows if r['label']==label and str(r.get('training_seed','unknown'))==training]
        seeds=[r.get('generation_seed') for r in selected]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate model/training/generation case; no hidden double counting')
        values={}
        for key in selected[0]['statistics']:
            array=[r['statistics'][key] for r in selected]
            values[key]={'mean':float(np.mean(array)) if all(x is not None for x in array) else None,
                'generation_SD_ddof1':float(np.std(array,ddof=1)) if len(array)>1 and all(x is not None for x in array) else None}
        result.append({'label':label,'training_seed':selected[0].get('training_seed'),
            'generation_seeds':seeds,'generation_replicates':len(selected),'kind':selected[0]['kind'],
            'expected_generation_seeds':selected[0].get('expected_generation_seeds'),
            'generation_set_complete':(set(seeds)==set(selected[0]['expected_generation_seeds'])
                if selected[0].get('expected_generation_seeds') else None),
            'count_source':selected[0]['count_source'],'statistics':values})
    return result


def format_number(value):return 'undefined' if value is None else f'{value:.5f}'


def write_visual(path,real,rows):
    """Dependency-free SVG dashboard; all cases and Given-N labels remain visible."""
    panels=[]
    for key,title in [('amount_std_ddof1','Physical amount std'),('amount_p99','Physical amount P99'),
                      ('amount_zero_percent','Amount zero %'),('customer_mean_amount_SD','Customer mean amount SD')]:
        values=[('REAL',real[key])]+[(r['label']+' train'+str(r.get('training_seed')),r['statistics'][key]['mean']) for r in rows]
        maximum=max([value for _,value in values if value is not None]+[1.])
        content=[]
        for index,(label,value) in enumerate(values):
            y=35+index*27;width=0 if value is None else 410*value/maximum
            color='#777' if 'GivenN' in label else '#b94d47' if label=='REAL' else '#367da2'
            content.append(f'<text x="8" y="{y+14}" font-size="11">{html.escape(label)}</text>'
                f'<rect x="230" y="{y}" width="{width:.3f}" height="19" fill="{color}"/>'
                f'<text x="{238+width:.3f}" y="{y+14}" font-size="11">{html.escape(format_number(value))}</text>')
        panels.append(f'<h2>{html.escape(title)}</h2><svg viewBox="0 0 760 {60+len(values)*27}" role="img">'+''.join(content)+'</svg>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>A7 feature teacher comparison</title>'
        '<style>body{font:15px system-ui;max-width:1100px;margin:30px auto}svg{width:100%;border:1px solid #ddd}h2{font-size:18px}</style>'
        '<h1>Full417 physical distributions</h1><p>Generation means within each training seed. Given-N/fixed-graph rows are descriptive contrasts, not count-free competitors. All individual seeds remain in CSV/JSON.</p>'+''.join(panels),encoding='utf-8')


def run(runtime,output,new_samples=None,spec=None,*,raw_root=None,evaluator=None):
    runtime,output=Path(runtime),Path(output)
    if (output/'comparison.json').exists():raise ValueError('Preserve prior report; choose a new output namespace')
    raw_root=Path(raw_root) if raw_root else runtime/'reference/data'
    evaluator=Path(evaluator) if evaluator else runtime/'reference/eval/score.py'
    raw=raw_root/'online_retail/raw';real=pd.read_csv(raw/'trans.csv');parents=pd.read_csv(raw/'account.csv').account_id.tolist()
    if (len(parents),len(real))!=EXPECTED['online_retail']:raise ValueError('Original full417/18284 cohort required')
    scorer=score_module(evaluator)
    protected={str(p):sha256(p) for p in [raw/'trans.csv',raw/'account.csv',evaluator,Path(__file__)]}
    cases=baseline_cases(runtime)+new_cases(new_samples)
    if spec:
        spec=Path(spec);protected[str(spec)]=sha256(spec);cases+=read_json(spec)['samples']
    real_summary,real_arrays,real_table=analyze(real,parents,'online_retail',retail_cumulative_error=True)
    self_compare=compare(real_arrays,real_arrays,real_summary,real_summary)
    real_flat=flat_stats(real_summary,self_compare)
    rows=[];statuses=[];output.mkdir(parents=True,exist_ok=True)
    real_table.to_csv(output/'real_customer_patterns.csv',index=False)
    for index,case in enumerate(cases):
        checked=verify_case(case,raw,protected);statuses.append(checked)
        if checked['status']!='VERIFIED_NATIVE_INPUT':continue
        frame=pd.read_csv(checked['path'])
        summary,arrays,table=analyze(frame,parents,'online_retail',retail_cumulative_error=True)
        compared=compare(real_arrays,arrays,real_summary,summary)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            scores={'ts':scorer.ts(real,frame),'cs':scorer.cs('online_retail',real,frame),
                'sd':scorer.sd('online_retail',real,frame),'lg':scorer.lg('online_retail',real,frame),
                'joint_ff':scorer.jointff(real,frame)}
        clean={key:float(value) if np.isfinite(value) else None for key,value in scores.items()}
        statistics=flat_stats(summary,compared);statistics.update({'paper_'+key:value for key,value in clean.items()})
        rows.append({**checked,'statistics':statistics,'details':summary,'comparison':compared})
        table.to_csv(output/f'case_{index}_customer_patterns.csv',index=False)
        print(checked['label'],checked.get('training_seed'),checked.get('generation_seed'),clean,flush=True)
    summarized=aggregate(rows)
    if any(sha256(Path(path))!=expected for path,expected in protected.items()):raise RuntimeError('Protected input changed')
    result={'scope':'Full417 same-parent IN-SAMPLE fidelity; no heldout generalization claim',
        'real':real_flat,'individual':rows,'within_training_seed_generation_means':summarized,'input_status':statuses,
        'scorer_sha256':SCORER_SHA,'protected_sha256':protected,'protected_sources_unchanged':True,
        'limitations':['Generation SD is not training-seed uncertainty; no best seed selected.',
            'RDBDiff Given-N/fixed-graph row is not evidence of count-free generation.',
            'Existing A7_POINT_OFF is suffix auxiliary-off, not a verified complete no-JEPA pipeline.',
            'Different legacy training budgets/front/teacher reuse costs are not controlled by this descriptive table.',
            'New outputs are scored only after fixed-final provenance; pending/oracle/forecast cases are shown separately.',
            'Native generation artifacts and oracle teacher reconstruction analyses must not be pooled.']}
    write_json(output/'comparison.json',result)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('details','comparison','statistics')}|row['statistics'] for row in rows]).to_csv(output/'individual_seed_statistics.csv',index=False)
    pd.DataFrame([{'label':row['label'],'training_seed':row['training_seed'],'generation_seeds':str(row['generation_seeds']),
        'kind':row['kind'],'count_source':row['count_source'],**{k:v['mean'] for k,v in row['statistics'].items()}} for row in summarized]).to_csv(output/'generation_means_by_training_seed.csv',index=False)
    lines=['# A7 feature-teacher / baseline comparison','',result['scope']+'.','',
        'All available predeclared generation seeds are retained. Each row averages generations within ONE training seed. '
        'Teacher-oracle reconstruction is separate from native generation; pending new paths are not scored as zero.','',
        '| Model / training seed | Gen seeds | TS | CS | SD | LG | Joint FF | Count source |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in summarized:
        stats=row['statistics'];label=row['label']+' / '+str(row['training_seed'])
        if row['generation_set_complete'] is False:label+=' PARTIAL'
        lines.append('| '+label+' | '+str(row['generation_seeds'])+' | '+' | '.join(format_number(stats['paper_'+key]['mean']) for key in ('ts','cs','sd','lg','joint_ff'))+' | '+row['count_source']+' |')
    lines+=['','| Model / training seed | Amount mean | Amount std | Amount P99 | Zero % | Customer mean SD | Cum std | Cum decrease % | Δcum−amount MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| REAL | '+' | '.join(format_number(real_flat[k]) for k in ('amount_mean','amount_std_ddof1','amount_p99','amount_zero_percent','customer_mean_amount_SD','cumulative_std_ddof1','cumulative_decrease_percent','cumulative_increment_MAE'))+' |']
    for row in summarized:
        lines.append('| '+row['label']+' / '+str(row['training_seed'])+' | '+' | '.join(format_number(row['statistics'][k]['mean']) for k in ('amount_mean','amount_std_ddof1','amount_p99','amount_zero_percent','customer_mean_amount_SD','cumulative_std_ddof1','cumulative_decrease_percent','cumulative_increment_MAE'))+' |')
    lines+=['','N/time distances, every individual seed and original log-column statistics are retained in CSV/JSON. '
        'All source-account IDs remain in the N distribution including empty generated parents. '
        'Cum decreases are descriptive; this Retail increment test is evaluated with the same source floating baseline, never enforced in a generator.','',
        '## Pending or excluded inputs','']
    lines += ['- '+item['label']+': '+item['status']+' — '+item.get('reason','') for item in statuses if item['status']!='VERIFIED_NATIVE_INPUT']
    lines+=['','## Limits','',*['- '+x for x in result['limitations']], '', '[Physical-distribution dashboard](comparison.html)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_visual(output/'comparison.html',real_flat,summarized)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path);parser.add_argument('--evaluator',type=Path)
    parser.add_argument('--new-sample',action='append',help='MODEL_LABEL=completed full-native child.csv; repeat for seeds')
    parser.add_argument('--new-manifest',type=Path,help='JSON samples list withlabel/path andoptional seed/provenancefields')
    args=parser.parse_args()
    run(args.runtime_root,args.output,args.new_sample,args.new_manifest,raw_root=args.raw_root,evaluator=args.evaluator)
