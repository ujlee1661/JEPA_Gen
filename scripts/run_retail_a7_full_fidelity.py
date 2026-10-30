"""Full417 A7 population fidelity study; no held-out generalization claim.

Frozen A7 training functions are reused. Default teacher stage never starts a
generator. Native exports copy original child bytes without changing values.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import shutil

import pandas as pd
import torch

from pgjepa.a6_heldout_data import calibration_digest
from pgjepa.a7_teacher_audit import audit_teacher
from pgjepa.a7_training import (a7_config, cache_suffix_codes, generate,
    load_data, prepare_run, source_hashes, train_generators, train_teacher)
from pgjepa.experiment import read_json, write_json
from pgjepa.paper_data import load_prepared_paper, sha256
from pgjepa.paper_train import evaluate_paper_outputs


SCORER_SHA = 'f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'
SCOPE = 'FULL417 training-population fidelity and in-sample descriptive code-use; NOT held-out generalization'
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT/'configs/retail_a7_full_fidelity.json'


def full_config(path=CONFIG_PATH):
    config = a7_config(read_json(path))
    strict = a7_config(read_json(ROOT/'configs/retail_a7_strict.json'))
    if {k:v for k,v in config.items() if k!='model_name'} != {k:v for k,v in strict.items() if k!='model_name'}:
        raise ValueError('Full417 and strict335 training hyperparameters must match exactly')
    return config


def validate_full_population(data, bundle, parent_ids):
    if len(data['records']) != 417 or len(data['attrs']) != 417 or len(parent_ids) != 417 or len(set(map(str,parent_ids))) != 417:
        raise ValueError('Full fidelity requires all417 unique training parents')
    if len(bundle['records']) != 417 or len(bundle['queries']) != 3753:
        raise ValueError('Full417 descriptive bundle must contain417 records and3753 queries')
    if bundle.get('is_heldout') is not False or bundle.get('is_generalization_evaluation') is not False:
        raise ValueError('Full417 queries must explicitly disclaim held-out generalization')
    if calibration_digest(data['state']) != calibration_digest(bundle['state']):
        raise ValueError('Full417 queries must retain the full417 training calibration')
    if set(map(str,bundle['entity_ids'])) != set(map(str,parent_ids)):
        raise ValueError('Prepared and diagnostic populations differ')
    indices = torch.as_tensor(bundle['audit_indices'])
    if indices.ndim != 1 or len(indices) != 738 or len(indices.unique()) != 738 or bool((indices<0).any()) or bool((indices>=3753).any()):
        raise ValueError('Require738 distinct deterministic diagnostic query indices')
    return indices.long()


def _pin_files(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)):sha256(path) for path in sorted(directory.rglob('*')) if path.is_file()}


def prepare_study(data_dir, prepared, query_path, output, raw_root, evaluator, config, config_path=CONFIG_PATH):
    data = load_data(data_dir)
    bundle = torch.load(query_path,map_location='cpu',weights_only=False)
    parents = read_json(Path(prepared)/'parents.json')
    validate_full_population(data,bundle,parents)
    records,_,_ = load_prepared_paper(prepared)
    if len(records) != 417:
        raise ValueError('Full417 prepared population required')
    real = pd.read_csv(Path(raw_root)/'online_retail/raw/trans.csv')
    if set(map(str,real.account_id.unique())) != set(map(str,parents)):
        raise ValueError('Raw scorer population differs from all417 prepared parents')
    if sha256(evaluator) != SCORER_SHA:
        raise ValueError('Exact original frozen paper evaluator required')
    query_builder = ROOT/'scripts/prepare_a7_full_fidelity_queries.py'
    identity = {'scope':SCOPE,'is_generalization_evaluation':False,'fit_entities':417,
        'data_sha256':sha256(Path(data_dir)/'data.pt'),'queries_sha256':sha256(query_path),
        'prepared_files':_pin_files(prepared),'raw_files':_pin_files(Path(raw_root)/'online_retail/raw'),
        'evaluator_sha256':sha256(evaluator),'frozen_core_sources':source_hashes(),
        'wrapper_sha256':sha256(__file__),'config_sha256':sha256(config_path),
        'query_builder_sha256':sha256(query_builder),'config':config,
        'teacher_gate':'in-sample descriptive code-use; unchanged decision rule; no gate override',
        'initial_stage':'teacher+diagnostic only; generator requires explicit train stage',
        'native_values':'byte-identical child.csv export from frozen train_bos generator'}
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    scope_path = output/'study_scope.json'
    if scope_path.exists() and read_json(scope_path) != identity:
        raise ValueError('Full417 study identity changed; preserve this namespace')
    write_json(scope_path,identity)
    # Keep the frozen manifest's exact original formula. Scope is a sidecar.
    prepare_run(data_dir,output,config,query_path)
    return data,bundle


def _relabel(value):
    if isinstance(value,dict):
        return {_relabel(key):_relabel(item) for key,item in value.items()}
    if isinstance(value,list):
        return [_relabel(item) for item in value]
    if isinstance(value,str):
        value = re.sub('held-out','in-sample diagnostic',value,flags=re.IGNORECASE)
        return re.sub('heldout','diagnostic',value,flags=re.IGNORECASE)
    return value


def canonical_audit(result):
    canonical = _relabel(copy.deepcopy(result))
    canonical.update(scope=SCOPE,is_generalization_evaluation=False,is_heldout=False,
        diagnostic_population='subset of the same417 entities used to fit teacher and preprocessing',
        legacy_helper_output='internal_legacy_helper_output; historical field names retained only for provenance')
    if canonical['decision'] != result['decision'] or canonical['summary'] != result['summary']:
        raise RuntimeError('Scope relabeling must preserve every decision and summary metric')
    return canonical


def descriptive_gate(teacher,data,bundle,prepared,output,config,device):
    indices = validate_full_population(data,bundle,read_json(Path(prepared)/'parents.json'))
    audit_bundle = dict(bundle)
    audit_bundle['queries'] = [bundle['queries'][int(index)] for index in indices]
    internal = Path(output)/'internal_legacy_helper_output'
    _,prep,_ = load_prepared_paper(prepared)
    original = audit_teacher(teacher,data,audit_bundle,internal,device=device,
        batch_size=config['teacher_batch_size'],prep=prep)
    internal.joinpath('README.md').write_text(
        '# Internal unmodified legacy helper output\n\n'+SCOPE+'\n\nHistorical held-out labels in this directory are helper naming only. '
        'Use ../teacher_audit/teacher_audit.json for the canonical scope. No held-out generalization is measured here.\n',encoding='utf-8')
    canonical = canonical_audit(original)
    destination = Path(output)/'teacher_audit'
    destination.mkdir(exist_ok=True)
    write_json(destination/'teacher_audit.json',canonical)
    lines = ['# Full417 in-sample teacher code-use diagnostic','',SCOPE,'',
        f'Decision: **{canonical["decision"]}**. Queries: {canonical["diagnostic_queries"]}.','',
        'Teacher reconstruction uses true N/face/calendar. This is not native generation.','',
        '| Control | Mark MSE | Scaled CLR MSE |','|---|---:|---:|']
    for name,values in canonical['summary'].items():
        lines.append(f'| {name} | {values["mark_mse"]} | {values["time_clr_scaled_mse"]} |')
    lines.extend(['','The original decision and numeric metrics are unchanged. A PASS does not establish rich or lossless embeddings.'])
    (destination/'teacher_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return canonical


def export_native(output,config,parent_ids,arms=None):
    output = Path(output)
    for arm in arms or config['arms']:
        for seed in config['generation_seeds']:
            source = output/'generating'/arm/'train_bos'/str(seed)
            destination = output/'generating'/arm/'native'/str(seed)
            receipt = read_json(source/'generation.json')
            if receipt['count_or_face_or_teacher_oracle'] or sha256(source/'child.csv') != receipt['child_sha256']:
                raise ValueError('Native full417 requires intact free-count nonoracle output')
            counts = pd.read_csv(source/'counts.csv')
            if len(counts)!=417 or counts.entity.nunique()!=417 or set(map(str,counts.entity))!=set(map(str,parent_ids)):
                raise ValueError('Native output counts must cover all417 parents, including empty paths')
            destination.mkdir(parents=True,exist_ok=True)
            for name in ['child.csv','counts.csv']:
                target = destination/name
                if target.exists() and sha256(target)!=sha256(source/name):
                    raise ValueError('Existing native export changed')
                if not target.exists():
                    shutil.copyfile(source/name,target)
                if sha256(target)!=sha256(source/name):
                    raise RuntimeError('Native export changed file bytes')
            coverage = {'fixed_parents':417,'nonempty_parents':int((counts['count']>0).sum()),
                        'zero_event_parents':int((counts['count']==0).sum()),'generated_events':int(counts['count'].sum())}
            write_json(destination/'generation.json',{**receipt,'interface':'native','scope':SCOPE,
                'coverage':coverage,'source_population':'train_bos','is_generalization_evaluation':False,
                'source_child_sha256':sha256(source/'child.csv'),'child_sha256':sha256(destination/'child.csv'),
                'count_regime':'Generated-N native; all417 static parents; empty paths retained in counts.csv'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','prepared','queries','output','raw-root','evaluator']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=CONFIG_PATH)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--stage',choices=['prepare','teacher','cache','train','generate','evaluate'],default='teacher')
    parser.add_argument('--arms',nargs='+',choices=['A7_AUX_ON','A7_AUX_OFF'])
    args = parser.parse_args()
    config = full_config(args.config)
    data,bundle = prepare_study(args.data_dir,args.prepared,args.queries,args.output,args.raw_root,args.evaluator,config,args.config)
    if args.stage=='prepare':
        return
    if args.stage=='teacher':
        teacher = train_teacher(data,args.output,config,args.device)
        result = descriptive_gate(teacher,data,bundle,args.prepared,args.output,config,args.device)
        print('Full417 in-sample teacher diagnostic:',result['decision'],flush=True)
        return  # Never implicitly launch generator training.
    if args.stage in ['cache','train']:
        gate = read_json(args.output/'teacher_audit/teacher_audit.json')
        if gate['decision']!='PASS' or gate.get('is_generalization_evaluation') is not False:
            raise RuntimeError('Full417 descriptive teacher gate requires review; no generator launched')
        teacher = train_teacher(data,args.output,config,args.device)
        codes = cache_suffix_codes(teacher,data,args.output,config,args.device)
        if args.stage=='train':
            train_generators(data,args.output,config,codes,args.device,args.arms)
    if args.stage=='generate':
        generate(data,args.prepared,args.output,config,args.device,args.arms,population='train_bos')
        export_native(args.output,config,read_json(args.prepared/'parents.json'),args.arms)
    if args.stage=='evaluate':
        export_native(args.output,config,read_json(args.prepared/'parents.json'),args.arms)
        print(evaluate_paper_outputs(args.raw_root,'online_retail',args.output,args.evaluator,
            arms=args.arms or config['arms'],interface='native'),flush=True)


if __name__=='__main__':
    main()
