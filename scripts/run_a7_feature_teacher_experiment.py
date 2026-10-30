"""Feature-only ENERGY teacher and matched A7 native student, separate study.

Oracle diagnostics condition on actual future time/layout and are not native
quality gates. A separate numerical-execution gate records the human-requested
continuation even when code-use diagnostics are negative.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from pgjepa.a7_data import collate_a7
from pgjepa.a7_distribution_teacher_training import experiment_sources,paired_step_inputs
from pgjepa.a7_teacher_audit import paired_entity_bootstrap,raw_code_statistics
from pgjepa.a7_training import (atomic_save,cache_suffix_codes,generate,load_data,
    make_model,tensor_digest,train_generators)
from pgjepa.experiment import read_json,seed_all,write_json
from pgjepa.paper_data import sha256
from pgjepa.paper_train import evaluate_paper_outputs
from scripts.run_retail_a7_full_fidelity import export_native,validate_full_population


ROOT=Path(__file__).resolve().parents[1]
DEFAULT={'seed':9,'teacher_steps':5000,'teacher_batch_size':32,'teacher_learning_rate':.0003,
    'weight_decay':.01,'gradient_clip':1.,'log_interval':100,'oracle_samples':32,'oracle_batch_size':8,
    'oracle_seed':9147,'selection':'fixed_final_update_no_validation',
    'teacher':{'code_dim':64,'width':128,'layers':2,'heads':4,'decoder_layers':4}}


def config(overrides=None,*,scientific=True):
    value=copy.deepcopy(DEFAULT)
    for key,item in (overrides or {}).items():
        if key=='teacher':value[key].update(item)
        else:value[key]=copy.deepcopy(item)
    if scientific and value!=DEFAULT:raise ValueError('Fixed feature teacher5k and original student20k budgets required')
    return value


def sources():
    from pgjepa import a7_feature_teacher
    return {**experiment_sources(),str(Path(a7_feature_teacher.__file__).resolve()):sha256(a7_feature_teacher.__file__),
        str(Path(__file__).resolve()):sha256(__file__),str(ROOT/'scripts/run_retail_a7_full_fidelity.py'):sha256(ROOT/'scripts/run_retail_a7_full_fidelity.py')}


def make_feature_teacher(data,cfg,device='cpu'):
    from pgjepa.a7_feature_teacher import A7FeatureTeacher
    model=A7FeatureTeacher(data['state']['token_dim'],data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'],**cfg['teacher']).to(device)
    if type(model).__name__!='A7FeatureTeacher':raise TypeError('Explicit feature-teacher class required')
    return model


def load_feature_teacher(path,data,cfg,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('constructor')!='A7FeatureTeacher' or saved['experiment_config']!=cfg or saved['step']!=cfg['teacher_steps'] or not saved['calibrated_after_freeze']:
        raise ValueError('Completed feature teacher checkpoint/class/config differs')
    if saved['data_sha256']!=sha256(Path(saved['data_reference'])/'data.pt'):
        raise ValueError('Feature teacher data identity changed')
    model=make_feature_teacher(data,cfg,device);model.load_state_dict(saved['model'],strict=True)
    return model.freeze()


def prepare(data_dir,prepared,queries,original,output,cfg):
    data_dir,prepared,queries,original,output=map(Path,(data_dir,prepared,queries,original,output))
    if output.resolve()==original.resolve() or output.resolve().is_relative_to(original.resolve()):raise ValueError('Separate new experiment namespace required')
    data=load_data(data_dir);bundle=torch.load(queries,map_location='cpu',weights_only=False)
    validate_full_population(data,bundle,read_json(prepared/'parents.json'))
    origin=read_json(original/'manifest.json')
    if origin['fit_entities']!=417 or origin['data_sha256']!=sha256(data_dir/'data.pt'):raise ValueError('Identical full417 original study required')
    gcfg=copy.deepcopy(origin['config']);gcfg['model_name']='A7 FULL417 feature-only time-conditioned ENERGY teacher student'
    gcfg['teacher']=copy.deepcopy(cfg['teacher']);gcfg['teacher_family']='FEATURE_ENERGY';gcfg['arms']=['A7_AUX_ON'];gcfg['aux_weights']={'A7_AUX_ON':1.}
    if origin['config']['generator_steps']!=20000 or origin['config']['batch_size']!=16 or cfg['teacher']['code_dim']!=gcfg['model']['code_dim']:
        raise ValueError('Keep common64D generator20k/b16 architecture')
    initial=torch.load(original/'generator_initial.pt',map_location='cpu',weights_only=False)
    off=torch.load(original/'A7_AUX_OFF/final.pt',map_location='cpu',weights_only=False)
    if off['step']!=20000 or off['arm']!='A7_AUX_OFF' or off['initialization_digest']!=tensor_digest(initial['model']):raise ValueError('Completed common OFF provenance differs')
    identity={'config':gcfg,'experiment_config':cfg,'code':sources(),'data_reference':str(data_dir),'data_sha256':origin['data_sha256'],
        'data_manifest':data['manifest'],'fit_entities':417,'prepared_reference':str(prepared),
        'prepared_files':{name:sha256(prepared/name) for name in ('population.pt','parents.json','preprocessor.json','manifest.json')},
        'queries_reference':str(queries),'queries_sha256':sha256(queries),'original_reference':str(original),
        'original_manifest_sha256':sha256(original/'manifest.json'),'common_initialization_digest':tensor_digest(initial['model']),
        'common_off_reference':str(original/'A7_AUX_OFF/final.pt'),'common_off_sha256':sha256(original/'A7_AUX_OFF/final.pt'),
        'scope':'FULL417 same-parent fidelity; teacher marks-only oracle uses TRUE future time/layout, not native inputs',
        'teacher_constructor':'A7FeatureTeacher','native_teacher_code_input':False,
        'diagnostic_policy':'record actual feature-code tests; human explicitly requested native experiment even if diagnostic negative; no information-preservation claim'}
    output.mkdir(parents=True,exist_ok=True);manifest=output/'manifest.json'
    if manifest.exists() and read_json(manifest)!=identity:raise ValueError('Feature study identity changed; preserve namespace')
    write_json(manifest,identity)
    fresh=make_model(data,gcfg);fresh.load_state_dict(initial['model'],strict=True)
    initial_path=output/'generator_initial.pt'
    if not initial_path.exists():atomic_save({'model':initial['model'],'config':gcfg,
        'initialization':'metadata-only derivation of original common untrained A7 seed9 tensor bytes',
        'source_reference':str(original/'generator_initial.pt'),'source_sha256':sha256(original/'generator_initial.pt')},initial_path)
    derived=torch.load(initial_path,map_location='cpu',weights_only=False)
    if derived['config']!=gcfg or tensor_digest(derived['model'])!=identity['common_initialization_digest']:raise ValueError('Common generator initialization changed')
    (output/'PROSPECTIVE_PROTOCOL.md').write_text('# A7 feature-only teacher experiment\n\n'
        'New feature teacher5k/b32, explicit A7FeatureTeacher factory, feature-only ENERGY conditional on actual futuretime/layout. '
        'Same seed9 untrained896670 generator tensors,20k/b16/AdamW3e-4,100v/DDIM50,native1/2/3. Truecodes onlyauxiliarytrainingloss. '
        'Actual zero/shuffle/time-only-encode oracle diagnostics are descriptive, not trained time-only comparisons or information guarantees. '
        'The requested native experiment continues if these are negative; numerical readiness and diagnostic outcome are separate receipts. '
        'Old completed OFF remains an unchanged reference; reuse requires actual new finite-cache CPU/CUDA10step invariance. '
        'No actual future N/time/face/code enters native sampling, no value correction. All parent/tail/variance/coherence and original f254 evaluated.\n',encoding='utf-8')
    return data,bundle


def guard(output):
    output=Path(output);m=read_json(output/'manifest.json')
    if m['code']!=sources():raise ValueError('Feature experiment scientific source changed')
    for key,expected in [('data_reference','data_sha256'),('queries_reference','queries_sha256'),('common_off_reference','common_off_sha256')]:
        path=Path(m[key]);path=path/'data.pt' if key=='data_reference' else path
        if sha256(path)!=m[expected]:raise ValueError('Protected feature experiment input changed')
    for name,value in m['prepared_files'].items():
        if sha256(Path(m['prepared_reference'])/name)!=value:raise ValueError('Canonical original preprocessing changed')
    return m


def train_teacher(data,output,cfg,device='cuda',stop_after=None):
    output=Path(output);m=guard(output);folder=output/'teacher';folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():return load_feature_teacher(folder/'final.pt',data,cfg,device)
    seed_all(cfg['seed']);model=make_feature_teacher(data,cfg,device)
    initial=folder/'initial.pt'
    if not initial.exists():atomic_save({'model':model.state_dict(),'experiment_config':cfg,'constructor':'A7FeatureTeacher'},initial)
    init=torch.load(initial,map_location='cpu',weights_only=False)
    if init['experiment_config']!=cfg or tensor_digest(init['model'])!=tensor_digest(model.state_dict()):raise ValueError('Feature teacher fresh factory initialization differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['teacher_learning_rate'],weight_decay=cfg['weight_decay'])
    first=1;previous=0.;history=[];latest=folder/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['experiment_config']!=cfg or saved['source_manifest_sha256']!=sha256(output/'manifest.json'):raise ValueError('Feature teacher resume identity changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        first=saved['step']+1;previous=saved['elapsed_seconds'];history=saved['history']
    paircfg={'seed':cfg['seed'],'batch_size':cfg['teacher_batch_size'],'teacher':cfg['teacher']}
    stop=min(cfg['teacher_steps'],stop_after) if stop_after is not None else cfg['teacher_steps'];started=time.perf_counter()
    for step in range(first,stop+1):
        model.train();batch,global_noise,local_noise,inputs=paired_step_inputs(data,paircfg,step,device)
        seed_all(cfg['seed']+104729*step+1)
        loss,stats=model.loss(batch,objective='energy',global_noise_pair=global_noise,local_noise_pair=local_noise)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite feature teacher loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite feature teacher gradient')
        optimizer.step()
        if step==1 or step%cfg['log_interval']==0 or step==stop:
            elapsed=previous+time.perf_counter()-started
            history.append({'step':step,'loss':float(loss.detach()),'diagnostics':stats,'input_receipt':inputs,'elapsed_seconds':elapsed})
            atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'experiment_config':cfg,'constructor':'A7FeatureTeacher',
                'step':step,'history':history,'elapsed_seconds':elapsed,'source_manifest_sha256':sha256(output/'manifest.json')},latest)
            write_json(folder/'history.json',history);print(f'FEATURE teacher {step}/{cfg["teacher_steps"]} loss={float(loss):.6f}',flush=True)
    if stop<cfg['teacher_steps']:return None
    model.freeze()
    with torch.no_grad():
        codes=torch.cat([model.encode(collate_a7(data['windows'][i:i+cfg['teacher_batch_size']],device)).cpu() for i in range(0,len(data['windows']),cfg['teacher_batch_size'])])
        if not bool(torch.isfinite(codes).all()):raise RuntimeError('Nonfinite train codes')
        model.set_code_calibration(codes.to(device))
    atomic_save({'model':model.state_dict(),'experiment_config':cfg,'constructor':'A7FeatureTeacher','step':cfg['teacher_steps'],
        'calibrated_after_freeze':True,'fit_entities':len(data['records']),'data_reference':m['data_reference'],'data_sha256':m['data_sha256'],
        'source_manifest_sha256':sha256(output/'manifest.json'),'objective':'feature-only ENERGY given true futuretime/layout'},folder/'final.pt')
    write_json(folder/'complete.json',{'updates':cfg['teacher_steps'],'constructor':'A7FeatureTeacher','frozen':True,'calibration_fit':'training windows only',
        'checkpoint_sha256':sha256(folder/'final.pt'),'elapsed_seconds':previous+time.perf_counter()-started})
    guard(output);return model


def feature_energy(samples,batch):
    truth=batch['tokens'][...,1:];mask=batch['event_valid'][...,None]
    pred=torch.where(mask,samples[...,1:],torch.zeros_like(samples[...,1:])).transpose(0,1).flatten(2)
    target=torch.where(mask,truth,torch.zeros_like(truth)).flatten(1)
    distance=torch.linalg.vector_norm(pred-target[:,None],dim=-1).mean(-1)
    pair=torch.cdist(pred,pred);pair.diagonal(dim1=-2,dim2=-1).zero_();count=samples.shape[0]
    dof=(batch['count']*(batch['tokens'].shape[-1]-1)).clamp_min(1).sqrt()
    return (distance-.5*pair.sum((1,2))/(count*(count-1)))/dof


@torch.no_grad()
def diagnose(data,bundle,output,cfg,device='cuda'):
    output=Path(output);model=load_feature_teacher(output/'teacher/final.pt',data,cfg,device)
    queries=[bundle['queries'][int(i)] for i in bundle['audit_indices']];size=cfg['oracle_batch_size']
    encode=lambda items:torch.cat([model.encode(collate_a7(items[i:i+size],device)).cpu() for i in range(0,len(items),size)])
    proper=encode(queries);permutation=np.random.default_rng(cfg['oracle_seed']).permutation(len(queries));time_codes=[]
    for i in range(0,len(queries),size):
        b=collate_a7(queries[i:i+size],device);b['tokens']=b['tokens'].clone();b['tokens'][...,1:]=0;time_codes.append(model.encode(b).cpu())
    controls={'proper':proper,'zero_code_time_only_decoder':torch.zeros_like(proper),'shuffled':proper[permutation],'time_only_encode':torch.cat(time_codes)}
    scores:dict[str,list[float]]={name:[] for name in controls};before=tensor_digest(model.state_dict())
    for i in range(0,len(queries),size):
        b=collate_a7(queries[i:i+size],device);conditions=model.decoder_conditions(b)
        gen=torch.Generator(device=device).manual_seed(cfg['oracle_seed']+i);draws:dict[str,list[torch.Tensor]]={name:[] for name in controls}
        for _ in range(cfg['oracle_samples']):
            global_noise=torch.randn(len(b['count']),model.code_dim,generator=gen,device=device)
            local_noise=torch.randn(*b['valid'].shape,model.token_dim,generator=gen,device=device)
            for name,values in controls.items():draws[name].append(model.sample_reconstruction(values[i:i+len(b['count'])].to(device),conditions,global_noise,local_noise))
        for name,values in draws.items():scores[name].extend(feature_energy(torch.stack(values),b).cpu().tolist())
    entities=[q['entity_id'] for q in queries];comparisons={name:paired_entity_bootstrap(entities,scores['proper'],values) for name,values in scores.items() if name!='proper'}
    statistics=raw_code_statistics(proper,model);benefit=all(v['ci95'][0] is not None and v['ci95'][0]>0 for k,v in comparisons.items() if k!='time_only_encode')
    result={'diagnostic_decision':'POSITIVE_CODE_USE' if statistics['finite'] and statistics['nonconstant'] and benefit else 'NO_POSITIVE_CODE_USE_ESTABLISHED',
        'code_statistics':statistics,'feature_energy_per_query':scores,'paired_entity_comparisons':comparisons,'queries':len(queries),
        'scope':'same417 training-population descriptive feature-only oracle; actual futuretimeline/layout supplied to ALL controls; not native fidelity/generalization',
        'zero_code_scope':'same trained decoder, code fixed TRAINmean0; actual timeline retained; NOT retrained optimized time-only baseline',
        'time_only_encode_scope':'encoder sees time and zeroed mark channels, still an encoded control not a code-free trained baseline',
        'student_continuation':'human-requested native experiment proceeds regardless of descriptive diagnostic direction',
        'information_preservation_proven':False,'teacher_state_unchanged':before==tensor_digest(model.state_dict())}
    write_json(output/'teacher_diagnostics.json',result);guard(output);return result


def cache_and_readiness(data,output,cfg,device='cuda'):
    output=Path(output);m=guard(output);model=load_feature_teacher(output/'teacher/final.pt',data,cfg,device)
    codes=cache_suffix_codes(model,data,output,m['config'],device)
    valid=len(codes)==len(data['windows']) and all(z.shape==(w['count']+1,64) and bool(torch.isfinite(z).all()) for z,w in zip(codes,data['windows']))
    if not valid:raise RuntimeError('Actual complete finite64D suffix cache is required for numerical execution')
    directory=output/'teacher_audit';directory.mkdir(exist_ok=True)
    diagnostic=read_json(output/'teacher_diagnostics.json') if (output/'teacher_diagnostics.json').exists() else {'diagnostic_decision':'NOT_YET_RUN'}
    write_json(directory/'teacher_audit.json',{'decision':'PASS','gate_kind':'NUMERICAL_EXECUTION_READINESS_NOT_FEATURE_QUALITY',
        'actual_checks':{'explicit_feature_teacher_class':True,'teacher_completed_fixed5k':True,'complete_finite64D_suffix_cache':valid},
        'diagnostic_decision':diagnostic['diagnostic_decision'],'information_preservation_proven':False,
        'authorization':'human explicitly requested native experiment even if descriptive code-use tests negative',
        'teacher_checkpoint_sha256':sha256(output/'teacher/final.pt'),'cache_sha256':sha256(output/'teacher/suffix_codes.pt')})
    return codes


def student(data,output,cfg,device='cuda'):
    output=Path(output);m=guard(output);codes=cache_and_readiness(data,output,cfg,device)
    train_generators(data,output,m['config'],codes,device,['A7_AUX_ON']);guard(output)


def evaluate(output,raw_root,evaluator):
    output=Path(output);m=guard(output);gcfg=m['config']
    result=evaluate_paper_outputs(raw_root,'online_retail',output,evaluator,arms=['A7_AUX_ON'],interface='native')
    if len(result)!=3:raise ValueError('All three native generation seeds required')
    from scripts.evaluate_native_path_distributions import evaluate as raw_evaluate
    samples={};counts={};labels={}
    for seed in (1,2,3):
        name=f'A7_FEATURE_TEACHER_GEN{seed}';folder=output/'generating/A7_AUX_ON/native'/str(seed)
        samples[name]=folder/'child.csv';counts[name]=folder/'counts.csv';labels[name]='Generated N; A7FeatureTeacher targets only in auxiliary training, not native conditions'
    raw_evaluate('online_retail',raw_root,samples,output/'raw_feature_report',counts=counts,count_sources=labels,retail_cumulative_error=True)
    write_json(output/'comparison_references.json',{'original_A7':m['original_reference'],
        'all_old_models':'outputs/paper_reproduction/overnight_final_comparison_v1/all_results.json',
        'notes':'ARGN1generation; RDB GivenN+graph not freecount; teacher family coordinate NLLs not comparable',
        'own_generation_seeds':gcfg['generation_seeds']})
    guard(output);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','prepared','queries','original','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','teacher','diagnose','cache','student','generate','evaluate','all'],default='prepare')
    parser.add_argument('--device',default='cuda');parser.add_argument('--raw-root',type=Path);parser.add_argument('--evaluator',type=Path)
    args=parser.parse_args();cfg=config();data,bundle=prepare(args.data_dir,args.prepared,args.queries,args.original,args.output,cfg)
    if args.stage in ('teacher','all'):train_teacher(data,args.output,cfg,args.device)
    if args.stage in ('diagnose','all'):print(diagnose(data,bundle,args.output,cfg,args.device)['diagnostic_decision'],flush=True)
    if args.stage in ('cache','all'):cache_and_readiness(data,args.output,cfg,args.device)
    if args.stage in ('student','all'):student(data,args.output,cfg,args.device)
    if args.stage in ('generate','all'):
        generate(data,args.prepared,args.output,read_json(args.output/'manifest.json')['config'],args.device,['A7_AUX_ON'],population='train_bos')
        export_native(args.output,read_json(args.output/'manifest.json')['config'],read_json(args.prepared/'parents.json'),['A7_AUX_ON'])
    if args.stage in ('evaluate','all'):
        if args.raw_root is None or args.evaluator is None:raise ValueError('Original raw/evaluator paths required')
        print(evaluate(args.output,args.raw_root,args.evaluator),flush=True)


if __name__=='__main__':main()
