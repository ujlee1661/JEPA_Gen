"""Matched A9 native experiment with N-conditioned time-pattern teacher.

Teacher and A9 generator are fitted on all 417 entities: fidelity, not
heldout generalization. Teacher sees actual future only while learning targets.
"""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import torch

import pgjepa.a7_training as a7
from pgjepa.a7_data import collate_a7
from pgjepa.a9_conditional_time_teacher import A9ConditionalTimeTeacher
from pgjepa.experiment import read_json, seed_all, write_json
from pgjepa.paper_data import sha256
from pgjepa.paper_train import evaluate_paper_outputs
from pgjepa.a7_distribution_teacher_training import paired_step_inputs
from scripts.run_retail_a7_full_fidelity import export_native
import scripts.run_a9_retail as base

ROOT = base.ROOT
OUT = ROOT/'retail_a9_conditioned_time_v1/online_retail'
ARM = 'A9_TIME_COND'

def teacher_factory(data, device):
    return A9ConditionalTimeTeacher(data['state']['token_dim'],data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'],code_dim=64,width=128,layers=2,heads=4).to(device)

def prepare():
    data=a7.load_data(base.DATA)
    origin=read_json(base.OUTPUT/'manifest.json')
    config=copy.deepcopy(origin['config']);config['arms']=[ARM];config['aux_weights']={ARM:1.}
    config['model_name']='A9 conditioned time teacher matched student'
    paths=[base.DATA/'data.pt',base.OUTPUT/'generator_initial.pt',base.FEATURE/'teacher/final.pt',
           base.FEATURE/'teacher/suffix_codes.pt',Path(__file__).resolve(),
           Path(__file__).resolve().parents[1]/'pgjepa/a9_conditional_time_teacher.py',
           Path(__file__).resolve().parents[1]/'pgjepa/a9_model.py',Path(base.__file__).resolve()]
    hashes={str(p):sha256(p) for p in paths}
    feature=torch.load(paths[3],map_location='cpu',weights_only=False)
    if len(data['records'])!=417 or feature['data_sha256']!=hashes[str(paths[0])] or feature['teacher_checkpoint_sha256']!=hashes[str(paths[2])]:
        raise ValueError('Feature data or checkpoint identity differs')
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'config':config,'data_sha256':hashes[str(paths[0])],'sources':hashes,'scope':'FULL417 in-sample fidelity; not heldout',
              'teacher':'timing-only encoder; N,calendar,attrs-conditioned face NLL + unit Gaussian CLR NLL',
              'native_future_oracle':False,'teacher_steps':5000,'teacher_batch':32,'generator_steps':20000,'feature_aux_weight':1.}
    path=OUT/'manifest.json'
    if path.exists() and read_json(path)!=manifest: raise ValueError('Scientific identity changed')
    if not path.exists(): write_json(path,manifest)
    src=torch.load(paths[1],map_location='cpu',weights_only=False)
    init=OUT/'generator_initial.pt'
    if not init.exists(): a7.atomic_save({'model':src['model'],'config':config},init)
    saved=torch.load(init,map_location='cpu',weights_only=False)
    if a7.tensor_digest(saved['model'])!=a7.tensor_digest(src['model']) or saved['config']!=config: raise ValueError('A9 common initialization mismatch')
    model=base.factory(data,config);model.load_state_dict(saved['model'],strict=True)
    return data,config,feature['codes'],hashes

def train_teacher(data,device):
    folder=OUT/'teacher';folder.mkdir(exist_ok=True);done=folder/'final.pt'
    model=teacher_factory(data,device)
    if done.exists():
        cp=torch.load(done,map_location=device,weights_only=False)
        if cp['manifest_sha256']!=sha256(OUT/'manifest.json') or cp['step']!=5000: raise ValueError('Teacher checkpoint mismatch')
        model.load_state_dict(cp['model']);return model.freeze()
    seed_all(9);model=teacher_factory(data,device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    latest=folder/'latest.pt';start=1;elapsed=0.;history=[]
    if latest.exists():
        cp=torch.load(latest,map_location=device,weights_only=False)
        if cp['manifest_sha256']!=sha256(OUT/'manifest.json'):raise ValueError('Teacher resume provenance differs')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);start=cp['step']+1;elapsed=cp['elapsed'];history=cp['history']
    started=time.perf_counter();pair={'seed':9,'batch_size':32,'teacher':{'code_dim':64}}
    for step in range(start,5001):
        model.train();batch,_,_,_=paired_step_inputs(data,pair,step,device)
        loss,stats=model.loss(batch)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Teacher loss not finite')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise RuntimeError('Teacher gradient not finite')
        opt.step()
        if step==1 or step%100==0 or step==5000:
            history.append({'step':step,'loss':float(loss.detach()),'stats':stats})
            a7.atomic_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'history':history,
                'elapsed':elapsed+time.perf_counter()-started,'manifest_sha256':sha256(OUT/'manifest.json')},latest)
            print(f'TIME_TEACHER {step}/5000 loss={float(loss):.6f}',flush=True)
    model.freeze()
    with torch.no_grad():
        codes=torch.cat([model.encode(collate_a7(data['windows'][i:i+32],device)).cpu() for i in range(0,len(data['windows']),32)])
        if not bool(torch.isfinite(codes).all()):raise ValueError('Invalid code calibration')
        model.set_code_calibration(codes.to(device))
    a7.atomic_save({'model':model.state_dict(),'step':5000,'constructor':'A9ConditionalTimeTeacher',
        'manifest_sha256':sha256(OUT/'manifest.json'),'calibrated_training_only':True},done)
    write_json(folder/'complete.json',{'updates':5000,'checkpoint_sha256':sha256(done),'elapsed_seconds':elapsed+time.perf_counter()-started})
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','teacher','all'],default='prepare');p.add_argument('--device',default='cuda');args=p.parse_args()
    data,config,features,hashes=prepare()
    if args.stage=='prepare':return
    teacher=train_teacher(data,args.device)
    if args.stage=='teacher':return
    codes=a7.cache_suffix_codes(teacher,data,OUT,config,args.device)
    hashes={**hashes,str(OUT/'teacher/final.pt'):sha256(OUT/'teacher/final.pt'),str(OUT/'teacher/suffix_codes.pt'):sha256(OUT/'teacher/suffix_codes.pt')}
    base.OUTPUT=OUT;base.ARMS={ARM:1.}
    base.train(data,config,[codes,features],hashes,ARM,args.device)
    a7.make_model=base.factory
    a7.generate(data,base.PREPARED,OUT,config,args.device,[ARM],population='train_bos')
    export_native(OUT,config,read_json(base.PREPARED/'parents.json'),[ARM])
    print(evaluate_paper_outputs(ROOT/'reference/data','online_retail',OUT,ROOT/'reference/eval/score.py',arms=[ARM],interface='native'),flush=True)

if __name__=='__main__':main()
