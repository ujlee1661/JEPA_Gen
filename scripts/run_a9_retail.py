"""Matched A9 Retail feature-prediction ON/OFF experiment; no future oracle at sampling."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
import pgjepa.a7_training as a7
from pgjepa.a7_data import collate_a7
from pgjepa.a9_model import A9Generator
from pgjepa.experiment import read_json, seed_all, write_json
from pgjepa.paper_data import sha256
from pgjepa.paper_train import evaluate_paper_outputs
from scripts.run_retail_a7_full_fidelity import export_native, validate_full_population

ROOT = Path('/home/kangmin/codex_paper_jepa_20261029_3cd171')
DATA = ROOT/'a7_retail_data_v1'
PREPARED = ROOT/'ours_fast_prepared/online_retail'
ORIGINAL = ROOT/'retail_a7_full_fidelity_v1/online_retail'
FEATURE = ROOT/'retail_a7_feature_teacher_v1/online_retail'
OUTPUT = ROOT/'retail_a9_v1/online_retail'
ARMS = {'A9_FEATURE_ON': 1., 'A9_FEATURE_OFF': 0.}

def factory(data, config, device='cpu'):
    return A9Generator(data['state']['token_dim'], data['attrs'].shape[1],
        data['state']['mark_token_state']['observation_end'], time_scale=data['state']['time_scale'],
        **config['model']).to(device)

def prepare():
    data = a7.load_data(DATA)
    queries = torch.load(ROOT/'a7_retail_full_fidelity_v1_queries/queries.pt', map_location='cpu', weights_only=False)
    validate_full_population(data, queries, read_json(PREPARED/'parents.json'))
    old = read_json(ORIGINAL/'manifest.json')
    config = copy.deepcopy(old['config'])
    config['model_name'] = 'A9 matched additional feature JEPA branch'
    config['arms'] = list(ARMS)
    config['aux_weights'] = {name: 1. for name in ARMS}
    if config['generator_steps'] != 20000 or config['batch_size'] != 16:
        raise ValueError('Original A7 training budget mismatch')
    sources = [DATA/'data.pt', ORIGINAL/'teacher/final.pt', ORIGINAL/'teacher/suffix_codes.pt',
               ORIGINAL/'generator_initial.pt', FEATURE/'teacher/final.pt', FEATURE/'teacher/suffix_codes.pt',
               Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'pgjepa/a9_model.py']
    hashes = {str(p): sha256(p) for p in sources}
    caches = [torch.load(p, map_location='cpu', weights_only=False) for p in (sources[2], sources[5])]
    for cache, checkpoint in zip(caches, (sources[1], sources[4])):
        if cache['data_sha256'] != hashes[str(sources[0])] or cache['teacher_checkpoint_sha256'] != hashes[str(checkpoint)]:
            raise ValueError('Teacher cache provenance mismatch')
        if len(cache['codes']) != len(data['windows']) or not all(
            x.shape == (w['count']+1, 64) and bool(torch.isfinite(x).all()) for x,w in zip(cache['codes'],data['windows'])):
            raise ValueError('Incomplete teacher cache')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {'config': config, 'sources': hashes, 'feature_aux_weights': ARMS,
        'scope': 'FULL417 training population, same-data fidelity, not heldout generalization',
        'native_future_teacher_inputs': False, 'comparison': 'identical architecture, feature JEPA loss 1 vs 0'}
    path = OUTPUT/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('A9 experiment identity changed')
    write_json(path, manifest)
    initial_path = OUTPUT/'generator_initial.pt'
    if not initial_path.exists():
        seed_all(config['training_seed'])
        model = factory(data, config)
        original = torch.load(ORIGINAL/'generator_initial.pt', map_location='cpu', weights_only=False)
        result = model.load_state_dict(original['model'], strict=False)
        if result.unexpected_keys or not result.missing_keys or any(not k.startswith(('feature_prediction.', 'feature_residual.')) for k in result.missing_keys):
            raise ValueError(f'Unexpected A9 initialization mismatch: {result}')
        a7.atomic_save({'model': model.state_dict(), 'config': config}, initial_path)
    return data, config, [c['codes'] for c in caches], hashes

def train(data, config, caches, hashes, arm, device):
    directory = OUTPUT/arm
    directory.mkdir(exist_ok=True)
    initial = torch.load(OUTPUT/'generator_initial.pt', map_location='cpu', weights_only=False)
    digest = a7.tensor_digest(initial['model'])
    if (directory/'complete.json').exists():
        done = read_json(directory/'complete.json')
        if done['checkpoint_sha256'] != sha256(directory/'final.pt') or done['updates'] != 20000:
            raise ValueError('Existing completed checkpoint invalid')
        return
    seed_all(config['training_seed'])
    model = factory(data, config, device)
    model.load_state_dict(initial['model'], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    seed_all(config['training_seed'])
    rng = np.random.default_rng(config['training_seed'])
    aux_rng = torch.Generator(device=device).manual_seed(config['training_seed']+1000)
    feature_rng = torch.Generator(device=device).manual_seed(config['training_seed']+2000)
    history, first, elapsed0 = [], 1, 0.
    latest = directory/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['config'] != config or saved['arm'] != arm or saved['sources'] != hashes:
            raise ValueError('Resume provenance mismatch')
        a7._restore(saved, model, opt, rng)
        aux_rng.set_state(saved['aux_rng'].cpu()); feature_rng.set_state(saved['feature_rng'].cpu())
        first, elapsed0 = saved['step']+1, saved['elapsed_seconds']
        history = read_json(directory/'history.json')
    start = time.perf_counter()
    for step in range(first, config['generator_steps']+1):
        model.train()
        indices = rng.integers(len(data['windows']), size=config['batch_size'])
        batch = collate_a7([data['windows'][int(i)] for i in indices], device)
        targets = a7.dense_targets(caches[0], indices, batch['tokens'].shape[1], device, aux_rng, config['code_noise_std'])
        features = a7.dense_targets(caches[1], indices, batch['tokens'].shape[1], device, feature_rng, config['code_noise_std'])
        steps = torch.randint(model.diffusion_steps, (len(indices),), device=device)
        noise = torch.randn_like(batch['tokens'])
        loss, stats = model.loss(batch, targets, targets[:,0], 1., feature_teacher_codes=features,
            feature_aux_weight=ARMS[arm], noise=noise, steps=steps)
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite A9 loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite A9 gradient')
        opt.step()
        if step == 1 or step % config['log_interval'] == 0 or step == config['generator_steps']:
            elapsed = elapsed0+time.perf_counter()-start
            history.append({'step': step, 'loss': float(loss.detach()), 'diagnostics': stats, 'elapsed_seconds': elapsed,
                'diffusion_noise_signature': a7.tensor_digest({'indices':torch.as_tensor(indices),'steps':steps,'noise':noise})})
            a7._snapshot(model,opt,config,step,rng,elapsed,history,latest,part='generator',arm=arm,
                sources=hashes,aux_rng=aux_rng.get_state(),feature_rng=feature_rng.get_state())
            print(f'{arm} {step}/{config["generator_steps"]} loss={loss.item():.6f} elapsed={elapsed:.1f}s',flush=True)
    if any(sha256(Path(path)) != value for path,value in hashes.items()): raise ValueError('Source changed during training')
    final = directory/'final.pt'
    a7.atomic_save({'model':model.state_dict(),'config':config,'part':'generator','arm':arm,'step':config['generator_steps'],
        'training_seed':config['training_seed'],'selection':config['selection'],'initialization_digest':digest,'sources':hashes},final)
    write_json(directory/'complete.json',{'updates':config['generator_steps'],'arm':arm,'initialization_digest':digest,
        'feature_aux_weight':ARMS[arm],'elapsed_seconds':elapsed0+time.perf_counter()-start,'checkpoint_sha256':sha256(final)})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare','train','generate','evaluate','all'], default='prepare')
    p.add_argument('--arm', choices=list(ARMS)); p.add_argument('--device', default='cuda')
    args=p.parse_args(); data, config, caches, hashes=prepare(); arms=[args.arm] if args.arm else list(ARMS)
    if args.stage in ('train','all'):
        for arm in arms: train(data,config,caches,hashes,arm,args.device)
    if args.stage in ('generate','all'):
        a7.make_model=factory
        a7.generate(data,PREPARED,OUTPUT,config,args.device,arms,population='train_bos')
        export_native(OUTPUT,config,read_json(PREPARED/'parents.json'),arms)
    if args.stage in ('evaluate','all'):
        result=evaluate_paper_outputs(ROOT/'reference/data','online_retail',OUTPUT,ROOT/'reference/eval/score.py',arms=arms,interface='native')
        print(result,flush=True)

if __name__=='__main__': main()
