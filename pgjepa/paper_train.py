"""Declared own-model training and unchanged paper fidelity scoring."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from .experiment import digest, read_json, seed_all, write_json
from .observation_codec import attach_quantile_feedback
from .paper_data import SCHEMAS, load_prepared_paper, sha256
from .paper_model import PaperPGGenerator, paper_sample_batch, parent_generate_clock
from .quantile_ae import QuantileEventAE
from .window_train import exposure_generation_loss


DEFAULT_CONFIG: dict[str, Any] = {
    'protocol': 'parent_only_full_fidelity', 'training_seed': 9,
    'generation_seeds': [1, 2, 3], 'arms': ['A0', 'A1', 'A2', 'A3', 'A4', 'A6'],
    'ae_steps': 3000, 'jepa_steps': 1000, 'generation_steps': 3000,
    'batch_size': 64, 'row_batch_size': 512, 'log_interval': 100,
    'horizons': [4, 8, 16], 'sample_steps': 50, 'max_generated_events': 4096,
    'learning_rates': {'A': .001, 'B': .0003, 'C': .0003},
    'model': {'latent_dim': 64, 'ae_hidden': 256, 'hidden': 128, 'entity_dim': 32,
              'width': 256, 'blocks': 4, 'diffusion_steps': 100, 'diffusion_beta_max': .1},
    'checkpoint_selection': 'fixed_final_update_no_validation',
}


def paper_config(overrides=None):
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if key in {'model', 'learning_rates'}:
            result[key].update(value)
        else:
            result[key] = copy.deepcopy(value)
    if result['protocol'] != 'parent_only_full_fidelity':
        raise ValueError('Different conditioning protocols cannot share this paper run')
    if result['checkpoint_selection'] != 'fixed_final_update_no_validation':
        raise ValueError('Paper fidelity path does not select checkpoints on metric scores')
    if result.get('customer_latent', False) and (result['horizons'] != [1] or
            not set(result['arms']).issubset({'A2', 'A4'})):
        raise ValueError('Hierarchical experiment uses A2/A4 and next-event JEPA only')
    return result


def build_paper_generator(codec, prep, config, arm):
    cfg = config['model']
    generator = PaperPGGenerator
    if config.get('customer_latent', False):
        from .hierarchical_model import HierarchicalPGGenerator
        generator = HierarchicalPGGenerator
    return generator(copy.deepcopy(codec), attrs_dim=prep.attrs_dim, arm=arm,
        horizons=tuple(config['horizons']), hidden=cfg['hidden'], entity_dim=cfg['entity_dim'],
        width=cfg['width'], blocks=cfg['blocks'], diffusion_steps=cfg['diffusion_steps'],
        diffusion_beta_max=cfg['diffusion_beta_max'], zero_gap=True)


def paper_auxiliary_batch(records, count, rng, device, horizons=(4, 8, 16)):
    """Full prefixes from BOS; unavailable longer future blocks stay masked."""
    minimum, maximum = min(horizons), max(horizons)
    longest = max(len(record['delta']) for record in records)
    if longest < minimum:
        raise ValueError('No complete future block is available')
    cut = int(rng.integers(0, longest - minimum + 1))
    eligible = [record for record in records if len(record['delta']) >= cut + minimum]
    selected = [eligible[int(index)] for index in rng.integers(len(eligible), size=count)]
    width = cut + maximum
    keys = ('x_num', 'x_cat', 'mask', 'delta')
    batch = {key: selected[0][key].new_zeros((count, width, selected[0][key].shape[-1])) for key in keys}
    batch['t_valid'] = torch.zeros(count, width, dtype=torch.bool)
    batch['attrs'] = torch.stack([record['attrs'] for record in selected])
    for row, record in enumerate(selected):
        length = min(width, len(record['delta']))
        for key in keys:
            batch[key][row, :length] = record[key][:length]
        batch['t_valid'][row, :length] = True
    return {key: value.to(device) for key, value in batch.items()}, cut


def _rows(records, device):
    return {key: torch.cat([record[key] for record in records]).to(device)
            for key in ('x_num', 'x_cat', 'mask', 'delta', 't_valid')}


def _code_hashes():
    names = ('paper_data.py', 'paper_model.py', 'paper_train.py', 'quantile.py', 'quantile_ae.py',
             'models.py', 'stable_diffusion.py', 'window_train.py', 'observation_codec.py',
             'hierarchical_model.py', 'hierarchical_training.py')
    return {name: sha256(Path(__file__).parent / name) for name in names}


def _fixed_stage(model, stage, records, config, directory, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = config[{'A': 'ae_steps', 'B': 'jepa_steps', 'C': 'generation_steps'}[stage]]
    seed = int(config['training_seed'])
    if (directory / 'complete.json').exists():
        saved = torch.load(directory / 'final.pt', map_location=device)
        if saved['config'] != config or saved['stage'] != stage or saved['step'] != steps:
            raise ValueError('Stage identity differs from the declared fixed budget')
        model.load_state_dict(saved['model'])
        return read_json(directory / 'complete.json')
    if stage != 'A':
        model.configure_stage(stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rates'][stage])
    seed_all(seed)
    rng = np.random.default_rng(seed)
    all_rows = _rows(records, device) if stage == 'A' else None
    customer_summaries = None
    if stage == 'C' and config.get('customer_latent', False):
        from .hierarchical_training import cache_customer_summaries
        customer_summaries = cache_customer_summaries(model, records, device)
    history = []
    first_step = 1
    resume = directory / 'latest.pt'
    if resume.exists():
        saved = torch.load(resume, map_location=device)
        if saved['config'] != config or saved['stage'] != stage:
            raise ValueError('Interrupted stage belongs to another experiment')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        first_step = saved['step'] + 1
        history = read_json(directory / 'history.json') if (directory / 'history.json').exists() else []
    started = time.perf_counter()
    for step in range(first_step, steps + 1):
        model.train()
        if stage == 'A':
            indices = torch.as_tensor(rng.integers(len(all_rows['x_num']), size=config['row_batch_size']), device=device)
            batch = {key: value[indices] for key, value in all_rows.items()}
            loss, stats = model.reconstruction_loss(batch)
        elif stage == 'B':
            batch, cut = paper_auxiliary_batch(records, config['batch_size'], rng, device, config['horizons'])
            loss, stats = model.jepa_loss(batch, cut)
        else:
            batch = paper_sample_batch(records, config['batch_size'], rng, device)
            if customer_summaries is not None:
                from .hierarchical_training import hierarchical_generation_loss
                loss, stats = hierarchical_generation_loss(model, batch, customer_summaries,
                    mean_terms=float(np.mean([len(record['delta']) + 1 for record in records])))
            else:
                loss, stats = exposure_generation_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite {stage} loss at update {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f'Nonfinite {stage} gradient at update {step}')
        optimizer.step()
        if (stage == 'B' or (stage == 'C' and customer_summaries is not None and model.arm == 'A4')) and model.arm != 'A6':
            model.update_ema()
        if step == 1 or step % config['log_interval'] == 0 or step == steps:
            row = {'step': step, 'training_loss': float(loss.detach()), 'diagnostics': stats,
                   'gradient_norm': float(gradient), 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            write_json(directory / 'history.json', history)
            saved = {'model': model.state_dict(), 'config': config, 'stage': stage, 'step': step,
                     'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            temporary = directory / 'latest.pt.tmp'
            torch.save(saved, temporary)
            temporary.replace(resume)
            print(f'{directory} stage{stage} {step}/{steps} loss={row["training_loss"]:.6f}', flush=True)
    # Final, never lowest-loss or best-metric, checkpoint is the scientific artifact.
    saved = {'model': model.state_dict(), 'config': config, 'stage': stage, 'step': steps,
             'training_seed': seed, 'selection': 'fixed_final_update_no_validation'}
    torch.save(saved, directory / 'final.pt')
    result = {'stage': stage, 'updates': steps, 'training_seed': seed,
              'selection': saved['selection'], 'elapsed_seconds_this_process': time.perf_counter() - started,
              'final_training_loss': history[-1]['training_loss'] if history else None}
    write_json(directory / 'complete.json', result)
    return result


def train_paper_campaign(data, output, config=None, *, device='cuda', arms=None):
    config = paper_config(config)
    data, output = Path(data), Path(output)
    records, prep, data_manifest = load_prepared_paper(data)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'data_manifest_hash': digest(data_manifest), 'code': _code_hashes(),
                'protocol': 'same fixed parents, full histories, no validation or true-prefix conditioning'}
    if (output / 'manifest.json').exists() and read_json(output / 'manifest.json') != identity:
        raise ValueError('Paper campaign identity changed; use a new output directory')
    write_json(output / 'manifest.json', identity)
    seed_all(config['training_seed'])
    cfg = config['model']
    codec = QuantileEventAE(prep.n_num, prep.cat_sizes, cfg['latent_dim'], cfg['ae_hidden']).to(device)
    _fixed_stage(codec, 'A', records, config, output / 'AE', device)
    codec.eval()
    if not (output / 'AE/frozen.pt').exists():
        codec.fit_latent_stats(_rows(records, device))
        torch.save(codec.state_dict(), output / 'AE/frozen.pt')
    codec.load_state_dict(torch.load(output / 'AE/frozen.pt', map_location=device))
    selected = list(arms or config['arms'])
    for branch in ('A4', 'A6'):
        if not any(arm in selected for arm in (('A3', 'A4') if branch == 'A4' else ('A6',))):
            continue
        seed_all(config['training_seed'])
        model = build_paper_generator(codec, prep, config, branch).to(device)
        _fixed_stage(model, 'B', records, config, output / f'B_{branch}', device)
    for arm in selected:
        if arm not in config['arms']:
            raise ValueError(f'Undeclared model arm {arm}')
        seed_all(config['training_seed'])
        model = build_paper_generator(codec, prep, config, arm).to(device)
        if arm in {'A3', 'A4', 'A6'}:
            branch = 'A6' if arm == 'A6' else 'A4'
            state = torch.load(output / f'B_{branch}/final.pt', map_location=device)['model']
            model.load_state_dict(state)
        _fixed_stage(model, 'C', records, config, output / arm, device)
        write_json(output / arm / 'parameters.json', model.parameter_counts())
    if all((output / arm / 'complete.json').exists() for arm in config['arms']):
        write_json(output / 'training_complete.json', {'arms': config['arms'], 'training_seed': config['training_seed'],
                   'selection': 'fixed_final_update_no_validation'})


@torch.no_grad()
def generate_on_scaffold(model, attrs, parent_ids, scaffold, *, sample_steps=50):
    """Consume one supplied current time at a time; never inspect future marks."""
    table = scaffold.copy()
    if not {'account_id', 't_offset'}.issubset(table.columns):
        raise ValueError('Scaffold requires account_id and t_offset')
    if not set(table.account_id).issubset(set(parent_ids)):
        raise ValueError('Scaffold contains an unknown fixed parent')
    values = table.t_offset.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Scaffold calendar times must be finite and nonnegative')
    groups = {key: group.sort_values(['seq'] if 'seq' in group else ['t_offset'], kind='stable').t_offset.to_numpy(dtype=np.float64)
              for key, group in table.groupby('account_id', sort=False)}
    sequences = [groups.get(key, np.empty(0, dtype=np.float64)) for key in parent_ids]
    if any(np.any(np.diff(times) < 0) for times in sequences):
        raise ValueError('Scaffold sequence order has decreasing timestamps')
    width = max(map(len, sequences), default=0)
    device = model.initial.weight.device
    attrs = attrs.to(device)
    z, state = model.parent_state(attrs)
    valid = torch.zeros((len(attrs), width), dtype=torch.bool, device=device)
    offsets = torch.zeros((len(attrs), width, 1), dtype=torch.float64, device=device)
    rows: dict[str, list] = {key: [] for key in ('x_num', 'x_cat', 'mask', 'delta')}
    previous = np.zeros(len(attrs), dtype=np.float64)
    for index in range(width):
        active = np.asarray([index < len(times) for times in sequences])
        clock = np.asarray([times[index] if index < len(times) else previous[row]
                            for row, times in enumerate(sequences)])
        delta = torch.tensor((clock - previous)[:, None], dtype=torch.float32, device=device)
        accepted = torch.tensor(active, device=device)
        condition = model._condition(z, state)
        row = model.ae.decode(model.sample_latent(condition, delta, sample_steps), sample=True)
        for key in ('x_num', 'x_cat', 'mask', 'delta'):
            value = delta if key == 'delta' else row[key]
            rows[key].append(torch.where(accepted[:, None], value, torch.zeros_like(value)))
        valid[:, index] = accepted
        offsets[:, index, 0] = torch.tensor(np.where(active, clock, 0.), dtype=torch.float64, device=device)
        if model.arm != 'A0':
            updated = model.feedback(state, row, delta)
            state = torch.where(accepted[:, None], updated, state)
        previous = clock
    result = {}
    for key, pieces in rows.items():
        features = {'x_num': model.ae.n_num, 'x_cat': len(model.ae.cat_sizes),
                    'mask': model.ae.n_num + len(model.ae.cat_sizes), 'delta': 1}[key]
        dtype = torch.long if key == 'x_cat' else torch.float32
        result[key] = torch.stack(pieces, 1) if pieces else torch.zeros((len(attrs), 0, features), device=device, dtype=dtype)
    result.update(t_offset=offsets, t_valid=valid, lengths=valid.sum(1),
                  cap_hit=torch.zeros(len(attrs), dtype=torch.bool, device=device))
    return result


def export_paper_rows(samples, prep, parent_ids, dataset):
    schema = SCHEMAS[dataset]
    raw = prep.inverse_numeric(samples['x_num'])
    masks = samples['mask'].detach().cpu().numpy()
    cats = samples['x_cat'].detach().cpu().numpy()
    valid = samples['t_valid'].detach().cpu().numpy()
    clocks = samples['t_offset'].detach().cpu().numpy()[..., 0]
    pieces = []
    unknown = {name: 0 for name in schema['categorical']}
    for row, parent in enumerate(parent_ids):
        keep = valid[row]
        columns: dict[str, Any] = {'entity': np.full(int(keep.sum()), parent), 't_offset': clocks[row, keep]}
        for j, name in enumerate(schema['numeric']):
            columns[name] = np.where(masks[row, keep, j] >= .5, raw[row, keep, j], np.nan)
        for j, name in enumerate(schema['categorical']):
            encoded = cats[row, keep, j]
            observed = masks[row, keep, prep.n_num + j] >= .5
            decoded = np.full(len(encoded), np.nan)
            for index, value in enumerate(encoded):
                if observed[index] and 1 <= value <= len(prep.vocabularies[j]):
                    decoded[index] = float(prep.vocabularies[j][int(value) - 1])
                elif observed[index]:
                    unknown[name] += 1
            columns[name] = decoded
        pieces.append(pd.DataFrame(columns))
    return pd.concat(pieces, ignore_index=True), {'unknown_category_outputs': unknown,
        'parents': len(parent_ids), 'parents_with_events': int(valid.any(1).sum()),
        'zero_event_parents': int((~valid.any(1)).sum()), 'events': int(valid.sum())}


def generate_paper_samples(data, output, config=None, *, device='cuda', arms=None,
                           interface='native', scaffold_dir=None, nmode='gn'):
    config = paper_config(config)
    data, output = Path(data), Path(output)
    records, prep, manifest = load_prepared_paper(data)
    parent_ids = read_json(data / 'parents.json')
    attrs = torch.stack([record['attrs'] for record in records]).to(device)
    cfg = config['model']
    generated = []
    if interface not in {'native', 'timefirst'} or nmode not in {'gn', 'matched'}:
        raise ValueError('Unknown paper generation interface')
    if interface == 'timefirst' and scaffold_dir is None:
        raise ValueError('TimeFIRST interface requires a unique supplied scaffold directory')
    interface_label = 'native' if interface == 'native' else f'timefirst_{nmode}'
    for arm in arms or config['arms']:
        path = output / arm / 'final.pt'
        saved = torch.load(path, map_location=device)
        if saved['config'] != config or float(saved['model']['alpha_bar'][-1]) != 0:
            raise ValueError('Checkpoint does not match the stable paper experiment')
        codec = QuantileEventAE(prep.n_num, prep.cat_sizes, cfg['latent_dim'], cfg['ae_hidden'])
        model = build_paper_generator(codec, prep, config, arm).to(device)
        model.load_state_dict(saved['model'])
        model.eval()
        # State feedback must encode exactly the observation represented by the
        # external inverse quantile (including endpoint saturation), not an
        # out-of-support decoder coordinate. The wrapper changes no parameters.
        feedback_audit = attach_quantile_feedback(model.ae, prep)
        for seed in config['generation_seeds']:
            seed_all(seed)
            target = output / 'generating' / arm / interface_label / str(seed)
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'generation.json').exists():
                receipt = read_json(target / 'generation.json')
                csv = target / 'child.csv'
                if (receipt['checkpoint_sha256'] != sha256(path) or
                        receipt['data_manifest_hash'] != digest(manifest) or
                        receipt['code_hashes'] != _code_hashes() or
                        not csv.exists() or sha256(csv) != receipt['child_sha256']):
                    raise ValueError('Saved generation provenance changed; use a new run directory')
                generated.append(str(csv))
                continue
            started = time.perf_counter()
            scaffold_path = None
            if interface == 'native':
                samples = parent_generate_clock(model, attrs, manifest['source']['observation_end'],
                    max_events=config['max_generated_events'], sample_steps=config['sample_steps'])
            else:
                scaffold_path = Path(scaffold_dir) / nmode / f'seed_{seed}' / 'skeleton.csv'
                scaffold = pd.read_csv(scaffold_path)
                samples = generate_on_scaffold(model, attrs, parent_ids, scaffold, sample_steps=config['sample_steps'])
            if bool(samples['cap_hit'].any()):
                write_json(target / 'invalid_cap.json', {'count': int(samples['cap_hit'].sum()), 'seed': seed})
                raise RuntimeError('Native generation safety cap exhausted; headline scoring prohibited')
            frame, coverage = export_paper_rows(samples, prep, parent_ids, manifest['source']['dataset'])
            csv = target / 'child.csv'
            frame.to_csv(csv, index=False)
            arrays = {key: samples[key].detach().cpu().numpy() for key in
                      ('x_num', 'x_cat', 'mask', 'delta', 't_offset', 't_valid', 'lengths')}
            if 'customer_z' in samples:
                arrays['customer_z'] = samples['customer_z'].detach().cpu().numpy()
            np.savez_compressed(target / 'samples.npz', **arrays)
            receipt = {'arm': arm, 'interface': interface_label, 'generation_seed': seed,
                'training_seed': config['training_seed'], 'coverage': coverage, 'elapsed_seconds': time.perf_counter() - started,
                'checkpoint_sha256': sha256(path), 'child_sha256': sha256(csv),
                'data_manifest_hash': digest(manifest), 'code_hashes': _code_hashes(),
                'count_regime': 'Generated-N native' if interface == 'native' else
                    ('Generated-N external TimeFIRST count model' if nmode == 'gn' else 'Given-N external real counts'),
                'scaffold': str(scaffold_path) if scaffold_path else None,
                'scaffold_sha256': sha256(scaffold_path) if scaffold_path else None,
                'feedback_canonicalization': dict(feedback_audit),
                'customer_conditioning': ('one prior draw per parent; no real child history at generation'
                                          if config.get('customer_latent', False) else 'deterministic static parent'),
                'scope': 'same-parent training-population fidelity, not held-out test'}
            write_json(target / 'generation.json', receipt)
            generated.append(str(csv))
            print(f'Generated {arm} {interface_label} seed{seed}: {len(frame)} events', flush=True)
    return generated


def evaluate_paper_outputs(raw_root, dataset, output, evaluator_path, *, arms=None, interface='native'):
    """Call exactly the five original functions; do not extend headline Joint-FF."""
    evaluator_path = Path(evaluator_path)
    expected = 'f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'
    if sha256(evaluator_path) != expected:
        raise ValueError('Paper evaluator bytes do not match the original frozen source')
    spec = importlib.util.spec_from_file_location('paper_frozen_score', evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the pinned evaluator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = pd.read_csv(Path(raw_root) / dataset / 'raw/trans.csv')
    output = Path(output)
    results = []
    patterns = [output / 'generating' / arm / interface for arm in arms] if arms else list((output / 'generating').glob(f'*/{interface}'))
    for folder in patterns:
        for csv in sorted(folder.glob('*/child.csv')):
            generated = pd.read_csv(csv)
            receipt = read_json(csv.parent / 'generation.json')
            if sha256(csv) != receipt['child_sha256']:
                raise ValueError('Generated output changed after provenance was written')
            required_columns = {'entity', 't_offset', *SCHEMAS[dataset]['numeric'], *SCHEMAS[dataset]['categorical']}
            if set(generated.columns) != required_columns or generated.empty:
                write_json(csv.parent / 'invalid_evaluation.json', {
                    'reason': 'empty output or incompatible event schema', 'coverage': receipt['coverage']})
                raise ValueError('Empty/incompatible generation is outside the paper scorer valid-input domain')
            if not set(generated.entity).issubset(set(real.account_id)):
                raise ValueError('Generated rows changed the fixed parent population')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                metrics = {'ts': module.ts(real, generated), 'cs': module.cs(dataset, real, generated),
                           'sd': module.sd(dataset, real, generated), 'lg': module.lg(dataset, real, generated),
                           'joint_ff': module.jointff(real, generated)}
            clean = {name: float(value) if np.isfinite(value) else None for name, value in metrics.items()}
            result = {'arm': receipt['arm'], 'interface': interface, 'generation_seed': receipt['generation_seed'],
                      'metrics': clean, 'coverage': receipt['coverage'], 'evaluator_sha256': expected,
                      'scope': receipt['scope']}
            write_json(csv.parent / 'metrics.json', result)
            results.append(result)
    if not results:
        raise ValueError('No completed scientific outputs to evaluate')
    write_json(output / f'{interface}_evaluation.json', results)
    return results
