"""Fresh quantile-codec campaign with terminal exposure survival likelihood.

Stage C samples an entity uniformly and one of its event-prefix likelihood
terms uniformly, including its final censored term. Multiplication by the
number of available terms estimates an entity's full conditional likelihood;
normalization by the dataset mean term count keeps optimization scales manageable.
Only observed history is passed to the generator. Exposure is a loss target.
"""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .experiment import (_ae_rows, _gate_ae, _gate_jepa, _load_split,
                         _restore_rng, _rng_state, digest, read_json, seed_all,
                         source_hashes, train_stage, write_json)
from .models import PGGenerator


def log_gap_survival(head: Any, condition: Tensor, remaining: Tensor) -> Tensor:
    """Log P(next gap >= remaining), stable for a lognormal mixture hurdle.

    The zero atom contributes no survival for strictly positive remaining
    exposure. At zero exposure the probability is one, including that atom.
    """
    remaining = remaining.reshape(-1)
    if not bool(torch.isfinite(remaining).all()) or bool((remaining < 0).any()):
        raise ValueError('Remaining exposure must be finite and nonnegative')
    zero, logits, mean, sigma = head.parameters_at(condition)
    boundary = remaining.clamp_min(torch.finfo(remaining.dtype).tiny).log().unsqueeze(-1)
    positive = torch.logsumexp(F.log_softmax(logits, -1)
                               + torch.special.log_ndtr((mean - boundary) / sigma), -1)
    if head.zero_gap:
        positive = positive - F.softplus(zero)
    return torch.where(remaining > 0, positive, torch.zeros_like(positive))


def sample_exposure_batch(records: list[dict], count: int, rng: np.random.Generator,
                          device: str, history_max_events: int = 64) -> dict[str, Any]:
    """Sample prefix likelihood terms without conditioning on future activity."""
    if not records or count < 1 or history_max_events < 1:
        raise ValueError('Nonempty records, positive batch and history limits required')
    selected = rng.integers(len(records), size=count)
    examples = []
    for index in selected:
        entity = records[int(index)]
        size = len(entity['delta'])
        if size < 1 or 'exposure_end' not in entity:
            raise ValueError('Every entity needs an event prefix and known administrative exposure')
        cut = int(rng.integers(1, size + 1))
        start = max(0, cut - history_max_events)
        times = entity['delta'].reshape(-1).double().cumsum(0)
        end = float(torch.as_tensor(entity['exposure_end']))
        if not np.isfinite(end) or end + 1e-5 < float(times[-1]):
            raise ValueError('Administrative end precedes an observed event')
        remaining = max(0., end - float(times[cut - 1]))
        examples.append((entity, cut, start, remaining, size))
    width = max(cut - start for _, cut, start, _, _ in examples)
    keys = ('x_num', 'x_cat', 'mask', 'delta')
    prototype = examples[0][0]
    prefix = {key: prototype[key].new_zeros((count, width, prototype[key].shape[-1]))
              for key in keys}
    prefix['t_valid'] = torch.zeros(count, width, dtype=torch.bool)
    prefix['attrs'] = torch.stack([e.get('attrs', torch.zeros(0)) for e, *_ in examples])
    target = {key: prototype[key].new_zeros((count, prototype[key].shape[-1])) for key in keys}
    observed = torch.zeros(count, dtype=torch.bool)
    for row, (entity, cut, start, _, size) in enumerate(examples):
        length = cut - start
        for key in keys:
            prefix[key][row, :length] = entity[key][start:cut]
            if cut < size:
                target[key][row] = entity[key][cut]
        # History starts at its own landmark; do not use a gap to an unseen row.
        prefix['delta'][row, 0] = 0
        prefix['t_valid'][row, :length] = True
        observed[row] = cut < size
    weights = torch.tensor([size for *_, size in examples], dtype=torch.float32)
    weights /= float(np.mean([len(entity['delta']) for entity in records]))
    result = {'prefix': {key: value.to(device) for key, value in prefix.items()},
            'target': {key: value.to(device) for key, value in target.items()},
            'observed': observed.to(device),
            'remaining': torch.tensor([r for _, _, _, r, _ in examples], device=device),
            'weights': weights.to(device),
            'cuts': [cut for _, cut, *_ in examples]}
    if any('raw_x_num' in entity for entity, *_ in examples):
        if not all('raw_x_num' in entity for entity, *_ in examples):
            raise ValueError('Raw validation numeric metadata must cover the full cohort')
        result['raw_target_numeric'] = torch.stack([
            torch.as_tensor(entity['raw_x_num'][cut], dtype=torch.float64) if cut < size else
            torch.full((prototype['x_num'].shape[-1],), float('nan'), dtype=torch.float64)
            for entity, cut, _, _, size in examples]).to(device)
    return result


def exposure_generation_loss(model: PGGenerator, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
    """Observed next-event density/diffusion plus terminal right-censoring."""
    z, state = model.init_state(batch['prefix'])
    condition = model._condition(z, state)
    time_condition = model._time_condition(condition)
    observed, target, weights = batch['observed'], batch['target'], batch['weights']
    density = model.time_head.nll(time_condition, target['delta'])
    censoring = -log_gap_survival(model.time_head, time_condition, batch['remaining'])
    time_loss = (torch.where(observed, density, censoring) * weights).mean()
    diffusion = condition.sum() * 0
    if bool(observed.any()):
        with torch.no_grad():
            raw_latent = model.ae.encode({key: value[observed] for key, value in target.items()})
        if hasattr(model, 'latent_diffusion_loss'):
            errors = model.latent_diffusion_loss(raw_latent, condition[observed], target['delta'][observed])
        else:
            latent = model.ae.normalize(raw_latent)
            steps = torch.randint(model.diffusion_steps, (len(latent),), device=latent.device)
            noise = torch.randn_like(latent)
            alpha = model.alpha_bar[steps].unsqueeze(-1)
            noisy = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
            predicted = model.denoiser(noisy, steps, torch.cat([
                condition[observed], target['delta'][observed].log1p()], -1))
            errors = (predicted - noise).square().mean(-1)
        # Divide by the whole batch so sampling censored terms does not upweight
        # entities with few events through conditional event-only averaging.
        diffusion = (errors * weights[observed]).sum() / len(weights)
    loss = time_loss + diffusion
    return loss, {'time_nll': float(time_loss.detach()), 'diffusion_loss': float(diffusion.detach()),
                  'censored_fraction': float((~observed).float().mean()),
                  'observed_fraction': float(observed.float().mean())}


def make_window_codec(prep: Any, config: dict) -> Any:
    """Select the original MLP AE or explicitly requested legacy token VAE.

    Quantile preprocessing is independent of the event-codec architecture.
    The architecture-preserving path never imports or constructs a VAE.
    """
    kind = config.get('codec', 'tabsyn_vae')
    if kind == 'mlp_ae':
        from .quantile_ae import QuantileEventAE

        cfg = config['model']
        return QuantileEventAE(prep.n_num, prep.cat_sizes,
                               latent_dim=cfg.get('latent_dim', 64),
                               hidden=cfg.get('ae_hidden', 256))
    if kind != 'tabsyn_vae':
        raise ValueError(f'Unsupported event codec {kind}')
    from .tabsyn_vae import TabSynVAE

    cfg = config.get('vae', {})
    return TabSynVAE(prep.n_num, prep.cat_sizes, token_dim=cfg.get('token_dim', 4),
                     layers=cfg.get('layers', 2), heads=cfg.get('heads', 1),
                     beta=cfg.get('beta', .01))


def build_window_generator(vae: Any, prep: Any, config: dict, arm: str) -> PGGenerator:
    cfg = config['model']
    prediction = config.get('diffusion_prediction', 'epsilon')
    generator_class = PGGenerator
    if prediction == 'v':
        from .stable_diffusion import StablePGGenerator

        generator_class = StablePGGenerator
    elif prediction != 'epsilon':
        raise ValueError(f'Unsupported diffusion prediction {prediction}')
    return generator_class(copy.deepcopy(vae), attrs_dim=prep.attrs_dim, arm=arm,
                       horizons=tuple(config['horizons']), hidden=cfg['hidden'],
                       entity_dim=cfg['entity_dim'], width=cfg['width'], blocks=cfg['blocks'],
                       diffusion_steps=cfg['diffusion_steps'], zero_gap=True,
                       diffusion_beta_max=cfg.get('diffusion_beta_max', .1))


def generation_validation_rank(quality: dict) -> tuple[int, int, float, float]:
    """Prefer every passing snapshot to every failed one, independent of KS.

    Failed snapshots are ranked by the number of failed primitive criteria,
    then their total normalized threshold excess, then the aggregate score.
    Missing/nonfinite diagnostics cannot improve a checkpoint's rank.
    """
    score = quality.get('selection_score')
    score = float(score) if score is not None and np.isfinite(score) else 1e12
    if quality.get('all_gates_pass') is True:
        return 0, 0, 0., score
    thresholds = quality.get('gate_thresholds', {})
    failed, excess = 0, 0.
    for row in quality.get('gates', {}).values():
        if not isinstance(row, dict):
            if row is False:
                failed += 1
                excess += 1e6
            continue
        if not row.get('supported', True):
            continue
        if row.get('finite') is False:
            failed += 1
            excess += 1e6
        for flag, metric, threshold in (
            ('zero_fraction_pass', 'zero_fraction_absolute_error', 'zero_fraction_abs_max'),
            ('mean_error_pass', 'mean_error_in_real_std', 'mean_error_std_max'),
        ):
            if row.get(flag) is False:
                failed += 1
                value = row.get(metric)
                limit = thresholds.get(threshold)
                if value is None or limit is None or not np.isfinite(value):
                    excess += 1e6
                else:
                    excess += max(0., float(value) - float(limit)) / max(abs(float(limit)), 1e-8)
        if row.get('std_ratio_pass') is False:
            failed += 1
            value = row.get('std_ratio')
            lower, upper = thresholds.get('std_ratio_min', 0.), thresholds.get('std_ratio_max')
            if value is None or upper is None or not np.isfinite(value):
                excess += 1e6
            else:
                excess += (max(0., float(value) - float(upper)) / max(abs(float(upper)), 1e-8)
                           + max(0., float(lower) - float(value)) / max(abs(float(lower)), 1e-8))
    # all_gates_pass also includes latent/mask/category validity, which need not
    # appear in the per-variable dictionaries supplied by the diagnostic.
    if not failed:
        failed, excess = 1, 1e6
    return 1, failed, float(excess), score


def train_exposure_stage(model: PGGenerator, train: list[dict], val: list[dict],
                         config: dict, output: Path, seed: int, device: str,
                         *, prep: Any = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    model.configure_stage('C')
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rates']['C'])
    seed_all(seed)
    rng = np.random.default_rng(seed)
    best = float('inf')
    best_quality_rank: tuple[int, int, float, float] | None = None
    history = []
    started = time.perf_counter()
    steps = config['diffusion_steps_train']
    quality_enabled = 'validation_generation_gate' in config
    quality_cache = None
    quality_interval = int(config.get('generation_validation_interval', 500))
    if steps < 1 or quality_interval < 1:
        raise ValueError('Positive training and generation validation intervals required')
    if quality_enabled:
        if prep is None:
            raise ValueError('Generation quality selection requires the train-fitted preprocessor')
        from .generation_validation import prepare_generation_validation

        quality_cache = prepare_generation_validation(val, prep, config, device)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        model.train()
        batch = sample_exposure_batch(train, config['batch_size'], rng, device,
                                      config.get('history_max_events', 64))
        loss, _ = exposure_generation_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite exposure objective at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not bool(torch.isfinite(gradient)):
            raise RuntimeError(f'Nonfinite exposure gradient at step {step}')
        optimizer.step()
        quality_due = quality_enabled and (step % quality_interval == 0 or step == steps)
        if step == 1 or step % config['validation_interval'] == 0 or step == steps or quality_due:
            state = _rng_state()
            seed_all(seed + 800000)
            vrng = np.random.default_rng(seed + 800000)
            validation = []
            diagnostics = []
            model.eval()
            with torch.no_grad():
                for _ in range(config['validation_batches']):
                    vb = sample_exposure_batch(val, config['batch_size'], vrng, device,
                                               config.get('history_max_events', 64))
                    vl, stats = exposure_generation_loss(model, vb)
                    validation.append(float(vl))
                    diagnostics.append(stats)
            _restore_rng(state)
            value = float(np.mean(validation))
            merged = {key: float(np.mean([row[key] for row in diagnostics])) for key in diagnostics[0]}
            record = {'step': step, 'train_loss': float(loss.detach()), 'validation_loss': value,
                      'diagnostics': merged, 'gradient_norm': float(gradient),
                      'elapsed_seconds': time.perf_counter() - started}
            checkpoint = {'model': model.state_dict(), 'stage': 'C', 'step': step,
                          'config': config, 'seed': seed, 'validation_loss': value,
                          'diagnostics': merged}
            if quality_due:
                from .generation_validation import generation_validation

                with torch.no_grad():
                    quality = generation_validation(model, quality_cache, prep, config)
                rank = generation_validation_rank(quality)
                record['generation_validation'] = quality
                record['generation_selection_rank'] = list(rank)
                checkpoint['generation_validation'] = quality
                checkpoint['generation_selection_rank'] = list(rank)
                checkpoint['selection_method'] = 'generation_gate_then_failed_count_then_threshold_excess_then_score'
                snapshots = output / 'snapshots'
                snapshots.mkdir(exist_ok=True)
                torch.save(checkpoint, snapshots / f'step_{step:06d}.pt')
                if best_quality_rank is None or rank < best_quality_rank:
                    best_quality_rank = rank
                    torch.save(checkpoint, output / 'best.pt')
            elif not quality_enabled and value < best:
                best = value
                checkpoint['selection_method'] = 'validation_exposure_nll'
                torch.save(checkpoint, output / 'best.pt')
            history.append(record)
            write_json(output / 'history.json', history)
            print(f'{output} stageC {step}/{steps} exposure_val={value:.5f}', flush=True)
    checkpoint = torch.load(output / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    result = {'stage': 'C', 'budget_steps': steps, 'selected_step': checkpoint['step'],
              'validation_loss': checkpoint['validation_loss'], 'diagnostics': checkpoint['diagnostics'],
              'elapsed_seconds': time.perf_counter() - started, 'device': device,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
              'objective': 'next_event_density_and_diffusion_plus_terminal_survival',
              'prefix_sampling': 'uniform_entity_uniform_observed_prefix_including_terminal',
              'term_weights': 'number_of_prefix_terms_normalized_by_dataset_mean',
              'history_max_events': config.get('history_max_events', 64),
              'selection_method': checkpoint.get('selection_method', 'validation_exposure_nll'),
              'generation_validation': checkpoint.get('generation_validation'),
              'quality_gate_pass': (checkpoint['generation_validation']['all_gates_pass']
                                    if quality_enabled else None)}
    write_json(output / 'complete.json', result)
    return result


def train_window_campaign(data: str | Path, output: str | Path, config: dict,
                          device: str = 'cuda') -> None:
    """Train a fresh codec, JEPA/control stages and exposure-aware generation."""
    from .quantile import QuantilePreprocessor

    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ('quantile.py', 'tabsyn_vae.py', 'quantile_ae.py', 'stable_diffusion.py',
             'window_train.py', 'window_data.py', 'generation_validation.py', 'raw_validation.py')
    new_sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in names if (Path(__file__).parent / name).exists()}
    codec_kind = config.get('codec', 'tabsyn_vae')
    raw_validation = None
    if codec_kind == 'mlp_ae' and config.get('quantile_grid') == 'all_observed' and 'validation_generation_gate' in config:
        from .raw_validation import ensure_raw_validation_truth

        prepared = read_json(data / 'manifest.json')
        raw_validation = ensure_raw_validation_truth(data, prepared['config'], prepared['dataset'])
    scope = ('original_mlp_pg_jepa_quantile_' + config.get('diffusion_prediction', 'epsilon')
             + '_prediction_exposure_survival' if codec_kind == 'mlp_ae' else
             'fresh_quantile_tabsyn_vae_exposure_survival_training')
    training_sources = source_hashes()
    if codec_kind == 'mlp_ae':
        required = {'data.py', 'models.py', 'experiment.py', *names}
        training_sources = {path: value for path, value in training_sources.items()
                            if Path(path).name in required}
    manifest = {'config': config, 'data_manifest_hash': digest(read_json(data / 'manifest.json')),
                'source_hashes': training_sources, 'window_sources': new_sources, 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
                'scope': scope}
    if raw_validation is not None:
        manifest['raw_validation_manifest_hash'] = digest(read_json(data / 'val_raw_manifest.json'))
    if (output / 'manifest.json').exists() and read_json(output / 'manifest.json') != manifest:
        raise ValueError('Fresh campaign config/source/data mismatch; use a new output directory')
    write_json(output / 'manifest.json', manifest)
    # No test split is opened in this function or any training-stage helper.
    train, val = _load_split(data, 'train'), _load_split(data, 'val')
    # The legacy auxiliary sampler slices every field and expects event arrays;
    # scalar administrative metadata is a loss target, never an encoder input.
    auxiliary_train = [{key: value for key, value in entity.items() if key != 'exposure_end'}
                       for entity in train]
    auxiliary_val = [{key: value for key, value in entity.items() if key != 'exposure_end'}
                     for entity in val]
    if raw_validation is not None:
        val = [{**entity, 'raw_x_num': torch.as_tensor(values, dtype=torch.float64)}
               for entity, values in zip(val, raw_validation)]
    preprocessing = read_json(data / 'preprocessor.json')
    if digest(preprocessing) != read_json(data / 'manifest.json')['preprocessor_hash']:
        raise ValueError('Train-fitted quantile preprocessing changed')
    prep = QuantilePreprocessor.from_state_dict(preprocessing)
    seed_all(config['seeds'][0])
    vae = make_window_codec(prep, config).to(device)
    codec_directory = 'AE' if codec_kind == 'mlp_ae' else 'VAE'
    directory = output / codec_directory
    if (directory / 'complete.json').exists():
        statistics = read_json(directory / 'complete.json')
        if _gate_ae(statistics, config):
            vae.load_state_dict(torch.load(directory / 'frozen.pt', map_location=device))
    else:
        statistics = train_stage(vae, 'A', train, val, config, directory, config['seeds'][0], device)
        if _gate_ae(statistics, config):
            vae.eval()
            vae.fit_latent_stats(_ae_rows(train, device))
            torch.save(vae.state_dict(), directory / 'frozen.pt')
    if not _gate_ae(statistics, config):
        write_json(output / 'failed_gate.json', {'stage': codec_directory, 'statistics': statistics})
        raise RuntimeError(f'{codec_directory} validation reconstruction gate failed')
    quality_failures = []
    quality_enabled = 'validation_generation_gate' in config
    for seed in config['seeds']:
        for b_arm in ('A4', 'A6'):
            if not any(arm in config['arms'] for arm in (('A3', 'A4') if b_arm == 'A4' else ('A6',))):
                continue
            seed_all(seed)
            model = build_window_generator(vae, prep, config, b_arm).to(device)
            directory = output / f'seed_{seed}' / f'B_{b_arm}'
            if not (directory / 'complete.json').exists():
                statistics = train_stage(model, 'B', auxiliary_train, auxiliary_val, config, directory, seed, device)
            else:
                statistics = read_json(directory / 'complete.json')
            if not _gate_jepa(statistics, config):
                write_json(directory / 'failed_gate.json', statistics)
                raise RuntimeError(f'Stage B validation gate failed for seed{seed} {b_arm}')
        for arm in config['arms']:
            directory = output / f'seed_{seed}' / arm
            if (directory / 'complete.json').exists():
                statistics = read_json(directory / 'complete.json')
                if quality_enabled and statistics.get('quality_gate_pass') is not True:
                    quality_failures.append({'seed': seed, 'arm': arm,
                                             'generation_validation': statistics.get('generation_validation')})
                    write_json(output / 'quality_failures.json', quality_failures)
                    if arm == 'A4':
                        raise RuntimeError(f'Target A4 observation validation gate failed for seed {seed}')
                continue
            seed_all(seed)
            model = build_window_generator(vae, prep, config, arm).to(device)
            if arm in {'A3', 'A4', 'A6'}:
                b_arm = 'A6' if arm == 'A6' else 'A4'
                checkpoint = torch.load(output / f'seed_{seed}' / f'B_{b_arm}' / 'best.pt', map_location=device)
                model.load_state_dict(checkpoint['model'])
            statistics = train_exposure_stage(model, train, val, config, directory, seed, device, prep=prep)
            write_json(directory / 'parameters.json', model.parameter_counts())
            if quality_enabled and statistics.get('quality_gate_pass') is not True:
                quality_failures.append({'seed': seed, 'arm': arm,
                                         'generation_validation': statistics.get('generation_validation')})
                write_json(output / 'quality_failures.json', quality_failures)
                if arm == 'A4':
                    raise RuntimeError(f'Target A4 observation validation gate failed for seed {seed}')
    if quality_enabled:
        write_json(output / 'quality_failures.json', quality_failures)
    write_json(output / 'training_complete.json', {'seeds': config['seeds'], 'arms': config['arms'],
               'scope': scope, 'quality_failures': quality_failures})
