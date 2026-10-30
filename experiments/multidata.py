"""Full-budget, independent-seed JETS study on the four paper data tables.

Each training seed prepares its own data bank, both target encoders, both
suffix caches and generator. Each trained checkpoint produces three native
generations. The source evaluator and original raw tables are never modified.
This runner is intentionally sequential on one accelerator; it does not run or
label external baselines. Inputs are trusted local research artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.data import read_json, sha256, write_json
from experiments.retail import source_hashes


DATASETS = {
    'online_retail': {'numeric': ['log_amount', 'log_cumamt'], 'categorical': [],
                      'parent_numeric': [], 'parent_categorical': ['Country']},
    'berka': {'numeric': ['log_amount', 'log_balance'], 'categorical': ['cat_code'],
              'parent_numeric': [], 'parent_categorical': ['frequency', 'district_id_attr']},
    'coupon': {'numeric': ['log_page', 'log_cumpage'], 'categorical': ['cat_code'],
               'parent_numeric': ['AGE'], 'parent_categorical': ['SEX_ID', 'PREF_NAME']},
    'sparkov': {'numeric': ['log_amount', 'log_cumamt'], 'categorical': ['cat_code'],
                'parent_numeric': ['age'], 'parent_categorical': ['gender', 'state']},
}
TRAIN_SEEDS = (1, 2, 3)
GENERATION_SEEDS = (1, 2, 3)
SCORER_SHA256 = 'f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing file differs; preserve prior study: {path}')
    else:
        write_json(path, value)


def _run(argv, logfile, cwd):
    """Capture output for restart diagnosis; errors never silently advance."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps([str(arg) for arg in argv], ensure_ascii=False) + '\n')
        stream.flush()
        process = subprocess.run([str(arg) for arg in argv], cwd=cwd, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
        stream.write(f'EXIT {process.returncode}\n')
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}); inspect {logfile}')


def _score(real, generated, scorer, dataset, dest, cwd):
    argv = [sys.executable, '-m', 'experiments.retail', 'evaluate', '--real', real,
            '--synthetic', generated, '--scorer', scorer, '--dataset', dataset]
    proc = subprocess.run([str(arg) for arg in argv], cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(f'Evaluation failed for {generated}: {proc.stderr[-3000:]}')
    value = json.loads(proc.stdout)
    if value.get('scorer_sha256') != SCORER_SHA256:
        raise ValueError('Original evaluator digest changed')
    value.update(dataset=dataset, real_sha256=sha256(real), child_sha256=sha256(generated))
    _immutable_json(dest, value)


def run(raw_root, scorer, output, config_path, device='cuda'):
    raw_root, scorer, output, config_path = map(lambda p: Path(p).resolve(),
                                                (raw_root, scorer, output, config_path))
    source = Path(__file__).resolve().parents[1]
    if output == raw_root or output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError('Study output must not overlap original inputs')
    if sha256(scorer) != SCORER_SHA256:
        raise ValueError('Expected the original pinned evaluator')
    config = read_json(config_path)
    if config.get('generator_steps') != 20000 or config.get('batch_size') != 16:
        raise ValueError('This study requires the predeclared full generator budget')
    target = config.get('target_encoder', {})
    if target.get('steps', 5000) != 5000 or target.get('batch_size', 32) != 32:
        raise ValueError('This study requires both full target encoder budgets')
    raw = {dataset: raw_root / dataset / 'raw' for dataset in DATASETS}
    hashes = {str(folder / name): sha256(folder / name) for folder in raw.values()
              for name in ('account.csv', 'trans.csv')}
    identity = {'raw_sha256': hashes, 'scorer_sha256': SCORER_SHA256,
                'base_config_sha256': sha256(config_path), 'source_sha256': source_hashes(),
                'datasets': list(DATASETS), 'train_seeds': list(TRAIN_SEEDS),
                'generation_seeds_per_training': list(GENERATION_SEEDS), 'device': device,
                'condition': 'deterministic feature representation; no sampled epsilon',
                'scope': 'full supplied populations; in-sample fidelity; not heldout generalization',
                'budgets': '4096 windows, target encoders 5000 each, generator 20000, generation 50 steps'}
    manifest = output / 'study.json'
    if output.exists() and any(output.iterdir()) and not manifest.exists():
        raise FileExistsError('Non-empty output is not a JETS multidataset study')
    output.mkdir(parents=True, exist_ok=True)
    _immutable_json(manifest, identity)
    py = sys.executable
    for dataset, schema in DATASETS.items():
        for seed in TRAIN_SEEDS:
            if source_hashes() != identity['source_sha256']:
                raise RuntimeError('Study source changed; preserve existing results')
            if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
                raise RuntimeError('Original source data changed')
            run_dir = output / dataset / f'train_{seed}'
            run_dir.mkdir(parents=True, exist_ok=True)
            seed_config = {**config, 'training_seed': seed}
            cfg = run_dir / 'config.json'; schema_path = run_dir / 'schema.json'
            _immutable_json(cfg, seed_config); _immutable_json(schema_path, schema)
            prepared = run_dir / 'prepared'
            _run([py, '-m', 'experiments.pipeline', '--raw-dir', raw[dataset], '--config', cfg,
                  '--output', prepared, '--device', device, '--windows', '4096', '--schema', schema_path],
                 run_dir / 'pipeline.log', source)
            receipt = read_json(prepared / 'complete.json')
            if receipt.get('exploratory_budget') or receipt.get('target_steps') != 5000:
                raise ValueError('Target encoders did not use the full declared budget')
            trained = prepared / 'generator' / 'final.pt'
            complete = prepared / 'generator' / 'complete.json'
            if trained.exists():
                if not complete.is_file() or read_json(complete).get('checkpoint_sha256') != sha256(trained):
                    raise ValueError('Existing generator checkpoint is incomplete or changed')
            else:
                _run(receipt['generator_train_command'], run_dir / 'generator.log', source)
            if read_json(complete).get('updates') != 20000:
                raise ValueError('Generator did not finish the declared budget')
            for gen_seed in GENERATION_SEEDS:
                folder = run_dir / f'generation_{gen_seed}'
                child, generated_receipt = folder / 'child.csv', folder / 'generation.json'
                if child.exists():
                    if not generated_receipt.is_file() or read_json(generated_receipt).get('child_sha256') != sha256(child):
                        raise ValueError('Existing generated path changed or is incomplete')
                else:
                    art = receipt['artifacts']
                    _run([py, '-m', 'experiments.retail', 'generate', '--bank', art['bank']['path'],
                          '--checkpoint', trained, '--preprocessor', art['preprocessor']['path'],
                          '--parents', art['parents']['path'], '--output', folder, '--device', device,
                          '--seed', str(gen_seed), '--sample-steps', '50'], run_dir / f'generation_{gen_seed}.log', source)
                info = read_json(generated_receipt)
                if (info.get('count_or_face_or_teacher_oracle') is not False or
                        info.get('checkpoint_sha256') != sha256(trained) or
                        info.get('generation_seed') != gen_seed):
                    raise ValueError('Generated path is not the requested native checkpoint/seed')
                _score(raw[dataset] / 'trans.csv', child, scorer, dataset, folder / 'metrics.json', source)
                print(f'{dataset} training seed {seed}, generation seed {gen_seed}: complete', flush=True)
    _immutable_json(output / 'complete.json', {'status': 'complete', 'study_sha256': sha256(manifest),
                                              'training_runs': 12, 'native_generations': 36})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root', 'scorer', 'output', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.raw_root, args.scorer, args.output, args.config, args.device)


if __name__ == '__main__':
    main()
