"""Create a read-only manifest of the predetermined JETS 4 x 3 x 3 study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multidata import DATASETS, GENERATION_SEEDS, TRAIN_SEEDS


def build(root):
    root = Path(root).resolve()
    cases = []
    for dataset in DATASETS:
        for train_seed in TRAIN_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                directory = root / dataset / f'train_{train_seed}' / f'generation_{generation_seed}'
                generator = root / dataset / f'train_{train_seed}' / 'prepared' / 'generator'
                cases.append({'dataset': dataset, 'model': 'JETS', 'training_seed': train_seed,
                    'generation_seed': generation_seed, 'count_source': 'generated N and generated face',
                    'parent_identity': 'reference',
                    'child': str(directory / 'child.csv'), 'counts': str(directory / 'counts.csv'),
                    'receipt': str(directory / 'generation.json'),
                    'metrics': str(directory / 'metrics.json'),
                    'checkpoint': str(generator / 'final.pt'),
                    'training_receipt': str(generator / 'complete.json'),
                    'training_manifest': str(generator / 'manifest.json'),
                    'expected_updates': 20000})
    return {'schemas': DATASETS, 'expected_training_seeds': list(TRAIN_SEEDS),
            'expected_generation_seeds': list(GENERATION_SEEDS), 'samples': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(build(args.study_root), stream, ensure_ascii=False, indent=2)
    print(f'Wrote 36 planned cases: {args.output}')


if __name__ == '__main__':
    main()
