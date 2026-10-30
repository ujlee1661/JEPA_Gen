"""Create planned TabARGN case records without changing baseline outputs."""
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
            training = root / 'tabularargn' / dataset / f'train_{train_seed}'
            for generation_seed in GENERATION_SEEDS:
                directory = training / 'generated' / f'seed_{generation_seed}'
                cases.append({'dataset': dataset, 'model': 'TabARGN', 'training_seed': train_seed,
                    'generation_seed': generation_seed, 'parent_identity': 'reference',
                    'count_source': 'generated N; fixed reference parent attributes',
                    'child': str(directory / 'child.csv'), 'counts': str(directory / 'counts_native_tabularargn.csv'),
                    'counts_value_column': 'N', 'receipt': str(directory / 'complete.json'),
                    'metrics': str(directory / 'complete.json'),
                    'training_evidence': str(training / 'training_complete.json'),
                    'training_budget_field': 'max_epochs', 'expected_training_budget': 100})
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
    print(f'Wrote 36 planned TabARGN cases: {args.output}')


if __name__ == '__main__':
    main()
