"""Create planned ClavaDDPM cases with independently synthesized parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multidata import DATASETS, GENERATION_SEEDS, TRAIN_SEEDS


def build(root):
    root = Path(root).resolve()
    cases = []
    budget = {'diffusion_iterations_per_table': 20000, 'classifier_iterations': 5000,
              'batch_size': 4096, 'diffusion_steps': 1000, 'num_clusters': 20}
    for dataset in DATASETS:
        for train_seed in TRAIN_SEEDS:
            training = root / 'clavaddpm' / dataset / f'train_{train_seed}'
            for generation_seed in GENERATION_SEEDS:
                directory = training / f'generation_{generation_seed}'
                cases.append({'dataset': dataset, 'model': 'ClavaDDPM', 'training_seed': train_seed,
                    'generation_seed': generation_seed, 'parent_identity': 'generated',
                    'count_source': 'generated counts with synthesized parent table',
                    'child': str(directory / 'child.csv'), 'counts': str(directory / 'counts.csv'),
                    'generated_parents': str(directory / 'parent.csv'), 'generated_parent_column': 'entity',
                    'receipt': str(directory / 'generation.json'), 'metrics': str(directory / 'metrics.json'),
                    'training_evidence': str(training / 'training_complete.json'),
                    'training_evidence_profile': 'clavaddpm', 'training_budget_field': 'budgets',
                    'expected_training_budget': budget,
                    'adapter_semantics': 'native adapter lower-bounds time at zero and rounds categorical code'})
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
    print(f'Wrote 36 planned ClavaDDPM cases: {args.output}')


if __name__ == '__main__':
    main()
