"""Run predeclared matched ENERGY/MSE teachers; no generator training."""
import argparse

from pgjepa.a7_distribution_teacher_training import (distribution_config,
    prepare_distribution_run, train_distribution_teacher, evaluate_distribution_teacher)
from pgjepa.experiment import read_json
from pgjepa.paper_data import load_prepared_paper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--heldout', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arm', choices=['energy', 'mse', 'both'], default='both')
    parser.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'all'], default='prepare')
    args = parser.parse_args()
    config = distribution_config(read_json(args.config) if args.config else None)
    data = prepare_distribution_run(args.data, args.output, config, args.heldout, args.split_manifest)
    if args.stage != 'prepare':
        for objective in (config['objectives'] if args.arm == 'both' else [args.arm]):
            teacher = train_distribution_teacher(data, args.output, config, objective, args.device,
                stop_after=0 if args.stage == 'evaluate' else None)
            if args.stage in ('evaluate', 'all'):
                if teacher is None:
                    raise ValueError('Evaluation requires completed fixed-budget teacher')
                _, prep, _ = load_prepared_paper(args.prepared)
                evaluate_distribution_teacher(teacher, data, args.heldout,
                    f'{args.output}/{objective.upper()}', config, prep, args.device)
