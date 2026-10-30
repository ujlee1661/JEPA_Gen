"""A7 fixed teacher, explicit held-out gate, matched on/off generator training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pgjepa.a7_training import (a7_config, prepare_run, train_teacher, run_teacher_gate,
                                cache_suffix_codes, train_generators, generate)
from pgjepa.experiment import read_json
from pgjepa.paper_data import load_prepared_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--heldout-queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=['teacher', 'gate', 'cache', 'train', 'generate', 'all'], default='all')
    parser.add_argument('--arms', nargs='+', choices=['A7_AUX_ON', 'A7_AUX_OFF'])
    parser.add_argument('--generation-population', choices=['train_bos', 'heldout_queries'], default='train_bos')
    args = parser.parse_args()
    config = a7_config(read_json(args.config) if args.config else None)
    data = prepare_run(args.data_dir, args.output, config, args.heldout_queries)
    teacher = None
    if args.stage in ['teacher', 'gate', 'cache', 'all']:
        teacher = train_teacher(data, args.output, config, args.device)
    if args.stage in ['teacher', 'gate', 'all']:
        _, prep, _ = load_prepared_paper(args.prepared)
        result = run_teacher_gate(teacher, data, args.heldout_queries, args.output, config, args.device, prep=prep)
        print('A7 teacher gate:', result['decision'], flush=True)
        if args.stage == 'all' and result['decision'] != 'PASS':
            raise RuntimeError('Teacher gate needs review; generator training was not launched')
    if args.stage in ['cache', 'train', 'all']:
        if teacher is None:
            teacher = train_teacher(data, args.output, config, args.device)
        codes = cache_suffix_codes(teacher, data, args.output, config, args.device)
    if args.stage in ['train', 'all']:
        train_generators(data, args.output, config, codes, args.device, args.arms)
    if args.stage in ['generate', 'all']:
        queries = None
        if args.generation_population == 'heldout_queries':
            queries = torch.load(args.heldout_queries, map_location='cpu', weights_only=False)['queries']
        generate(data, args.prepared, args.output, config, args.device, args.arms, queries,
                 population=args.generation_population)


if __name__ == '__main__':
    main()
