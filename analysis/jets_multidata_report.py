"""Compare native temporal tabular generations with their source tables.

This read-only report accepts a JSON list of independent training/generation
replicates from any model. It never assumes supplied N is generated; the caller
must declare the count regime. Incomplete studies may be reported explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from experiments.data import sha256

SCORER_SHA256 = 'f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'


def distribution(values):
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if not len(good):
        return {'total': int(len(values)), 'observed': 0, 'missing': int(len(values)),
                'mean': None, 'std': None, 'p50': None, 'p95': None, 'p99': None,
                'p999': None, 'min': None, 'max': None, 'zero_fraction': None}
    return {'total': int(len(values)), 'observed': int(len(good)), 'missing': int(len(values)-len(good)),
            'mean': float(good.mean()), 'std': float(good.std(ddof=1)) if len(good)>1 else 0.,
            'p50': float(np.quantile(good, .5)), 'p95': float(np.quantile(good, .95)),
            'p99': float(np.quantile(good, .99)), 'p999': float(np.quantile(good, .999)),
            'min': float(good.min()), 'max': float(good.max()), 'zero_fraction': float((good == 0).mean())}


def distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(ks_2samp(a, b).statistic) if len(a) and len(b) else None


def temporal(frame, id_column, parents, counts_path=None):
    if id_column not in frame or 't_offset' not in frame:
        raise ValueError('Native event table needs entity and t_offset columns')
    if not frame[id_column].isin(parents).all():
        raise ValueError('Generated identifiers do not match original parent IDs')
    clean = frame.copy()
    clean['t_offset'] = pd.to_numeric(clean['t_offset'], errors='raise')
    ordered = clean.sort_values([id_column, 't_offset'], kind='stable')
    times = ordered['t_offset'].to_numpy(float)
    if not np.isfinite(times).all():
        raise ValueError('Native event timestamps must be finite')
    groups = ordered.groupby(id_column, sort=False)['t_offset']
    gaps = np.round(groups.diff().dropna().to_numpy(float), 9)
    spans = (groups.max() - groups.min()).to_numpy(float)
    observed = ordered.groupby(id_column).size().reindex(parents, fill_value=0).to_numpy(int)
    if counts_path is not None:
        counts = pd.read_csv(counts_path)
        if counts['entity'].duplicated().any() or set(counts['entity']) != set(parents):
            raise ValueError('Count receipt does not cover the original parents exactly')
        declared = counts.set_index('entity')['count'].reindex(parents).to_numpy(int)
        if not np.array_equal(observed, declared):
            raise ValueError('Count receipt differs from native events')
    return {'counts': observed, 'gaps': gaps, 'spans': spans}


def categorical_tv(real, generated):
    """TV over observed values; separately report missing rates."""
    a, b = pd.Series(real), pd.Series(generated)
    an, bn = pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')
    if an.notna().sum() == a.notna().sum() and bn.notna().sum() == b.notna().sum():
        a, b = an, bn
    else:
        a, b = a.dropna().astype(str), b.dropna().astype(str)
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b):
        return None
    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    keys = p.index.union(q.index)
    return float(.5 * (p.reindex(keys, fill_value=0) - q.reindex(keys, fill_value=0)).abs().sum())


def compare(parent_file, real_file, child_file, numeric, categorical, counts_file=None):
    parent = pd.read_csv(parent_file)
    real, child = pd.read_csv(real_file), pd.read_csv(child_file)
    if parent['account_id'].duplicated().any():
        raise ValueError('Original parent identifiers must be unique')
    parents = parent['account_id'].tolist()
    reference, sample = temporal(real, 'account_id', parents), temporal(child, 'entity', parents, counts_file)
    output = {'parents': len(parents), 'events': len(child), 'count_ks_all_parents': distance(reference['counts'], sample['counts']),
              'internal_gap_ks': distance(reference['gaps'], sample['gaps']), 'active_span_ks': distance(reference['spans'], sample['spans']),
              'count_real': distribution(reference['counts']), 'count_generated': distribution(sample['counts']),
              'gap_real': distribution(reference['gaps']), 'gap_generated': distribution(sample['gaps']),
              'span_real': distribution(reference['spans']), 'span_generated': distribution(sample['spans']),
              'numeric': {}, 'categorical': {}}
    for column in numeric:
        raw_a = pd.to_numeric(real[column], errors='coerce').to_numpy(float)
        raw_b = pd.to_numeric(child[column], errors='coerce').to_numpy(float)
        entry = {'source_real': distribution(raw_a), 'source_generated': distribution(raw_b),
                 'source_ks': distance(raw_a, raw_b)}
        if column.startswith('log_'):
            # Per-row inverse of the supplied log1p columns. No generated value is changed.
            with np.errstate(over='ignore', invalid='ignore'):
                val_a, val_b = np.expm1(raw_a), np.expm1(raw_b)
            entry['expm1_real'] = distribution(val_a)
            entry['expm1_generated'] = distribution(val_b)
            entry['expm1_ks'] = distance(val_a, val_b)
        output['numeric'][column] = entry
    for column in categorical:
        a, b = real[column], child[column]
        output['categorical'][column] = {'real_missing_fraction': float(a.isna().mean()),
            'generated_missing_fraction': float(b.isna().mean()), 'observed_total_variation': categorical_tv(a, b)}
    return output


def scalar_metrics(row):
    """Select comparable scalar values without pooling event rows across seeds."""
    dist = row['distribution']
    result = {key: dist[key] for key in ('count_ks_all_parents', 'internal_gap_ks', 'active_span_ks', 'events')}
    for kind in ('count_generated', 'gap_generated', 'span_generated'):
        if kind in dist:
            for metric in ('mean', 'std', 'p50', 'p95', 'p99', 'zero_fraction'):
                result[f'{kind}.{metric}'] = dist[kind][metric]
    if row.get('metrics'):
        result.update({key: row['metrics'].get(key) for key in ('ts', 'cs', 'sd', 'lg', 'joint_ff')})
    for column, info in dist['numeric'].items():
        for key in ('source_ks', 'expm1_ks'):
            if key in info:
                result[f'numeric.{column}.{key}'] = info[key]
        for representation in ('source_generated', 'expm1_generated'):
            if representation in info:
                for metric in ('mean', 'std', 'p50', 'p95', 'p99', 'p999', 'min', 'max', 'zero_fraction', 'missing'):
                    result[f'numeric.{column}.{representation}.{metric}'] = info[representation].get(metric)
    for column, info in dist['categorical'].items():
        for key in ('observed_total_variation', 'generated_missing_fraction'):
            result[f'categorical.{column}.{key}'] = info[key]
    return result


def aggregates(rows, expected_train_seeds, expected_generation_seeds):
    """Generation replicates are averaged within each independent training run."""
    by_group: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        group = (row['dataset'], row['model'], str(row['training_seed']))
        by_group.setdefault(group, []).append(row)
    train_rows = []
    for (dataset, model, train_seed), values in sorted(by_group.items()):
        seeds = [v['generation_seed'] for v in values]
        if len(seeds) != len(set(map(str, seeds))):
            raise ValueError('Duplicate generation seed in a training run')
        record = {'dataset': dataset, 'model': model, 'training_seed': train_seed,
                  'generation_seeds': seeds, 'count_source': sorted(set(v['count_source'] for v in values)),
                  'complete_generations': set(map(str, seeds)) == set(map(str, expected_generation_seeds)),
                  'complete_scores': all(v.get('metrics') is not None for v in values)}
        flattened = [scalar_metrics(value) for value in values]
        for metric in sorted(set().union(*(x.keys() for x in flattened))):
            selected = [x[metric] for x in flattened if x.get(metric) is not None]
            record[metric + '_mean'] = float(statistics.mean(selected)) if selected else None
        train_rows.append(record)
    model_groups: dict[tuple[str, str], list[dict]] = {}
    for row in train_rows:
        model_groups.setdefault((row['dataset'], row['model']), []).append(row)
    overall = []
    for (dataset, model), values in sorted(model_groups.items()):
        seeds = [v['training_seed'] for v in values]
        row = {'dataset': dataset, 'model': model, 'training_seeds': seeds,
               'complete': set(seeds) == set(map(str, expected_train_seeds)) and all(
                   v['complete_generations'] and v['complete_scores'] for v in values),
               'interpretation': 'within-training generation means, followed by statistics across independent training runs'}
        keys = sorted({key for value in values for key in value if key.endswith('_mean')})
        for key in keys:
            selected = [v[key] for v in values if v.get(key) is not None]
            row[key] = float(statistics.mean(selected)) if selected else None
            row[key[:-5] + '_train_seed_sd'] = float(statistics.stdev(selected)) if len(selected)>1 else None
        overall.append(row)
    return train_rows, overall


def run(reference_root, cases_file, output, *, allow_partial=False):
    reference_root, cases_file, output = Path(reference_root).resolve(), Path(cases_file).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new report path; previous outputs are never overwritten')
    payload = json.loads(cases_file.read_text(encoding='utf-8'))
    case_manifest_digest = sha256(cases_file)
    datasets = {case['dataset'] for case in payload['samples']}
    source_files = {str(reference_root / d / 'raw' / n): sha256(reference_root / d / 'raw' / n)
                    for d in datasets for n in ('account.csv', 'trans.csv')}
    rows, pending = [], []
    for case in payload['samples']:
        required = ('dataset', 'model', 'training_seed', 'generation_seed', 'child', 'count_source')
        if any(name not in case for name in required) or not str(case['count_source']).strip():
            raise ValueError('Every sample needs dataset/model/seeds/child and declared count source')
        child = Path(case['child'])
        if not child.is_file():
            if allow_partial:
                pending.append(case); continue
            raise FileNotFoundError(child)
        dataset = case['dataset']
        schema = payload['schemas'][dataset]
        raw = reference_root / dataset / 'raw'
        counts = Path(case['counts']) if case.get('counts') else None
        if counts is not None and not counts.is_file():
            raise FileNotFoundError(counts)
        child_digest = sha256(child)
        counts_digest = sha256(counts) if counts else None
        value = compare(raw/'account.csv', raw/'trans.csv', child, schema['numeric'], schema['categorical'], counts)
        metric_path = Path(case['metrics']) if case.get('metrics') else None
        if metric_path and not metric_path.is_file() and not allow_partial:
            raise FileNotFoundError(metric_path)
        score, score_link = None, 'not available'
        metric_digest = sha256(metric_path) if metric_path and metric_path.is_file() else None
        if metric_path and metric_path.is_file():
            receipt = json.loads(metric_path.read_text(encoding='utf-8'))
            scorer = receipt.get('scorer_sha256', receipt.get('evaluator_sha256'))
            if scorer != SCORER_SHA256:
                raise ValueError('Metric receipt is not from the pinned original evaluator')
            if receipt.get('child_sha256') and receipt['child_sha256'] != child_digest:
                raise ValueError('Metric receipt does not match generated child table')
            score = receipt.get('metrics')
            score_link = 'child digest verified' if receipt.get('child_sha256') else 'scorer verified; receipt has no direct child digest'
        if (sha256(child) != child_digest or (counts and sha256(counts) != counts_digest)
                or (metric_path is not None and metric_digest and sha256(metric_path) != metric_digest)):
            raise ValueError('A generated source changed while the report was being prepared')
        rows.append({**case, 'child_sha256': child_digest, 'counts_sha256': counts_digest,
                     'metrics_sha256': metric_digest,
                     'metric_link': score_link, 'metrics': score, 'distribution': value})
    train, overall = aggregates(rows, payload.get('expected_training_seeds', [1, 2, 3]),
                                payload.get('expected_generation_seeds', [1, 2, 3]))
    if sha256(cases_file) != case_manifest_digest or any(sha256(Path(p)) != h for p, h in source_files.items()):
        raise ValueError('The case manifest or original source data changed during reporting')
    result = {'scope': 'descriptive native paths; no generalization claim without an explicit held-out protocol',
              'source_case_manifest_sha256': case_manifest_digest, 'source_real_sha256': source_files,
              'source_code_sha256': sha256(Path(__file__)),
              'partial': bool(pending) or any(not item['complete'] for item in overall),
              'pending': pending, 'per_generation': rows, 'per_training_seed': train, 'overall': overall}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = run(args.reference_root, args.cases, args.output, allow_partial=args.allow_partial)
    print(json.dumps({'completed': len(result['per_generation']), 'pending': len(result['pending']),
                      'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
