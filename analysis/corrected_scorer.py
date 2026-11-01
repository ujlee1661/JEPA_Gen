"""Evaluate with the pinned paper scorer and corrected categorical CS identity.

The underlying scorer and its source data are left unchanged.  This module
replaces only the per-day categorical TV helper, preserving true string
categories while treating numeric values such as 0 and 0.0 as equal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

SCORER_SHA256 = 'f254228d9917dfed8fefb1beb057cbf87daad041622c1f021b2d4e4ab67b8dd3'
SCORER_ENV = 'JETS_REFERENCE_SCORER'


def default_scorer():
    """Locate the pinned reference, or ask for an explicit path.

    The environment variable supports arbitrary installations.  A local
    reference checkout and one-level home directories are also considered for
    compatibility with existing experiment runners.  Automatic discovery is
    accepted only when precisely one candidate has the pinned contents.
    """
    configured = os.environ.get(SCORER_ENV)
    if configured:
        return Path(configured)
    candidates = {Path(__file__).resolve().parents[1] / 'reference/eval/score.py'}
    candidates.update(Path.home().glob('*/reference/eval/score.py'))
    matches = []
    for candidate in candidates:
        try:
            if hashlib.sha256(candidate.read_bytes()).hexdigest() == SCORER_SHA256:
                matches.append(candidate.resolve())
        except OSError:
            continue
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected one pinned reference scorer; pass --scorer or set {SCORER_ENV}'
        )
    return matches[0]


def category_key(value):
    """Canonicalize numeric scalars without merging distinct string labels.

    Non-numeric labels retain their type and spelling; e.g. strings '0', '00'
    and numeric 0 remain distinct. Missing values are dropped by the TV helper
    exactly as they are in the original scorer.
    """
    if pd.isna(value):
        return ('missing', '')
    if isinstance(value, (bool, np.bool_)):
        return ('bool', str(bool(value)))
    if isinstance(value, Integral):
        return ('number', int(value), 1)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            numerator, denominator = number.as_integer_ratio()
            return ('number', numerator, denominator)
        return ('number-special', str(number))
    return ('label', type(value).__name__, str(value))


def categorical_tv(real_grouped, synth_grouped, times, cat_col):
    """Mean per-shared-day TV complement, matching the original aggregation."""
    scores = []
    for t in times:
        if t not in real_grouped.groups or t not in synth_grouped.groups:
            continue
        if cat_col not in real_grouped.obj.columns or cat_col not in synth_grouped.obj.columns:
            continue
        real = real_grouped.get_group(t)[cat_col].dropna()
        synth = synth_grouped.get_group(t)[cat_col].dropna()
        if len(real) == 0 or len(synth) == 0:
            continue
        real_counts = pd.Series([category_key(v) for v in real], dtype=object).value_counts(normalize=True)
        synth_counts = pd.Series([category_key(v) for v in synth], dtype=object).value_counts(normalize=True)
        categories = set(real_counts.index) | set(synth_counts.index)
        tvd = .5 * sum(abs(real_counts.get(c, 0.) - synth_counts.get(c, 0.)) for c in categories)
        scores.append(1. - tvd)
    return float(np.mean(scores)) if scores else np.nan


def load_scorer(path=None):
    path = Path(path if path is not None else default_scorer()).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCORER_SHA256:
        raise ValueError(f'Unexpected scorer SHA-256: {digest}')
    spec = importlib.util.spec_from_file_location('_jets_pinned_paper_scorer', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load scorer {path}')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer._cs_tv_complement = categorical_tv
    return scorer


def evaluate_paths(scorer, dataset, real_path, child_path, *, id_dtype='infer'):
    """Score explicitly selected real and synthetic event tables.

    IDs can be read as strings to preserve identifiers such as leading zeros.
    This never reads the external scorer's default full real population.
    """
    if id_dtype not in ('infer', 'string'):
        raise ValueError('id_dtype must be infer or string')

    def read_table(path, column):
        options = {'converters': {column: str}} if id_dtype == 'string' else {}
        frame = pd.read_csv(path, **options)
        if column not in frame.columns:
            raise ValueError(f'Missing required identifier column: {column}')
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Invalid empty identifier in column: {column}')
        return frame

    real = read_table(real_path, 'account_id')
    synthetic = read_table(child_path, 'entity')
    metrics = {'ts': scorer.ts(real, synthetic),
               'cs': scorer.cs(dataset, real, synthetic),
               'sd': scorer.sd(dataset, real, synthetic),
               'lg': scorer.lg(dataset, real, synthetic),
               'joint_ff': scorer.jointff(real, synthetic)}
    return {key: float(value) if np.isfinite(value) else None for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('child', type=Path)
    parser.add_argument('--real', type=Path, required=True,
                        help='Explicit real comparison table, for example the held-out test events')
    parser.add_argument('--scorer', type=Path, help=f'Pinned scorer path (or set {SCORER_ENV})')
    parser.add_argument('--id-dtype', choices=('infer', 'string'), default='infer',
                        help='Read real and generated IDs as strings when identifiers require it')
    args = parser.parse_args()
    result = evaluate_paths(load_scorer(args.scorer), args.dataset, args.real, args.child,
                            id_dtype=args.id_dtype)
    print(json.dumps({'metrics': result, 'original_scorer_sha256': SCORER_SHA256,
                      'real_sha256': hashlib.sha256(args.real.read_bytes()).hexdigest(),
                      'child_sha256': hashlib.sha256(args.child.read_bytes()).hexdigest(),
                      'correction': 'cs-categorical-numeric-identity-v1'}, allow_nan=True))


if __name__ == '__main__':
    main()
