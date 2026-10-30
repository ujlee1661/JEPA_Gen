"""Observation-space suffix evaluation using the pinned tab_gen scoring code.

Real tensors are [entity,event,feature]; generated tensors additionally have a
replicate axis [entity,replicate,event,feature]. Times are elapsed days from the
prefix endpoint. Reference scores intentionally retain their daily bucketing.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


@lru_cache(maxsize=1)
def reference_module():
    path = Path(__file__).resolve().parents[1] / "vendor" / "tab_gen" / "score.py"
    spec = importlib.util.spec_from_file_location("pgjepa_tab_gen_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _arrays(data: dict, generated: bool = False) -> dict:
    result = {name: _numpy(data[name]).copy()
              for name in ("x_num", "x_cat", "mask", "delta")}
    ndim = result["x_num"].ndim
    if generated and ndim == 3:
        result = {name: value[:, None] for name, value in result.items()}
    expected = 4 if generated else 3
    if result["x_num"].ndim != expected:
        raise ValueError("Expected [E,K,F] real or [E,S,K,F] generated arrays")
    leading = result["x_num"].shape[:-1]
    if any(value.shape[:-1] != leading for value in result.values()):
        raise ValueError("All suffix arrays must share entity/replicate/event axes")
    if result["delta"].shape[-1] != 1:
        raise ValueError("delta must end in a singleton feature dimension")
    if np.any(~np.isfinite(result["delta"])) or np.any(result["delta"] < 0):
        raise ValueError("Suffix gaps must be finite and nonnegative")
    n, c = result["x_num"].shape[-1], result["x_cat"].shape[-1]
    if result["mask"].shape[-1] != n + c:
        raise ValueError("mask must cover numerical and categorical features")
    result["x_num"] = result["x_num"].astype(float)
    result["x_num"][result["mask"][..., :n] < .5] = np.nan
    return result


def _pair(real, generated):
    real, generated = _arrays(real), _arrays(generated, True)
    if (real["x_num"].shape[0] != generated["x_num"].shape[0]
            or real["x_num"].shape[1:] != generated["x_num"].shape[2:]):
        raise ValueError("Real and synthetic suffixes must have matched entities and lengths")
    if real["x_cat"].shape[-1] != generated["x_cat"].shape[-1]:
        raise ValueError("Categorical schemas differ")
    return real, generated


def export_frames(real: dict, generated: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert matched suffixes to the original scorer's two table schemas."""
    real, generated = _pair(real, generated)

    def frame(data, synthetic):
        shape = data["x_num"].shape
        n, c, k = shape[-1], data["x_cat"].shape[-1], shape[-2]
        if synthetic:
            ids = np.repeat([f"e{i}:r{s}" for i in range(shape[0])
                             for s in range(shape[1])], k)
        else:
            ids = np.repeat([f"e{i}" for i in range(shape[0])], k)
        columns = {"entity" if synthetic else "account_id": ids,
                   "t_offset": np.cumsum(data["delta"][..., 0], axis=-1).reshape(-1)}
        for j in range(n):
            columns[f"x_num_{j}"] = data["x_num"][..., j].reshape(-1)
        for j in range(c):
            values = data["x_cat"][..., j].astype(object).reshape(-1)
            values[data["mask"][..., n + j].reshape(-1) < .5] = np.nan
            columns[f"x_cat_{j}"] = values
        return pd.DataFrame(columns)

    return frame(real, False), frame(generated, True)


def _mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[valid], np.asarray(b)[valid]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _lag_correlation(values, lag):
    if values.shape[-2] <= lag + 2:
        return np.nan
    return _corr(values[..., :-lag, :].reshape(-1),
                 values[..., lag:, :].reshape(-1))


def _transition(data, feature, lag=1):
    cats = data["x_cat"][..., feature]
    observed = data["mask"][..., data["x_num"].shape[-1] + feature] >= .5
    valid = observed[..., :-lag] & observed[..., lag:]
    pairs = np.stack((cats[..., :-lag][valid], cats[..., lag:][valid]), axis=-1)
    if not len(pairs):
        return {}
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return {tuple(pair): count / counts.sum() for pair, count in zip(unique, counts)}


def _transition_error(real, generated, lag=1):
    errors = []
    for j in range(real["x_cat"].shape[-1]):
        rp, sp = _transition(real, j, lag), _transition(generated, j, lag)
        if rp and sp:
            errors.append(.5 * sum(abs(rp.get(v, 0) - sp.get(v, 0)) for v in rp.keys() | sp.keys()))
    return _mean(errors)


def _distances(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"ks": None, "wasserstein": None}
    return {"ks": float(ks_2samp(a, b).statistic),
            "wasserstein": float(wasserstein_distance(a, b))}


def _utility(real, generated):
    """Descriptive TSTR/TRTR with an internal entity-disjoint calibration split."""
    if real["x_num"].shape[0] < 8 or real["x_num"].shape[-1] == 0:
        return {"status": "insufficient_entities"}
    from sklearn.linear_model import Ridge
    split = max(4, real["x_num"].shape[0] // 2)
    calibration = np.nan_to_num(real["x_num"][:split])
    scale = np.std(calibration.reshape(-1, calibration.shape[-1]), axis=0).clip(1e-6)
    center = np.mean(calibration.reshape(-1, calibration.shape[-1]), axis=0)

    def xy(data, start, end):
        x = (data["x_num"][start:end] - center) / scale
        features = np.concatenate((np.nan_to_num(x[..., :-1, :]),
                                   np.isfinite(x[..., :-1, :]).astype(float),
                                   np.log1p(data["delta"][start:end, ..., :-1, :])), axis=-1)
        target = x[..., 1:, 0]
        valid = np.isfinite(target)
        return features[valid], target[valid]

    sx, sy = xy(generated, 0, split)
    rx, ry = xy(real, 0, split)
    tx, ty = xy(real, split, real["x_num"].shape[0])
    if min(len(sx), len(rx), len(tx)) < 4:
        return {"status": "insufficient_observed_targets"}
    syn_pred = Ridge(alpha=1).fit(sx, sy).predict(tx)
    real_pred = Ridge(alpha=1).fit(rx, ry).predict(tx)
    return {"status": "ok", "tstr_standardized_mae": float(np.mean(abs(syn_pred - ty))),
            "trtr_standardized_mae": float(np.mean(abs(real_pred - ty))),
            "test_events": len(tx), "target": "next_event_x_num_0",
            "protocol": "diagnostic_internal_entity_disjoint_half_split; not external benchmark utility"}


def _clock_counts(gaps, window_days=1.0):
    """First-window counts only for suffixes observed through that window.

    A fixed-K suffix ending before the boundary is censored for this statistic;
    its unobserved remainder must not be counted as an event-free interval.
    """
    times = np.cumsum(gaps, axis=-1)
    supported = times[..., -1] >= window_days
    counts = ((times >= 0) & (times < window_days)).sum(axis=-1)[supported]
    mean = float(counts.mean()) if len(counts) else 0.0
    return counts, {"supported_sequences": int(supported.sum()),
                    "total_sequences": int(supported.size), "window_days": window_days,
                    "fano": float(counts.var() / mean) if mean > 0 else None}


def entity_energy_scores(real: dict, generated: dict, numeric_scale=None) -> np.ndarray:
    """Conditional Energy Score on a fixed marginal/dynamics/time summary.

    This proper score concerns the explicit summary distribution, not the full
    sequence law. Numeric scales default to pooled real evaluation std; they
    never affect training. Replicates remain grouped under their source entity.
    """
    real, generated = _pair(real, generated)
    n = real["x_num"].shape[-1]
    if numeric_scale is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numeric_scale = np.nanstd(real["x_num"].reshape(-1, n), axis=0) if n else np.empty(0)
    scale = np.nan_to_num(np.asarray(numeric_scale, float), nan=1).clip(1e-6)

    def summary(data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(data["x_num"], axis=-2) / scale
            std = np.nanstd(data["x_num"], axis=-2) / scale
        duration = np.log1p(data["delta"].sum(axis=-2))
        missing = 1 - data["mask"].mean(axis=-2)
        parts = [mean, std, duration, missing]
        if data["x_num"].shape[-2] >= 4:
            quarters = np.array_split(data["x_num"], 4, axis=-2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = [np.nanmean(block, axis=-2) / scale for block in quarters]
            parts.extend(means[j + 1] - means[j] for j in range(3))
        for lag in (1, 4, 8, 16, 32):
            if data["x_num"].shape[-2] <= lag + 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                left, right = data["x_num"][..., :-lag, :], data["x_num"][..., lag:, :]
                valid = np.isfinite(left) & np.isfinite(right)
                left, right = np.where(valid, left, np.nan), np.where(valid, right, np.nan)
                left = left - np.nanmean(left, axis=-2, keepdims=True)
                right = right - np.nanmean(right, axis=-2, keepdims=True)
                den = np.sqrt(np.nanmean(left ** 2, axis=-2) * np.nanmean(right ** 2, axis=-2))
                corr = np.nanmean(left * right, axis=-2) / np.maximum(den, 1e-8)
            parts.append(corr)
            nnum = data["x_num"].shape[-1]
            observed = data["mask"][..., nnum:] >= .5
            keep = observed[..., :-lag, :] & observed[..., lag:, :]
            equal = data["x_cat"][..., :-lag, :] == data["x_cat"][..., lag:, :]
            parts.append((equal & keep).sum(axis=-2) / np.maximum(keep.sum(axis=-2), 1))
        return np.nan_to_num(np.concatenate(parts, axis=-1))

    r, s = summary(real), summary(generated)
    scores = np.linalg.norm(s - r[:, None], axis=-1).mean(axis=1)
    reps = s.shape[1]
    if reps > 1:
        for i in range(len(scores)):
            scores[i] -= cdist(s[i], s[i]).sum() / (2 * reps * (reps - 1))
    return scores


def bootstrap_compare(real: dict, gen_a: dict, gen_b: dict, n_boot=200, seed=11) -> dict:
    """Paired entity bootstrap of EnergyScore(B)-EnergyScore(A); negative favors B."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    a, b = entity_energy_scores(real, gen_a), entity_energy_scores(real, gen_b)
    delta = b - a
    if len(delta) < 2:
        raise ValueError("At least two matched entities are required")
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    return {"metric": "conditional_summary_energy_score", "delta_b_minus_a": float(delta.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "entities": len(delta),
            "bootstrap_resamples": n_boot, "seed": seed, "lower_is_better": True,
            "score_a": float(a.mean()), "score_b": float(b.mean()),
            "summary": "numeric means/std and consecutive quarter-block mean changes (pooled real std scaled), numeric lag correlations and categorical repeat rates at supported lags 1/4/8/16/32, log1p suffix duration, missing rates"}


def evaluate(real: dict, generated: dict, dataset="pg_synthetic", train: dict | None = None) -> dict:
    real, generated = _pair(real, generated)
    r, s = export_frames(real, generated)
    ref = reference_module()
    num = [f"x_num_{j}" for j in range(real["x_num"].shape[-1])]
    cat = [f"x_cat_{j}" for j in range(real["x_cat"].shape[-1])]
    ref.VAL[dataset] = (num, cat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = {"ts": ref.ts(r, s), "cs": ref.cs(dataset, r, s),
                  "sd": ref.sd(dataset, r, s), "lg": ref.lg(dataset, r, s)}
        cs_detail = ref.cs(dataset, r, s, full=True)
        sd_detail = ref.sd(dataset, r, s, full=True)
    num_dist = {col: _distances(r[col], s[col]) for col in num}
    cat_dist: dict[str, dict[str, float | None]] = {}
    for col in cat:
        rp, sp = r[col].dropna().value_counts(normalize=True), s[col].dropna().value_counts(normalize=True)
        support = rp.index.union(sp.index)
        if len(rp) and len(sp):
            p, q = rp.reindex(support, fill_value=0), sp.reindex(support, fill_value=0)
            cat_dist[col] = {"jsd": float(jensenshannon(p, q, base=2) ** 2),
                             "tvd": float(abs(p - q).sum() / 2)}
        else:
            cat_dist[col] = {"jsd": None, "tvd": None}
    corr_errors = []
    for i, col in enumerate(num):
        for other in num[i + 1:]:
            corr_errors.append(abs(_corr(r[col].values, r[other].values)
                                   - _corr(s[col].values, s[other].values)))
    lags = {}
    for lag in (1, 4, 8, 16, 32, 64):
        errors = []
        for j in range(len(num)):
            errors.append(abs(_lag_correlation(real["x_num"][..., j:j + 1], lag)
                              - _lag_correlation(generated["x_num"][..., j:j + 1], lag)))
        lags[str(lag)] = {"numeric_correlation_error": _mean(errors),
                          "event_transition_joint_tvd": _transition_error(real, generated, lag)
                          if real["x_num"].shape[1] > lag else None}
    rd, sd = real["delta"][..., 0], generated["delta"][..., 0]
    rcv = np.std(rd, axis=-1) / np.maximum(np.mean(rd, axis=-1), 1e-8)
    scv = np.std(sd, axis=-1) / np.maximum(np.mean(sd, axis=-1), 1e-8)
    coupling = {}
    for j in range(len(num)):
        values = []
        for data in (real, generated):
            gap, x = data["delta"][..., 0].reshape(-1), data["x_num"][..., j].reshape(-1)
            keep = np.isfinite(x)
            values.append(float(spearmanr(gap[keep], x[keep]).statistic)
                          if keep.sum() > 2 and np.std(gap[keep]) > 0 and np.std(x[keep]) > 0 else np.nan)
        coupling[str(j)] = {"real_spearman": values[0], "generated_spearman": values[1],
                            "absolute_error": abs(values[0] - values[1])}
    real_days, syn_days = set(np.round(r.t_offset)), set(np.round(s.t_offset))
    rc, rc_support = _clock_counts(rd)
    sc, sc_support = _clock_counts(sd)
    result = {"reference": scores, **scores, "reference_breakdown": {"cs": cs_detail, "sd": sd_detail},
              "numeric": num_dist, "categorical": cat_dist, "pairwise_correlation_error": _mean(corr_errors),
              "event_order_dynamics": lags, "time_mark_coupling": coupling,
              "timing": {"gap": _distances(rd, sd), "suffix_duration": _distances(rd.sum(-1), sd.sum(-1)),
                         "zero_rate_error": float(abs((rd == 0).mean() - (sd == 0).mean())),
                         "entity_gap_cv": _distances(rcv, scv),
                         "count_first_day": {"distance": _distances(rc, sc),
                                             "real": rc_support, "generated": sc_support},
                         "gap_lag1_error": abs(_lag_correlation(rd[..., None], 1) - _lag_correlation(sd[..., None], 1))},
              "missing_rate_mae": float(abs(real["mask"].mean((0, 1)) - generated["mask"].mean((0, 1, 2))).mean()),
              "primary_energy_score": float(entity_energy_scores(real, generated).mean()),
              "utility": _utility(real, generated),
              "coverage": {"entities": len(real["x_num"]), "replicates": generated["x_num"].shape[1],
                           "suffix_events": real["x_num"].shape[1],
                           "observed_numeric_nonfinite_fraction": float(
                               ((real["mask"][..., :len(num)] >= .5) & ~np.isfinite(real["x_num"])).sum()
                               / max((real["mask"][..., :len(num)] >= .5).sum(), 1)),
                           "generated_numeric_nonfinite_fraction": float(
                               ((generated["mask"][..., :len(num)] >= .5) & ~np.isfinite(generated["x_num"])).sum()
                               / max((generated["mask"][..., :len(num)] >= .5).sum(), 1)),
                           "real_day_buckets": len(real_days), "overlap_day_buckets": len(real_days & syn_days),
                           "cs_real_day_overlap_fraction": len(real_days & syn_days) / max(len(real_days), 1),
                           "finite_reference_axes": [k for k, v in scores.items() if np.isfinite(v)]},
              "protocol": {"reference_commit": "3cd17110f0f5fe7ea25377a8e2db6a5840b622b5",
                           "time_unit": "day", "time_origin": "observed_prefix_end", "fixed_k": True,
                           "jointff": "omitted: fixed event count makes count-axis evaluation uninformative",
                           "reference_ts": "active-span excludes prefix-to-first-event gap; timing.suffix_duration includes it",
                           "null": "unsupported or undefined metric; never imputed as perfect fidelity"}}
    if train is not None and len(num):
        from sklearn.neighbors import NearestNeighbors
        tr = _arrays(train)["x_num"].reshape(-1, len(num))
        sy = generated["x_num"].reshape(-1, len(num))
        tr, sy = tr[np.isfinite(tr).all(1)], sy[np.isfinite(sy).all(1)]
        if len(tr) and len(sy):
            scale = np.std(tr, axis=0).clip(1e-6)
            distances = NearestNeighbors(n_neighbors=1).fit(tr / scale).kneighbors(sy[:10000] / scale)[0]
            result["memorization"] = {"numeric_exact_train_row_fraction": float((distances < 1e-8).mean()),
                                      "median_standardized_train_row_distance": float(np.median(distances)),
                                      "sampled_rows": len(distances), "scope": "row diagnostic; not a privacy guarantee"}
    return _clean(result)
