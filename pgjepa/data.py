"""Leakage-safe mixed event data for the independent PG-JEPA experiment.

Times are days. Entity IDs and synthetic ground truth are grouping/audit fields only.
The synthetic process is fully specified here: a persistent AR(1) intensity factor,
periodic renewal gaps, a switching mark regime and lag-8/16 numeric feedback.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import gzip
from itertools import product
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch


def make_synthetic(n_entities: int = 2000, seed: int = 11,
                   min_events: int = 32, max_events: int = 128,
                   regularity: bool = False, burstiness: bool = True,
                   heterogeneity: bool = True, regime_shift: bool = True,
                   missingness: str = "mcar") -> list[dict]:
    """Generate a reproducible factorial DGP, never exposing its hidden labels as inputs.

    log gap = -.65*profile + .35*sin(2*pi*clock/14) + burst + .12*last mark
              + Normal(0,.35); burst = .92*burst + Normal(0,.28).
    First numeric mark also contains .18*mark[t-8] + .12*mark[t-16].
    """
    if n_entities < 1 or not 1 <= min_events <= max_events:
        raise ValueError("Require positive entity count and 1 <= min_events <= max_events")
    if missingness not in {"none", "mcar", "mnar"}:
        raise ValueError("missingness must be none, mcar, or mnar")
    rng = np.random.default_rng(seed)
    entities = []
    for i in range(n_entities):
        n = int(rng.integers(min_events, max_events + 1))
        group = int(rng.integers(3))
        profile = float((group - 1) * .8 + rng.normal(0, .15)) if heterogeneity else 0.
        regime = int(rng.integers(3))
        regimes = np.zeros(n, dtype=int)
        nums = np.empty((n, 3), dtype=np.float64)
        cats = np.empty((n, 2), dtype=object)
        delta = np.zeros(n, dtype=np.float64)
        clock, burst, dynamic, previous = 0., 0., 0., 0.
        for t in range(n):
            if regime_shift and rng.random() < .07:
                regime = (regime + int(rng.integers(1, 3))) % 3
            if not regime_shift:
                regime = 1
            regimes[t] = regime
            burst = .92 * burst + rng.normal(0, .28) if burstiness else 0.
            if t:
                if regularity:
                    delta[t] = np.exp(-.65 * profile)
                else:
                    loggap = (-.65 * profile + .35 * np.sin(2 * np.pi * clock / 14)
                              + burst + .12 * previous + rng.normal(0, .35))
                    delta[t] = np.exp(np.clip(loggap, -5, 5))
                    if rng.random() < .025:
                        delta[t] = 0.
                clock += delta[t]
            logits = np.array([-.7, .2, -.3]) + np.eye(3)[regime] * 1.8
            logits += np.array([-profile, 0., profile]) + .15 * dynamic
            probabilities = np.exp(logits - logits.max())
            mark = int(rng.choice(3, p=probabilities / probabilities.sum()))
            dynamic = np.tanh(.75 * dynamic + .35 * (mark - 1) + .2 * profile)
            lag = (.18 * nums[t - 8, 0] if t >= 8 else 0.)
            lag += .12 * nums[t - 16, 0] if t >= 16 else 0.
            nums[t, 0] = (.4 * previous + .6 * (mark - 1) + .3 * profile
                          + .5 * dynamic + lag + rng.normal(0, .35))
            nums[t, 1] = np.exp(.35 * nums[t, 0] + .15 * np.log1p(delta[t])
                                + rng.normal(0, .2))
            nums[t, 2] = float(rng.poisson(np.exp(.15 * profile + .2 * mark)))
            cats[t] = [str(mark), "high" if nums[t, 0] + rng.normal(0, .6) > .2 else "low"]
            previous = nums[t, 0]
        if missingness != "none":
            p = np.full(n, .04) if missingness == "mcar" else .02 + .14 / (1 + np.exp(-nums[:, 0]))
            nums[rng.random(nums.shape) < p[:, None]] = np.nan
            cats[rng.random(cats.shape) < p[:, None]] = None
        entities.append({"entity_id": f"S-{seed}-{i:06d}", "x_num": nums,
                         "x_cat": cats, "delta": delta, "attrs": {},
                         "numeric_names": ["signal", "amount", "count"],
                         "categorical_names": ["mark", "level"],
                         "metadata": {"group": group, "true_profile": profile,
                                      "regime": regimes.tolist(), "seed": seed}})
    return entities


def make_factorial(n_entities: int = 2000, seed: int = 11,
                   min_events: int = 32, max_events: int = 128,
                   missingness: str = "mcar") -> list[dict]:
    """Balance entities across the 2^4 DGP factor settings; flags stay in metadata.

    ``regularity=True`` deliberately dominates gap burstiness: the redundant
    regular+bursty cell is a negative control for the burst factor rather than a
    distinct generative distribution. Other factors still alter marks/profiles.
    """
    if n_entities < 16:
        raise ValueError("A complete 2^4 factorial needs at least 16 entities")
    flags = ("regularity", "burstiness", "heterogeneity", "regime_shift")
    entities = []
    for index, values in enumerate(product((False, True), repeat=4)):
        factors = dict(zip(flags, values))
        count = n_entities // 16 + int(index < n_entities % 16)
        block = make_synthetic(count, seed=seed + 1009 * index,
                               min_events=min_events, max_events=max_events,
                               missingness=missingness, **factors)
        for e in block:
            e["metadata"]["factors"] = factors.copy()
            e["metadata"]["factor_cell"] = index
        entities.extend(block)
    return entities


def split_entities(entities: list[dict], seed: int = 11) -> dict[str, list[dict]]:
    """Exact 70/15/remainder split after an ID-stable seeded permutation."""
    ids = [str(e["entity_id"]) for e in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate entity IDs")
    if len(ids) < 3:
        raise ValueError("At least three entities are needed for disjoint splits")
    ordered = sorted(entities, key=lambda e: str(e["entity_id"]))
    order = np.random.default_rng(seed).permutation(len(ordered))
    n_train = min(max(1, int(.7 * len(ids))), len(ids) - 2)
    n_val = min(max(1, int(.15 * len(ids))), len(ids) - n_train - 1)
    return {"train": [ordered[i] for i in order[:n_train]],
            "val": [ordered[i] for i in order[n_train:n_train + n_val]],
            "test": [ordered[i] for i in order[n_train + n_val:]]}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))


def _category(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


class Preprocessor:
    """Train-only signed-log scales, bounded category vocabularies and observed masks.

    All numeric columns use sign(x)*log1p(abs(x)), including signed refunds. This
    invertible transform has no fitted quantile clipping. Counts are not rounded on
    inverse: any output-domain correction belongs to the audited generation policy.
    """
    def __init__(self, max_categories: int = 128):
        if max_categories < 2:
            raise ValueError("max_categories must allow observed classes and UNK")
        self.max_categories = max_categories
        self.n_num = 0
        self.cat_sizes: list[int] = []
        self.attrs_dim = 0
        self.mean = np.empty(0)
        self.scale = np.empty(0)
        self.vocabularies: list[list[str]] = []
        self.attr_specs: list[dict] = []
        self.fitted = False

    def fit(self, train: list[dict]) -> "Preprocessor":
        if not train:
            raise ValueError("Cannot fit empty training data")
        numeric = np.concatenate([np.asarray(e["x_num"], dtype=float) for e in train])
        categorical = np.concatenate([np.asarray(e["x_cat"], dtype=object) for e in train])
        if numeric.ndim != 2 or categorical.ndim != 2:
            raise ValueError("Event arrays must have a feature dimension")
        self.n_num = numeric.shape[1]
        transformed = np.sign(numeric) * np.log1p(np.abs(numeric))
        self.mean = np.zeros(self.n_num)
        self.scale = np.ones(self.n_num)
        for j in range(self.n_num):
            values = transformed[:, j][np.isfinite(transformed[:, j])]
            if len(values):
                self.mean[j] = values.mean()
                self.scale[j] = max(float(values.std()), 1e-4)
        self.vocabularies = []
        for j in range(categorical.shape[1]):
            counts = Counter(_category(v) for v in categorical[:, j] if not _missing(v))
            values = sorted(counts, key=lambda v: (-counts[v], v))[:self.max_categories - 1]
            self.vocabularies.append(values)
        self.cat_sizes = [len(v) + 1 for v in self.vocabularies]
        first_attrs = train[0].get("attrs", {})
        self.attr_specs = []
        if isinstance(first_attrs, dict):
            for key in sorted(first_attrs):
                vals = [e.get("attrs", {}).get(key) for e in train]
                observed = [v for v in vals if not _missing(v)]
                if observed and all(isinstance(v, (int, float, np.number)) for v in observed):
                    values = np.asarray(observed, dtype=float)
                    self.attr_specs.append({"name": key, "kind": "numeric",
                                            "mean": float(values.mean()),
                                            "scale": max(float(values.std()), 1e-4)})
                else:
                    vocab = sorted({_category(v) for v in observed})[:self.max_categories - 1]
                    self.attr_specs.append({"name": key, "kind": "categorical", "vocab": vocab})
        else:
            arrays = [np.asarray(e.get("attrs", []), dtype=float).reshape(-1) for e in train]
            width = arrays[0].shape[0]
            if any(len(a) != width for a in arrays):
                raise ValueError("Static attribute widths differ")
            for j in range(width):
                values = np.asarray([a[j] for a in arrays])
                values = values[np.isfinite(values)]
                self.attr_specs.append({"name": j, "kind": "numeric",
                                        "mean": float(values.mean()) if len(values) else 0.,
                                        "scale": max(float(values.std()), 1e-4) if len(values) else 1.})
        self.attrs_dim = sum(1 if s["kind"] == "numeric" else len(s["vocab"]) + 1
                             for s in self.attr_specs)
        self.numeric_names = train[0].get("numeric_names", [f"numeric_{j}" for j in range(self.n_num)])
        self.categorical_names = train[0].get("categorical_names", [f"category_{j}" for j in range(len(self.cat_sizes))])
        self.fitted = True
        return self

    def transform(self, entity: dict) -> dict[str, torch.Tensor]:
        if not self.fitted:
            raise ValueError("Fit preprocessing on training entities first")
        numeric = np.asarray(entity["x_num"], dtype=float)
        cat = np.asarray(entity["x_cat"], dtype=object)
        delta = np.asarray(entity["delta"], dtype=float).reshape(-1, 1)
        if numeric.shape != (len(delta), self.n_num) or cat.shape != (len(delta), len(self.cat_sizes)):
            raise ValueError("Entity event shape disagrees with fitted schema")
        if not np.isfinite(delta).all() or (delta < 0).any():
            raise ValueError("Gaps must be finite and nonnegative")
        observed_num = np.isfinite(numeric)
        transformed = (np.sign(numeric) * np.log1p(np.abs(numeric)) - self.mean) / self.scale
        transformed = np.where(observed_num, transformed, 0.)
        encoded = np.zeros(cat.shape, dtype=np.int64)
        observed_cat = np.zeros(cat.shape, dtype=bool)
        for j, vocab in enumerate(self.vocabularies):
            mapping = {v: k + 1 for k, v in enumerate(vocab)}
            for t, value in enumerate(cat[:, j]):
                if not _missing(value):
                    encoded[t, j] = mapping.get(_category(value), 0)
                    observed_cat[t, j] = True
        attrs: list[float] = []
        raw_attrs = entity.get("attrs", {})
        for spec in self.attr_specs:
            value = raw_attrs.get(spec["name"]) if isinstance(raw_attrs, dict) else raw_attrs[spec["name"]]
            if spec["kind"] == "numeric":
                attrs.append(0. if _missing(value) else (float(value) - spec["mean"]) / spec["scale"])
            else:
                onehot = [0.] * (len(spec["vocab"]) + 1)
                index = spec["vocab"].index(_category(value)) + 1 if not _missing(value) and _category(value) in spec["vocab"] else 0
                onehot[index] = 1.
                attrs.extend(onehot)
        return {"x_num": torch.tensor(transformed, dtype=torch.float32),
                "x_cat": torch.tensor(encoded, dtype=torch.long),
                "mask": torch.tensor(np.concatenate([observed_num, observed_cat], axis=-1), dtype=torch.float32),
                "delta": torch.tensor(delta, dtype=torch.float32),
                "t_valid": torch.ones(len(delta), dtype=torch.bool),
                "attrs": torch.tensor(attrs, dtype=torch.float32)}

    def inverse_numeric(self, values: Any) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        transformed = np.asarray(values, dtype=float) * self.scale + self.mean
        return np.sign(transformed) * np.expm1(np.abs(transformed))

    def state_dict(self) -> dict:
        if not self.fitted:
            raise ValueError("Unfitted preprocessor")
        return {"max_categories": self.max_categories, "n_num": self.n_num,
                "cat_sizes": self.cat_sizes, "attrs_dim": self.attrs_dim,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "vocabularies": self.vocabularies, "attr_specs": self.attr_specs,
                "numeric_names": self.numeric_names, "categorical_names": self.categorical_names}

    def load_state_dict(self, state: dict) -> "Preprocessor":
        for key, value in state.items():
            setattr(self, key, value)
        self.mean = np.asarray(state["mean"], dtype=float)
        self.scale = np.asarray(state["scale"], dtype=float)
        self.fitted = True
        return self

    @classmethod
    def from_state_dict(cls, state: dict) -> "Preprocessor":
        return cls(state["max_categories"]).load_state_dict(state)


def _stack_windows(windows: list[dict], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.stack([w[key] for w in windows]).to(device)
            for key in ("x_num", "x_cat", "mask", "delta", "t_valid", "attrs")}


def sample_batch(encoded_entities: list[dict], batch_size: int,
                 prefix_range: tuple[int, int] = (8, 32), future: int = 16,
                 rng: np.random.Generator | None = None,
                 device: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], int]:
    """Sample full contiguous windows; no padded future is treated as a target."""
    rng = rng or np.random.default_rng()
    low, high = prefix_range
    if batch_size < 1 or not 1 <= low <= high or future < 1:
        raise ValueError("Invalid batch, prefix or future length")
    eligible = [e for e in encoded_entities if len(e["delta"]) >= low + future]
    if not eligible:
        raise ValueError("No entities support the requested prefix and future")
    max_cut = min(high, max(len(e["delta"]) - future for e in eligible))
    cut = int(rng.integers(low, max_cut + 1))
    eligible = [e for e in eligible if len(e["delta"]) >= cut + future]
    windows = []
    for index in rng.integers(len(eligible), size=batch_size):
        entity = eligible[int(index)]
        start = int(rng.integers(len(entity["delta"]) - cut - future + 1))
        window = {k: v[start:start + cut + future].clone() if k != "attrs" else v.clone()
                  for k, v in entity.items() if isinstance(v, torch.Tensor)}
        window["delta"][0] = 0.
        windows.append(window)
    return _stack_windows(windows, device), cut


def stack_prefixes(encoded_entities: list[dict], prefix_len: int,
                   device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    if not encoded_entities or prefix_len < 1:
        raise ValueError("Need entities and a positive prefix")
    if any(len(e["delta"]) < prefix_len for e in encoded_entities):
        raise ValueError("An entity has fewer events than the requested prefix")
    windows = [{k: v[:prefix_len] if k != "attrs" else v
                for k, v in e.items() if isinstance(v, torch.Tensor)} for e in encoded_entities]
    return _stack_windows(windows, device)


def _retail_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
    else:
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                iterator = sheet.iter_rows(values_only=True)
                header = [str(h) for h in next(iterator)]
                for row in iterator:
                    yield dict(zip(header, row))
        finally:
            workbook.close()


def _date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    value = str(value)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    raise ValueError(f"Unsupported timestamp: {value}")


def read_retail(path: str | Path) -> tuple[list[dict], dict]:
    """Invoice aggregates retain negative quantities/spend and cancellation events.

    Customer and invoice keys are excluded from model features. Country is the
    first chronological invoice's country, known at any nonempty prefix.
    """
    path = Path(path)
    invoices: dict[tuple, dict] = {}
    total, missing_id, invalid, return_lines = 0, 0, 0, 0
    for row in _retail_rows(path):
        total += 1
        customer = row.get("CustomerID", row.get("Customer ID"))
        if _missing(customer) or str(customer).strip() == "":
            missing_id += 1
            continue
        try:
            invoice = str(row.get("InvoiceNo", row.get("Invoice"))).strip()
            timestamp = _date(row.get("InvoiceDate"))
            quantity = float(row["Quantity"])
            price = float(row.get("UnitPrice", row.get("Price")))
            if not invoice or not np.isfinite([quantity, price]).all() or price < 0:
                raise ValueError("Invalid invoice values")
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        customer = _category(customer)
        key = (customer, invoice, timestamp)
        bucket = invoices.setdefault(key, {"quantity": 0., "spend": 0., "products": Counter(),
                                          "cancel": False, "country": str(row.get("Country", "UNKNOWN"))})
        bucket["quantity"] += quantity
        bucket["spend"] += quantity * price
        bucket["products"][str(row.get("StockCode", "UNKNOWN"))] += abs(quantity)
        cancellation = invoice.upper().startswith("C") or quantity < 0
        bucket["cancel"] |= cancellation
        return_lines += int(cancellation)
    grouped: dict[str, list] = {}
    for (customer, invoice, timestamp), record in invoices.items():
        grouped.setdefault(customer, []).append((timestamp, invoice, record))
    entities = []
    zero_gaps = 0
    for customer in sorted(grouped):
        records = sorted(grouped[customer], key=lambda r: (r[0], r[1]))
        numeric, categorical, delta = [], [], []
        for j, (timestamp, _, record) in enumerate(records):
            gap = (timestamp - records[j - 1][0]).total_seconds() / 86400 if j else 0.
            zero_gaps += int(j > 0 and gap == 0)
            delta.append(gap)
            numeric.append([record["spend"], record["quantity"], len(record["products"])])
            dominant = sorted(record["products"], key=lambda p: (-record["products"][p], p))[0]
            categorical.append(["return" if record["cancel"] else "sale", dominant])
        entities.append({"entity_id": customer, "x_num": np.asarray(numeric, dtype=float),
                         "x_cat": np.asarray(categorical, dtype=object), "delta": np.asarray(delta),
                         "attrs": {"country": records[0][2]["country"]},
                         "numeric_names": ["spend", "quantity", "distinct_products"],
                         "categorical_names": ["invoice_type", "dominant_product"],
                         "metadata": {"source": str(path), "event_unit": "invoice"}})
    lengths = np.asarray([len(e["delta"]) for e in entities])
    audit = {"source": str(path), "dataset": "Online Retail (I or II per supplied source)",
             "event_unit": "CustomerID/InvoiceNo/InvoiceDate", "time_unit": "days",
             "rows": total, "missing_customer_rows": missing_id,
             "missing_customer_fraction": missing_id / max(total, 1), "invalid_rows": invalid,
             "return_lines_retained": return_lines, "entities": len(entities),
             "invoice_events": len(invoices), "simultaneous_invoice_gaps": zero_gaps,
             "static_country_policy": "first chronological invoice, never full-history mode",
             "eligibility_P16": {str(k): int((lengths >= 16 + k).sum()) for k in (8, 16, 32, 64)}}
    return entities, audit


def read_bpi(path: str | Path) -> tuple[list[dict], dict]:
    """Read standard (optionally gzip) XES; static amount requested only, no outcome."""
    path = Path(path)
    entities: list[dict] = []
    excluded = 0
    with (gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")) as stream:
        for _, trace in ET.iterparse(stream, events=("end",)):
            if trace.tag.rsplit("}", 1)[-1] != "trace":
                continue
            static: dict = {}
            events: list[tuple] = []
            for node in trace:
                if node.tag.rsplit("}", 1)[-1] == "event":
                    fields = {child.get("key"): child.get("value") for child in node}
                    try:
                        timestamp = _date(fields["time:timestamp"])
                    except (KeyError, TypeError, ValueError):
                        excluded += 1
                        continue
                    events.append((timestamp, len(events), fields))
                else:
                    static[node.get("key")] = node.get("value")
            if events:
                events.sort(key=lambda e: (e[0], e[1]))
                delta = [0.] + [(events[j][0] - events[j - 1][0]).total_seconds() / 86400
                                for j in range(1, len(events))]
                categories = [[e[2].get("concept:name"), e[2].get("lifecycle:transition"),
                               e[2].get("org:resource")] for e in events]
                amount = static.get("AMOUNT_REQ", static.get("amount", "0"))
                try:
                    requested = float(amount)
                except (ValueError, TypeError):
                    requested = 0.
                entities.append({"entity_id": str(static.get("concept:name", len(entities))),
                                 "x_num": np.empty((len(events), 0)),
                                 "x_cat": np.asarray(categories, dtype=object),
                                 "delta": np.asarray(delta), "attrs": {"requested_amount": requested},
                                 "numeric_names": [], "categorical_names": ["activity", "lifecycle", "resource"],
                                 "metadata": {"source": str(path), "event_unit": "case_event"}})
            trace.clear()
    return entities, {"source": str(path), "dataset": "BPI Challenge 2012 XES",
                      "entities": len(entities), "invalid_timestamp_events": excluded,
                      "static_whitelist": ["AMOUNT_REQ"], "time_unit": "days"}
