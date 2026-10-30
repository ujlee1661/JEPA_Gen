"""Train-only mixed event preprocessing primitives."""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np
import torch

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
