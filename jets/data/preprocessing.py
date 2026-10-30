"""Train-only TabSyn-style Gaussian quantiles with explicit missing-value masks."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import QuantileTransformer
import torch

from .base import Preprocessor


class QuantilePreprocessor(Preprocessor):
    """Keep the existing categorical/static schema and replace numeric scaling.

    Each column is fitted on its observed training values only. Missing numeric
    inputs use zero in transformed space and retain a false observation mask.
    Inverse quantiles saturate at the observed training support, as in TabSyn;
    this is different from clamping negative raw values to zero.
    """

    def __init__(self, max_categories: int = 128, seed: int = 0,
                 quantile_grid: str = 'legacy_1000'):
        super().__init__(max_categories)
        if quantile_grid not in {'legacy_1000', 'all_observed'}:
            raise ValueError('quantile_grid must be legacy_1000 or all_observed')
        self.seed = seed
        self.quantile_grid = quantile_grid
        self.quantile_models: list[QuantileTransformer] = []
        self.observed_counts: list[int] = []
        self.integer_features: list[int] = []
        self.last_inverse_audit: dict[str, Any] = {}

    def fit(self, train: list[dict]) -> QuantilePreprocessor:
        super().fit(train)
        numeric = np.concatenate([np.asarray(e['x_num'], dtype=float) for e in train])
        self.quantile_models = []
        self.observed_counts = []
        for index in range(self.n_num):
            values = numeric[:, index]
            values = values[np.isfinite(values)]
            self.observed_counts.append(len(values))
            # A wholly missing column has no estimable support. Preserve a
            # finite sentinel model and expose that fact in serialized state.
            fitted_values = values if len(values) else np.zeros(1)
            if self.quantile_grid == 'all_observed':
                # Keep every empirical rank, including ties. A sparse grid can
                # spread a single rare outlier over an entire quantile interval.
                count = len(fitted_values)
            else:
                count = min(len(fitted_values), max(min(len(fitted_values) // 30, 1000), 10))
            model = QuantileTransformer(n_quantiles=count, output_distribution='normal',
                                        subsample=1_000_000_000, random_state=self.seed)
            model.fit(fitted_values[:, None])
            self.quantile_models.append(model)
        self.integer_features = [index for index, name in enumerate(self.numeric_names)
                                 if name in {'count', 'quantity', 'distinct_products'}]
        # Compatibility fields are not the numeric transform for this class.
        self.mean = np.zeros(self.n_num)
        self.scale = np.ones(self.n_num)
        return self

    def transform(self, entity: dict) -> dict[str, torch.Tensor]:
        result = super().transform(entity)
        numeric = np.asarray(entity['x_num'], dtype=float)
        transformed = np.zeros_like(numeric)
        for index, model in enumerate(self.quantile_models):
            valid = np.isfinite(numeric[:, index])
            if valid.any():
                transformed[valid, index] = model.transform(numeric[valid, index:index + 1])[:, 0]
        result['x_num'] = torch.tensor(transformed, dtype=torch.float32)
        if 'exposure_end' in entity:
            end = float(entity['exposure_end'])
            if not np.isfinite(end) or end < 0:
                raise ValueError('Administrative exposure must be finite and nonnegative')
            if len(entity['delta']) and end + 1e-8 < np.sum(entity['delta']):
                raise ValueError('Administrative exposure cannot precede the final observed event')
            result['exposure_end'] = torch.tensor(end, dtype=torch.float64)
        return result

    def inverse_numeric(self, values: Any) -> np.ndarray:
        if not self.fitted:
            raise ValueError('Unfitted preprocessor')
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=float)
        if array.ndim < 1 or array.shape[-1] != self.n_num:
            raise ValueError('Numeric inverse schema mismatch')
        result = array.copy()
        if self.n_num == 0:
            return result
        flat = result.reshape(-1, self.n_num)
        audit = {}
        for index, model in enumerate(self.quantile_models):
            valid = np.isfinite(flat[:, index])
            if valid.any():
                flat[valid, index] = model.inverse_transform(flat[valid, index:index + 1])[:, 0]
            if index in self.integer_features:
                original = flat[valid, index].copy()
                flat[valid, index] = np.rint(original)
                audit[self.numeric_names[index]] = {
                    'policy': 'nearest_integer_after_inverse_quantile',
                    'observed_values': int(valid.sum()),
                    'adjusted_fraction': float(np.mean(np.abs(original - flat[valid, index]) > 1e-8))
                    if len(original) else 0.,
                }
        self.last_inverse_audit = audit
        return result

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({'kind': 'tabsyn_gaussian_quantile_v1', 'seed': self.seed,
                      'observed_counts': self.observed_counts,
                      'integer_features': self.integer_features,
                      'integer_inverse_policy': 'nearest_integer_after_inverse_quantile',
                      'quantile_models': [{'quantiles': model.quantiles_[:, 0].tolist(),
                                           'references': model.references_.tolist()}
                                          for model in self.quantile_models]})
        # Omitting the default preserves existing serialized states/hashes.
        if self.quantile_grid != 'legacy_1000':
            state['quantile_grid'] = self.quantile_grid
        return state

    def load_state_dict(self, state: dict) -> QuantilePreprocessor:
        if state.get('kind') != 'tabsyn_gaussian_quantile_v1':
            raise ValueError('Expected a TabSyn Gaussian-quantile state')
        mode = state.get('quantile_grid', 'legacy_1000')
        if mode not in {'legacy_1000', 'all_observed'}:
            raise ValueError('quantile_grid must be legacy_1000 or all_observed')
        self.quantile_grid = mode
        models = state['quantile_models']
        super().load_state_dict({key: value for key, value in state.items()
                                 if key not in {'quantile_models', 'kind', 'integer_inverse_policy'}})
        self.quantile_models = []
        for values in models:
            quantiles = np.asarray(values['quantiles'], dtype=float)
            references = np.asarray(values['references'], dtype=float)
            if len(quantiles) < 1 or quantiles.shape != references.shape:
                raise ValueError('Invalid quantile state dimensions')
            if not np.isfinite(quantiles).all() or not np.isfinite(references).all():
                raise ValueError('Nonfinite quantile state')
            if np.any(np.diff(quantiles) < 0) or np.any(np.diff(references) < 0):
                raise ValueError('Nonmonotonic quantile state')
            model = QuantileTransformer(n_quantiles=len(quantiles), output_distribution='normal',
                                        subsample=1_000_000_000, random_state=self.seed)
            model.quantiles_ = quantiles[:, None]
            model.references_ = references
            model.n_quantiles_ = len(quantiles)
            model.n_features_in_ = 1
            self.quantile_models.append(model)
        if len(self.quantile_models) != self.n_num:
            raise ValueError('Quantile state feature count mismatch')
        return self

    @classmethod
    def from_state_dict(cls, state: dict) -> QuantilePreprocessor:
        return cls(state['max_categories'], state['seed'],
                   state.get('quantile_grid', 'legacy_1000')).load_state_dict(state)
