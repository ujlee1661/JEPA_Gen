"""Canonicalize quantile decoder rows before recurrent observation feedback."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def attach_quantile_feedback(codec: Any, prep: Any) -> dict:
    """Use exactly the exported integer observations in the recurrent state.

    The fitted inverse rounds declared integer features. Mapping those raw
    observations forward again prevents feedback from seeing fractional counts
    while exported trajectories contain integers. No additional raw clipping is
    performed, and missing-cell values remain masked.
    """
    original = codec.decode
    audit: dict[str, Any] = {'observed_integer_values': 0, 'integer_values_changed': 0,
             'policy': 'inverse_quantile_then_declared_integer_rounding_then_forward_quantile',
             'scope': 'all decoder calls, including sampled rows discarded after window stopping'}

    @torch.no_grad()
    def decode(latent, sample=True):
        row = original(latent, sample)
        transformed = row['x_num'].detach().cpu().numpy()
        shape = transformed.shape
        raw = prep.inverse_numeric(transformed)
        flat = raw.reshape(-1, prep.n_num)
        canonical = np.zeros_like(flat)
        for j, transformer in enumerate(prep.quantile_models):
            canonical[:, j] = transformer.transform(flat[:, j:j + 1])[:, 0]
            if j in prep.integer_features:
                before = transformer.inverse_transform(transformed.reshape(-1, prep.n_num)[:, j:j + 1])[:, 0]
                observed = row['mask'][..., j].detach().cpu().numpy().reshape(-1) >= .5
                audit['observed_integer_values'] += int(observed.sum())
                audit['integer_values_changed'] += int(np.sum((np.abs(before - flat[:, j]) > 1e-6) & observed))
        row['x_num'] = torch.as_tensor(canonical.reshape(shape), device=latent.device,
                                        dtype=row['x_num'].dtype)
        return row

    codec.decode = decode
    return audit
