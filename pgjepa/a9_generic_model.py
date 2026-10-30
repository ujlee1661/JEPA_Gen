"""A9 with an optional trajectory-level stochastic feature condition.

The Gaussian sample is an exploratory conditioning mechanism, not a claim
that the auxiliary predictor is a calibrated posterior. No target is read by
the generation path; one independent epsilon is reused across every prefix and
reverse step of a generated trajectory.
"""
from __future__ import annotations

import torch
from torch import Tensor
from .a9_model import A9Generator


class A9GenericGenerator(A9Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic_condition = False
        self._condition_epsilon: Tensor | None = None
        self.feature_residual.register_forward_pre_hook(self._condition_hook)

    def _condition_hook(self, module, args):
        x = args[0]
        if not self.stochastic_condition:
            return args
        e = self._condition_epsilon
        if e is None or e.shape != (x.shape[0], 1, self.code_dim):
            raise RuntimeError('Missing per-trajectory feature condition noise')
        d = self.code_dim
        # Input layout: existing mean/scale, then feature mean/scale.
        return (torch.cat([x[..., :2*d], x[..., 2*d:3*d] + x[..., 3*d:4*d]*e,
                           x[..., 3*d:]], dim=-1),)

    def _draw_condition(self, batch_size, device, dtype):
        if self._condition_epsilon is not None:
            raise RuntimeError('Nested trajectory contexts are unsupported')
        # Draw for both arms, including the deterministic control, to preserve
        # the matched RNG schedule between arms.
        self._condition_epsilon = torch.randn(batch_size, 1, self.code_dim, device=device, dtype=dtype)

    def loss(self, batch, *args, **kwargs):
        x = batch['tokens']
        self._draw_condition(len(x), x.device, x.dtype)
        try:
            return super().loss(batch, *args, **kwargs)
        finally:
            self._condition_epsilon = None

    @torch.no_grad()
    def sample_paths(self, attrs, anchor, delta, **kwargs):
        self._draw_condition(len(attrs), attrs.device, attrs.dtype)
        try:
            return super().sample_paths(attrs, anchor, delta, **kwargs)
        finally:
            self._condition_epsilon = None
