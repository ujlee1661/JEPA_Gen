"""Customer-level latent model with explicit calendar and next-event JEPA.

The full-history posterior is a training-only inference network. Native sampling
draws one customer latent from the attribute-conditioned prior, then keeps that
sample fixed while the recurrent history changes. No customer ID is an input.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .paper_model import PaperPGGenerator


def diagonal_gaussian_kl(q_mean: Tensor, q_scale: Tensor,
                         p_mean: Tensor, p_scale: Tensor) -> Tensor:
    """Analytic KL(q || p), summed over customer dimensions, per customer."""
    return (torch.log(p_scale) - torch.log(q_scale)
            + .5 * ((q_scale / p_scale).square()
                    + ((q_mean - p_mean) / p_scale).square() - 1)).sum(-1)


class HierarchicalPGGenerator(PaperPGGenerator):
    """A2/A4 with a persistent random customer and separate time conditioner.

    A4's predictive target is the *next event*, matching the modeled transition,
    rather than a hand-selected list of future block sizes. C-stage adaptation
    must be accompanied by next_event_jepa_loss in the training orchestrator.
    """

    conditioning_protocol = 'hierarchical_parent_only_full_fidelity'

    def __init__(self, *args: Any, **kwargs: Any):
        requested = tuple(kwargs.pop('horizons', (1,)))
        if requested != (1,):
            raise ValueError('Hierarchical JEPA predicts the next event: horizons must be (1,)')
        super().__init__(*args, horizons=(1,), **kwargs)
        if self.arm not in {'A2', 'A4'}:
            raise ValueError('Hierarchical comparison is defined for A2 and A4')
        self.summary_dim = 2 * (self.ae.latent_dim + 1) + 2
        self.customer_prior = nn.Sequential(
            nn.Linear(max(1, self.attrs_dim), self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.entity_dim))
        self.customer_posterior = nn.Sequential(
            nn.Linear(self.attrs_dim + self.summary_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.entity_dim))
        self.calendar_projection = nn.Sequential(
            nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.configure_stage('C')

    @staticmethod
    def _gaussian(output: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_scale = output.chunk(2, -1)
        # Softplus parameterizes a positive scale without clipping samples or KL.
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).eps
        return mean, scale

    def _check_attrs(self, attrs: Tensor) -> None:
        if attrs.ndim != 2 or attrs.shape[1] != self.attrs_dim or not bool(torch.isfinite(attrs).all()):
            raise ValueError('Finite parent attributes must match attrs_dim')

    def prior_parameters(self, attrs: Tensor) -> tuple[Tensor, Tensor]:
        self._check_attrs(attrs)
        inputs = attrs if self.attrs_dim else attrs.new_zeros((len(attrs), 1))
        return self._gaussian(self.customer_prior(inputs))

    @torch.no_grad()
    def posterior_summary(self, full_history: dict[str, Tensor]) -> Tensor:
        """Frozen-AE moments, log-gap moments, log count and log active span.

        This consumes full observed histories only for amortized training
        inference. It is never called by parent_state or native generation.
        """
        if not self.training:
            raise RuntimeError('Full-history posterior inference is training-only')
        latent = self.ae.encode(full_history)
        if latent.ndim != 3:
            raise ValueError('Full history requires [customer, event, feature] tensors')
        valid = full_history.get('t_valid', torch.ones(latent.shape[:2], dtype=torch.bool,
                                                      device=latent.device)).bool()
        if valid.shape != latent.shape[:2]:
            raise ValueError('History validity shape differs from event dimensions')
        gaps = full_history['delta']
        if gaps.shape != (*valid.shape, 1) or not bool(torch.isfinite(gaps[valid]).all()) or bool((gaps[valid] < 0).any()):
            raise ValueError('History time gaps must be finite and nonnegative')
        values = torch.cat([latent, gaps.log1p()], -1)
        weights = valid.unsqueeze(-1).to(values.dtype)
        count = weights.sum(1)
        divisor = count.clamp_min(1)
        mean = (values * weights).sum(1) / divisor
        variance = ((values - mean[:, None]).square() * weights).sum(1) / divisor
        std = (variance + torch.finfo(values.dtype).eps).sqrt()
        total = (gaps * weights).sum(1)
        first = gaps[:, 0] * weights[:, 0] if gaps.shape[1] else total.new_zeros(total.shape)
        span = (total - first).clamp_min(0)
        return torch.cat([mean, std, count.log1p(), span.log1p()], -1)

    def posterior_parameters(self, attrs: Tensor, summary: Tensor) -> tuple[Tensor, Tensor]:
        self._check_attrs(attrs)
        if not self.training:
            raise RuntimeError('Full-history posterior inference is training-only')
        if summary.shape != (len(attrs), self.summary_dim) or not bool(torch.isfinite(summary).all()):
            raise ValueError('Finite full-history summary must match customer and summary dimensions')
        return self._gaussian(self.customer_posterior(torch.cat([attrs, summary], -1)))

    def posterior_sample(self, attrs: Tensor, summary: Tensor) -> tuple[Tensor, Tensor]:
        q_mean, q_scale = self.posterior_parameters(attrs, summary)
        p_mean, p_scale = self.prior_parameters(attrs)
        z = q_mean + q_scale * torch.randn_like(q_mean)
        return z, diagonal_gaussian_kl(q_mean, q_scale, p_mean, p_scale)

    def parent_state(self, attrs: Tensor) -> tuple[Tensor, Tensor]:
        mean, scale = self.prior_parameters(attrs)
        z = mean + scale * torch.randn_like(mean)
        return z, self.initial(z)

    def init_state(self, prefix: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        numeric = prefix['x_num']
        if numeric.ndim != 3:
            raise ValueError('Event prefixes must have shape [customer, event, feature]')
        attrs = prefix.get('attrs', numeric.new_zeros((len(numeric), self.attrs_dim)))
        if 'customer_z' in prefix:
            if not self.training:
                raise RuntimeError('Posterior customer_z overrides are forbidden during generation')
            self._check_attrs(attrs)
            z = prefix['customer_z']
            if z.shape != (len(numeric), self.entity_dim) or not bool(torch.isfinite(z).all()):
                raise ValueError('Customer latent override must match customer and latent dimensions')
            initial = self.initial(z)
        else:
            z, initial = self.parent_state(attrs)
        if numeric.shape[1] == 0:
            return z, initial
        latent = self.ae.encode(prefix)
        valid = prefix.get('t_valid', torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device))
        return z, self.history_state(self._time_input(latent, prefix['delta']), valid, initial)

    def condition_at(self, z: Tensor, state: Tensor, elapsed: Tensor,
                     intervention: str = 'normal') -> Tensor:
        """Condition on the explicit current calendar, without mutable clock state."""
        elapsed = torch.as_tensor(elapsed, device=state.device, dtype=state.dtype)
        if elapsed.ndim == 1:
            elapsed = elapsed[:, None]
        if elapsed.shape != (len(state), 1) or not bool(torch.isfinite(elapsed).all()) or bool((elapsed < 0).any()):
            raise ValueError('Current calendar must be finite nonnegative [customer] or [customer,1]')
        clock_state = state + self.calendar_projection(elapsed.log1p())
        return torch.cat([clock_state, z, self.plan(state, z, intervention)], -1)

    def _condition(self, z: Tensor, g: Tensor, intervention: str = 'normal') -> Tensor:
        # Explicit-zero compatibility only. Native sampler and hierarchical
        # transition loss call condition_at with their actual elapsed clocks.
        return self.condition_at(z, g, g.new_zeros(len(g)), intervention)

    def _time_condition(self, condition: Tensor) -> Tensor:
        if self.time_mode in {'T0', 'T1', 'global'}:
            return torch.zeros_like(condition)
        # Time sees customer + current state/calendar, never predicted marks.
        base = condition[..., :self.hidden_dim + self.entity_dim]
        return torch.cat([base, torch.zeros_like(condition[..., -self.hidden_dim:])], -1)

    def configure_stage(self, stage: str) -> None:
        super().configure_stage(stage)
        if not hasattr(self, 'customer_prior'):
            return  # PGGenerator calls this during base construction.
        # The deterministic old entity encoder is superseded by p(z|s).
        for parameter in self.entity_encoder.parameters():
            parameter.requires_grad_(False)
        if stage == 'C':
            modules = [self.customer_prior, self.customer_posterior,
                       self.calendar_projection, self.initial, self.context]
            if self.arm == 'A4':
                modules += [self.horizon_embedding, self.predictor, self.aggregator]
        else:
            # JEPA pretraining uses sampled prior customers, not future histories.
            modules = [self.customer_prior, self.initial, self.context,
                       self.horizon_embedding, self.predictor]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def next_event_jepa_loss(self, prefix: dict[str, Tensor], z: Tensor,
                             target: dict[str, Tensor], observed: Tensor,
                             weights: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        """Prediction of the next event's EMA representation under the same z."""
        if self.arm != 'A4':
            raise ValueError('Next-event JEPA is defined only for A4')
        enriched = dict(prefix, customer_z=z)
        z, state = self.init_state(enriched)
        prediction = self.predict_blocks(state, z)[:, 0]
        with torch.no_grad():
            latent = self.ae.encode(target)
            teacher = self.target_context(self._time_input(latent, target['delta']),
                                           state.new_zeros(state.shape))
        distance = (F.normalize(prediction, dim=-1) - F.normalize(teacher, dim=-1)).square().sum(-1)
        selected = observed.bool()
        weights = torch.ones_like(distance) if weights is None else weights
        if weights.shape != distance.shape:
            raise ValueError('JEPA term weights must match the full sampled batch')
        loss = (distance[selected] * weights[selected]).sum() / len(distance)
        return loss, {'jepa_loss': float(loss.detach()), 'supported_blocks': float(selected.sum())}
