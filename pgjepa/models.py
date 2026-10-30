"""Independent mixed-event AE and predictive-guided latent diffusion.

Future observations occur only in the detached Stage B target branch. The
Stage C denoiser always conditions on predictions computed from the prefix.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def prefix_of(batch: dict[str, Tensor], cut: int) -> dict[str, Tensor]:
    return {k: (v[:, :cut] if k in {"x_num", "x_cat", "mask", "delta", "t_valid"} else v)
            for k, v in batch.items()}


def _mean_valid(values: Tensor, valid: Tensor) -> Tensor:
    return (values * valid).sum() / valid.sum().clamp_min(1)


class EventAE(nn.Module):
    """Feature-specific category embeddings, observed masks and a shared row MLP."""

    def __init__(self, n_num: int, cat_sizes: list[int], latent_dim: int = 64,
                 hidden: int = 256):
        super().__init__()
        if n_num < 0 or any(n < 1 for n in cat_sizes) or n_num + len(cat_sizes) < 1:
            raise ValueError("A nonempty mixed-event schema is required")
        self.n_num, self.cat_sizes, self.latent_dim = n_num, list(cat_sizes), latent_dim
        self.embeddings = nn.ModuleList([nn.Embedding(n, 16) for n in cat_sizes])
        inputs = 2 * n_num + 17 * len(cat_sizes)
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                     nn.Linear(hidden // 2, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden // 2), nn.SiLU(),
                                     nn.Linear(hidden // 2, hidden // 2), nn.SiLU())
        self.num_head = nn.Linear(hidden // 2, n_num) if n_num else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden // 2, n) for n in cat_sizes])
        self.mask_head = nn.Linear(hidden // 2, n_num + len(cat_sizes))
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer("numeric_scale", torch.full((n_num,), .05))
        self.register_buffer("domain_mean", torch.zeros(n_num))
        self.register_buffer("domain_scale", torch.ones(n_num))
        self.register_buffer("domain_codes", torch.zeros(n_num, dtype=torch.long))
        self.register_buffer("domain_configured", torch.tensor(False))
        self.register_buffer("domain_adjusted", torch.zeros(n_num, dtype=torch.long))
        self.register_buffer("domain_total", torch.zeros(n_num, dtype=torch.long))
        self.register_buffer("domain_overflow_protected", torch.zeros(n_num, dtype=torch.long))

    @torch.no_grad()
    def configure_numeric_domains(self, mean: list[float], scale: list[float],
                                  domains: list[str]) -> None:
        """Configure explicit, audited output-domain postprocessing.

        Inputs use the data adapter's train-fitted signed-log transform. This
        changes decoded samples, never the AE reconstruction objective.
        """
        names = ["real", "positive", "nonnegative_count", "signed_integer", "positive_count"]
        if len(mean) != self.n_num or len(scale) != self.n_num or len(domains) != self.n_num:
            raise ValueError("Numeric domain configuration must match the numeric feature count")
        means = self.domain_mean.new_tensor(mean)
        scales = self.domain_scale.new_tensor(scale)
        if not torch.isfinite(means).all() or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Numeric transform parameters must be finite with positive scales")
        if any(name not in names for name in domains):
            raise ValueError(f"Unsupported numeric domain; choose from {names}")
        self.domain_mean.copy_(means)
        self.domain_scale.copy_(scales)
        self.domain_codes.copy_(self.domain_codes.new_tensor([names.index(name) for name in domains]))
        self.domain_configured.fill_(True)
        self.reset_decoder_audit()

    @torch.no_grad()
    def reset_decoder_audit(self) -> None:
        self.domain_adjusted.zero_()
        self.domain_total.zero_()
        self.domain_overflow_protected.zero_()

    def decoder_audit(self, reset: bool = False) -> dict:
        names = ["real", "positive", "nonnegative_count", "signed_integer", "positive_count"]
        counts = self.domain_total.detach().cpu()
        adjusted = self.domain_adjusted.detach().cpu()
        overflow = self.domain_overflow_protected.detach().cpu()
        total = int(counts.sum())
        result = {
            "configured": bool(self.domain_configured),
            "policy": "inverse_signed_log_domain_projection_then_retransform",
            "feature_domains": [names[int(code)] for code in self.domain_codes.detach().cpu()],
            "domain_adjusted": adjusted.tolist(), "domain_total": counts.tolist(),
            "domain_adjustment_rate": (adjusted / counts.clamp_min(1)).tolist(),
            "overflow_protected": overflow.tolist(),
            "overflow_protection_rate": (overflow / counts.clamp_min(1)).tolist(),
            "overall_adjustment_rate": int(adjusted.sum()) / max(total, 1),
            "overall_overflow_protection_rate": int(overflow.sum()) / max(total, 1),
            "signed_log_exponent_limit": 30.,
            "numeric_sampling": "Gaussian decoder mean plus train-fitted reconstruction residual standard deviation",
        }
        if reset:
            self.reset_decoder_audit()
        return result

    def _project_numeric(self, numeric: Tensor, observed: Tensor) -> Tensor:
        if not bool(self.domain_configured) or self.n_num == 0:
            return numeric
        transformed = numeric * self.domain_scale + self.domain_mean
        if not bool(torch.isfinite(transformed).all()):
            raise FloatingPointError("Nonfinite decoder output before numeric domain projection")
        overflow = transformed.abs() > 30
        safe = transformed.clamp(-30, 30)
        raw = safe.sign() * torch.expm1(safe.abs())
        projected = raw.clone()
        for index, code in enumerate(self.domain_codes.tolist()):
            if code == 1:
                projected[..., index] = raw[..., index].clamp_min(0)
            elif code == 2:
                projected[..., index] = raw[..., index].round().clamp_min(0)
            elif code == 3:
                projected[..., index] = raw[..., index].round()
            elif code == 4:
                projected[..., index] = raw[..., index].round().clamp_min(1)
        adjusted = ~torch.isclose(projected, raw, rtol=0, atol=1e-6)
        weights = observed.bool()
        axes = tuple(range(numeric.ndim - 1))
        with torch.no_grad():
            self.domain_adjusted.add_((adjusted & weights).sum(axes))
            self.domain_total.add_(weights.sum(axes))
            self.domain_overflow_protected.add_((overflow & weights).sum(axes))
        new_transformed = projected.sign() * projected.abs().log1p()
        return (new_transformed - self.domain_mean) / self.domain_scale

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        numeric, categorical, mask = batch["x_num"], batch["x_cat"], batch["mask"].float()
        parts = [numeric.nan_to_num() * mask[..., :self.n_num], mask]
        for j, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[..., j]) * mask[..., self.n_num + j:self.n_num + j + 1])
        return self.encoder(torch.cat(parts, -1))

    def _heads(self, latent: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        hidden = self.decoder(latent)
        numeric = self.num_head(hidden) if self.num_head is not None else hidden[..., :0]
        return numeric, [head(hidden) for head in self.cat_heads], self.mask_head(hidden)

    def reconstruction_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        numeric, categories, mask_logits = self._heads(self.encode(batch))
        observed = batch["mask"].float()
        valid = batch.get("t_valid", torch.ones_like(observed[..., 0], dtype=torch.bool)).float()
        num_weights = observed[..., :self.n_num] * valid.unsqueeze(-1)
        num = ((numeric - batch["x_num"].nan_to_num()).square() * num_weights).sum() / num_weights.sum().clamp_min(1)
        cat_loss = numeric.sum() * 0
        cat_correct = numeric.sum() * 0
        cat_count = numeric.sum() * 0
        for j, logits in enumerate(categories):
            weights = observed[..., self.n_num + j] * valid
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 batch["x_cat"][..., j].reshape(-1), reduction="none").reshape_as(weights)
            cat_loss = cat_loss + (ce * weights).sum()
            cat_correct = cat_correct + ((logits.argmax(-1) == batch["x_cat"][..., j]) * weights).sum()
            cat_count = cat_count + weights.sum()
        cat_loss = cat_loss / cat_count.clamp_min(1)
        mask_loss = _mean_valid(F.binary_cross_entropy_with_logits(mask_logits, observed,
                                                                  reduction="none").mean(-1), valid)
        with torch.no_grad():
            mask_accuracy = _mean_valid(((mask_logits > 0) == observed.bool()).float().mean(-1), valid)
        return num + cat_loss + mask_loss, {
            "numeric_mse": float(num.detach()), "categorical_ce": float(cat_loss.detach()),
            "categorical_accuracy": float((cat_correct / cat_count.clamp_min(1)).detach()),
            "mask_bce": float(mask_loss.detach()), "mask_accuracy": float(mask_accuracy),
        }

    def decode(self, latent: Tensor, sample: bool = True) -> dict[str, Tensor]:
        numeric, categories, mask_logits = self._heads(latent)
        if sample:
            numeric = numeric + torch.randn_like(numeric) * self.numeric_scale
            cats = [torch.distributions.Categorical(logits=logits).sample() for logits in categories]
            mask = torch.bernoulli(mask_logits.sigmoid())
        else:
            cats = [logits.argmax(-1) for logits in categories]
            mask = (mask_logits >= 0).float()
        categorical = (torch.stack(cats, -1) if cats else
                       torch.empty((*latent.shape[:-1], 0), dtype=torch.long, device=latent.device))
        numeric = self._project_numeric(numeric, mask[..., :self.n_num])
        return {"x_num": numeric, "x_cat": categorical, "mask": mask}

    @torch.no_grad()
    def fit_latent_stats(self, train_batch: dict[str, Tensor]) -> None:
        latent = self.encode(train_batch)
        valid = train_batch.get("t_valid", torch.ones(latent.shape[:-1], device=latent.device,
                                                      dtype=torch.bool)).bool()
        rows = latent[valid]
        if len(rows) < 2:
            raise ValueError("At least two train rows are required for latent normalization")
        self.latent_mean.copy_(rows.mean(0))
        self.latent_std.copy_(rows.std(0, unbiased=False).clamp_min(.01))
        means, _, _ = self._heads(latent)
        residual = (means - train_batch["x_num"].nan_to_num()).square()
        weights = train_batch["mask"][..., :self.n_num] * valid.unsqueeze(-1)
        self.numeric_scale.copy_(((residual * weights).sum(tuple(range(weights.ndim - 1))) /
                                  weights.sum(tuple(range(weights.ndim - 1))).clamp_min(1)).sqrt().clamp_min(.01))

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.latent_std + self.latent_mean


class FiLMBlock(nn.Module):
    def __init__(self, width: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(condition_dim, width * 2)
        self.network = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, -1)
        return values + self.network(self.norm(values) * (1 + scale) + shift) / math.sqrt(2)


class ConditionalDenoiser(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, width: int, blocks: int):
        super().__init__()
        self.input = nn.Linear(latent_dim, width)
        self.condition = nn.Sequential(nn.Linear(condition_dim + 32, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([FiLMBlock(width, width) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, latent_dim))

    def forward(self, noisy: Tensor, steps: Tensor, condition: Tensor) -> Tensor:
        frequencies = torch.exp(torch.linspace(0, -math.log(10000), 16, device=noisy.device))
        angles = steps.float().unsqueeze(-1) * frequencies
        time_embedding = torch.cat([angles.sin(), angles.cos()], -1)
        cond = self.condition(torch.cat([condition, time_embedding], -1))
        hidden = self.input(noisy)
        for block in self.blocks:
            hidden = block(hidden, cond)
        return self.output(hidden)


class HurdleLogNormal(nn.Module):
    def __init__(self, condition_dim: int, width: int, components: int = 2, zero_gap: bool = True):
        super().__init__()
        self.components, self.zero_gap = components, zero_gap
        self.network = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(),
                                     nn.Linear(width, 3 * components + 1))

    def parameters_at(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raw = self.network(condition)
        zero = raw[..., 0]
        logits, mu, raw_scale = raw[..., 1:].chunk(3, -1)
        return zero, logits, mu, F.softplus(raw_scale) + .05

    def nll(self, condition: Tensor, delta: Tensor) -> Tensor:
        delta = delta.squeeze(-1)
        zero, logits, mu, sigma = self.parameters_at(condition)
        log_delta = delta.clamp_min(1e-12).log().unsqueeze(-1)
        log_density = (-.5 * ((log_delta - mu) / sigma).square() - sigma.log()
                       - log_delta - .5 * math.log(2 * math.pi))
        positive = -torch.logsumexp(F.log_softmax(logits, -1) + log_density, -1)
        if self.zero_gap:
            return torch.where(delta == 0, F.softplus(-zero), F.softplus(zero) + positive)
        if bool((delta == 0).any()):
            raise ValueError("Zero gaps require a hurdle head")
        return positive

    def sample(self, condition: Tensor) -> Tensor:
        zero, logits, mu, sigma = self.parameters_at(condition)
        component = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        means, scales = mu.gather(-1, component), sigma.gather(-1, component)
        # Float32 exponent overflow protection; the bound is logged in model metadata.
        delta = (means + scales * torch.randn_like(means)).clamp(-30, 30).exp()
        if self.zero_gap:
            delta = torch.where(torch.rand_like(delta) < zero.sigmoid().unsqueeze(-1), 0., delta)
        return delta


class PGGenerator(nn.Module):
    """Staged predictive-guided event generator with A0--A4/A6 controls."""

    def __init__(self, ae: EventAE, attrs_dim: int = 0, arm: str = "A4",
                 horizons: tuple[int, ...] = (4, 8, 16), hidden: int = 128,
                 entity_dim: int = 32, width: int = 256, blocks: int = 4,
                 diffusion_steps: int = 100, time_components: int = 2,
                 zero_gap: bool = True, time_mode: str = "T2",
                 diffusion_beta_max: float = .1):
        super().__init__()
        if arm not in {"A0", "A1", "A2", "A3", "A4", "A6"}:
            raise ValueError(f"Unknown experimental arm {arm}")
        if not horizons or min(horizons) < 1 or diffusion_steps < 2:
            raise ValueError("Positive horizons and at least two diffusion steps required")
        self.ae, self.arm, self.horizons = ae, arm, tuple(horizons)
        self.hidden_dim, self.entity_dim, self.attrs_dim = hidden, entity_dim, attrs_dim
        self.diffusion_steps = diffusion_steps
        if not 0 < diffusion_beta_max < 1:
            raise ValueError("Diffusion beta cap must be in (0, 1)")
        self.diffusion_beta_max = diffusion_beta_max
        if time_mode not in {"T0", "T1", "T2", "single", "global", "entity_mixture"}:
            raise ValueError(f"Unknown time model {time_mode}")
        self.time_mode = time_mode
        self.ema_momentum = .996
        latent = ae.latent_dim
        self.entity_encoder = nn.Sequential(nn.Linear(2 * (latent + 1) + attrs_dim, 128), nn.SiLU(),
                                            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, entity_dim))
        self.initial = nn.Linear(entity_dim, hidden)
        self.context = nn.GRUCell(latent + 1, hidden)
        self.target_context = copy.deepcopy(self.context)
        self.horizon_embedding = nn.Embedding(len(horizons), 16)
        self.predictor = nn.Sequential(nn.Linear(hidden + entity_dim + 16, 256), nn.SiLU(),
                                       nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, hidden))
        self.aggregator = nn.Sequential(nn.Linear(hidden * len(horizons), hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden))
        self.context_projection = nn.Sequential(nn.Linear(hidden + entity_dim, hidden), nn.SiLU(),
                                                nn.Linear(hidden, hidden))
        condition_dim = 2 * hidden + entity_dim
        components = 1 if time_mode in {"T0", "single", "global"} else time_components
        self.time_head = HurdleLogNormal(condition_dim, hidden, components, zero_gap)
        self.denoiser = ConditionalDenoiser(latent, condition_dim + 1, width, blocks)
        grid = torch.arange(diffusion_steps + 1, dtype=torch.float64) / diffusion_steps
        alpha = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
        alpha = alpha / alpha[0]
        # A 100-step epsilon model has severe x0 amplification at an almost
        # zero-SNR terminal cosine step. Capping per-step beta at .1 keeps
        # terminal alpha_bar about .0115 (100 steps), so amplification stays
        # below 10 rather than 2,000. This is an explicit schedule choice,
        # not a clipping operation on generated events or latent samples.
        beta = (1 - alpha[1:] / alpha[:-1]).clamp(max=diffusion_beta_max)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0).float())
        # Fixed, seeded random-feature map of explicit block mean/std is A6's
        # observation-summary target. It has no learned teacher and no EMA.
        rng = torch.Generator().manual_seed(181)
        projection = torch.randn(2 * (latent + 1), hidden, generator=rng) / math.sqrt(2 * (latent + 1))
        self.register_buffer("summary_projection", projection)
        self.stage = "C"
        self.configure_stage("C")

    def configure_stage(self, stage: str) -> None:
        if stage not in {"B", "C"}:
            raise ValueError("Generator stages are B and C; train the AE separately")
        self.stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        temporal = [self.entity_encoder, self.initial, self.context, self.horizon_embedding, self.predictor]
        plan_module = self.aggregator if self.arm in {"A4", "A6"} else self.context_projection
        train_modules = temporal if stage == "B" else [plan_module, self.time_head, self.denoiser]
        if stage == "C" and self.arm in {"A0", "A1", "A2"}:
            train_modules += temporal[:3]
        for module in train_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.ae.eval()
        self.target_context.eval()

    def train(self, mode: bool = True) -> PGGenerator:
        super().train(mode)
        self.ae.eval()
        self.target_context.eval()
        return self

    def _time_input(self, latent: Tensor, delta: Tensor) -> Tensor:
        return torch.cat([latent, delta.clamp_min(0).log1p()], -1)

    def init_state(self, prefix: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        latent = self.ae.encode(prefix)
        valid = prefix.get("t_valid", torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device))
        if bool((valid.sum(1) == 0).any()):
            raise ValueError("Each entity needs at least one observed prefix event")
        values = self._time_input(latent, prefix["delta"])
        weights = valid.unsqueeze(-1).float()
        count = weights.sum(1).clamp_min(1)
        mean = (values * weights).sum(1) / count
        variance = ((values - mean.unsqueeze(1)).square() * weights).sum(1) / count
        attrs = prefix.get("attrs", values.new_zeros((len(values), self.attrs_dim)))
        z = self.entity_encoder(torch.cat([mean, (variance + 1e-6).sqrt(), attrs], -1))
        if self.arm in {"A0", "A1"}:
            z = torch.zeros_like(z)
        state = self.initial(z)
        for t in range(values.shape[1]):
            updated = self.context(values[:, t], state)
            state = torch.where(valid[:, t:t + 1], updated, state)
        if self.arm == "A0":
            state = torch.zeros_like(state)
        return z, state

    def predict_blocks(self, g: Tensor, z: Tensor) -> Tensor:
        embeddings = self.horizon_embedding.weight
        return torch.stack([self.predictor(torch.cat([g, z, embedding.expand(len(g), -1)], -1))
                            for embedding in embeddings], 1)

    def plan(self, g: Tensor, z: Tensor, intervention: str = "normal") -> Tensor:
        if self.arm in {"A4", "A6"}:
            plan = self.aggregator(self.predict_blocks(g, z).flatten(1))
        else:
            plan = self.context_projection(torch.cat([g, z], -1))
        if intervention == "zero_plan":
            return torch.zeros_like(plan)
        if intervention == "shuffle_plan":
            # A rotation is a guaranteed derangement for B>1 and reproducible
            # independently of the diffusion RNG stream.
            return plan.roll(1, 0)
        return plan

    @torch.no_grad()
    def target_blocks(self, batch: dict[str, Tensor], cut: int, summary: bool = False) -> tuple[Tensor, Tensor]:
        latent = self.ae.encode(batch)
        values = self._time_input(latent, batch["delta"])
        valid = batch.get("t_valid", torch.ones(values.shape[:2], dtype=torch.bool, device=values.device))
        targets, supports = [], []
        for horizon in self.horizons:
            end = cut + horizon
            support = valid[:, cut:end].sum(1) == horizon
            support = support & (torch.ones_like(support) if end <= values.shape[1] else torch.zeros_like(support))
            future = values[:, cut:min(end, values.shape[1])]
            if summary and future.shape[1]:
                target = torch.cat([future.mean(1), future.std(1, unbiased=False)], -1) @ self.summary_projection
            else:
                target = values.new_zeros((len(values), self.hidden_dim))
                for j in range(future.shape[1]):
                    target = self.target_context(future[:, j], target)
            targets.append(target)
            supports.append(support)
        return torch.stack(targets, 1), torch.stack(supports, 1)

    def jepa_loss(self, batch: dict[str, Tensor], cut: int) -> tuple[Tensor, dict[str, float]]:
        z, g = self.init_state(prefix_of(batch, cut))
        prediction = self.predict_blocks(g, z)
        target, support = self.target_blocks(batch, cut, summary=self.arm == "A6")
        distance = (F.normalize(prediction, dim=-1) - F.normalize(target, dim=-1)).square().sum(-1)
        loss = _mean_valid(distance, support.float())
        stats = {"jepa_loss": float(loss.detach()), "supported_blocks": float(support.sum())}
        for j, horizon in enumerate(self.horizons):
            selected = support[:, j]
            if int(selected.sum()) > 1:
                stats[f"target_std_{horizon}"] = float(target[selected, j].std(0, unbiased=False).mean())
                stats[f"prediction_std_{horizon}"] = float(prediction[selected, j].detach().std(0, unbiased=False).mean())
                stats[f"forecast_error_{horizon}"] = float(distance[selected, j].detach().mean())
                shuffled_distance = (F.normalize(prediction[selected, j].detach().roll(1, 0), dim=-1)
                                     - F.normalize(target[selected, j], dim=-1)).square().sum(-1)
                stats[f"shuffled_history_error_{horizon}"] = float(shuffled_distance.mean())
        return loss, stats

    def supervised_loss(self, batch: dict[str, Tensor], cut: int) -> tuple[Tensor, dict[str, float]]:
        if self.arm != "A6":
            raise ValueError("Explicit fixed-summary supervision is the A6 control")
        return self.jepa_loss(batch, cut)

    @torch.no_grad()
    def update_ema(self) -> None:
        for target, current in zip(self.target_context.parameters(), self.context.parameters()):
            target.mul_(self.ema_momentum).add_(current, alpha=1 - self.ema_momentum)

    def _condition(self, z: Tensor, g: Tensor, intervention: str = "normal") -> Tensor:
        return torch.cat([g, z, self.plan(g, z, intervention)], -1)

    def _time_condition(self, condition: Tensor) -> Tensor:
        # T0/global: single global renewal law. T1: global mixture law.
        # T2/default: entity and history conditioned mixture.
        return torch.zeros_like(condition) if self.time_mode in {"T0", "T1", "global"} else condition

    def parameter_counts(self) -> dict[str, int]:
        return {"total": sum(p.numel() for p in self.parameters()),
                "without_ae_or_ema": sum(p.numel() for name, p in self.named_parameters()
                                          if not name.startswith(("ae.", "target_context."))),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def generation_loss(self, batch: dict[str, Tensor], cut: int) -> tuple[Tensor, dict[str, float]]:
        if cut < 1 or cut >= batch["x_num"].shape[1]:
            raise ValueError("Training cut requires a prefix and a next event")
        z, g = self.init_state(prefix_of(batch, cut))
        condition = self._condition(z, g)
        delta = batch["delta"][:, cut]
        with torch.no_grad():
            target = self.ae.normalize(self.ae.encode(batch)[:, cut])
        steps = torch.randint(self.diffusion_steps, (len(target),), device=target.device)
        noise = torch.randn_like(target)
        alpha = self.alpha_bar[steps].unsqueeze(-1)
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        eps = self.denoiser(noisy, steps, torch.cat([condition, delta.log1p()], -1))
        valid = batch.get("t_valid", torch.ones(batch["x_num"].shape[:2], dtype=torch.bool,
                                                device=target.device))[:, cut].float()
        diffusion_loss = _mean_valid((eps - noise).square().mean(-1), valid)
        time_loss = _mean_valid(self.time_head.nll(self._time_condition(condition), delta), valid)
        return diffusion_loss + time_loss, {"diffusion_loss": float(diffusion_loss.detach()),
                                             "time_nll": float(time_loss.detach())}

    @torch.no_grad()
    def sample_latent(self, condition: Tensor, delta: Tensor, sample_steps: int = 20) -> Tensor:
        count = min(max(int(sample_steps), 1), self.diffusion_steps)
        schedule = torch.linspace(self.diffusion_steps - 1, 0, count, device=condition.device).round().long()
        noisy = torch.randn((len(condition), self.ae.latent_dim), device=condition.device)
        full_condition = torch.cat([condition, delta.log1p()], -1)
        for index, step in enumerate(schedule):
            steps = step.expand(len(condition))
            eps = self.denoiser(noisy, steps, full_condition)
            alpha = self.alpha_bar[step]
            clean = (noisy - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            if index + 1 < len(schedule):
                previous = self.alpha_bar[schedule[index + 1]]
                noisy = previous.sqrt() * clean + (1 - previous).sqrt() * eps
            else:
                noisy = clean
        return self.ae.denormalize(noisy)

    def feedback(self, g: Tensor, row: dict[str, Tensor], delta: Tensor,
                 latent: Tensor | None = None) -> Tensor:
        encoded = self.ae.encode(row) if latent is None else latent
        return self.context(self._time_input(encoded, delta), g)

    @torch.no_grad()
    def generate(self, prefix: dict[str, Tensor], steps: int, sample_steps: int = 20,
                 intervention: str = "normal") -> dict[str, Tensor]:
        if steps < 1:
            raise ValueError("Generate a positive number of future events")
        if intervention not in {"normal", "zero_plan", "shuffle_plan", "direct_latent", "no_feedback"}:
            raise ValueError(f"Unknown intervention {intervention}")
        z, g = self.init_state(prefix)
        rows: dict[str, list[Tensor]] = {name: [] for name in ("x_num", "x_cat", "mask", "delta")}
        for _ in range(steps):
            condition = self._condition(z, g, intervention)
            delta = self.time_head.sample(self._time_condition(condition))
            latent = self.sample_latent(condition, delta, sample_steps)
            row = self.ae.decode(latent, sample=True)
            for key in ("x_num", "x_cat", "mask"):
                rows[key].append(row[key])
            rows["delta"].append(delta)
            if intervention != "no_feedback" and self.arm != "A0":
                g = self.feedback(g, row, delta, latent if intervention == "direct_latent" else None)
        output = {name: torch.stack(values, 1) for name, values in rows.items()}
        output["t_valid"] = torch.ones((len(z), steps), dtype=torch.bool, device=z.device)
        output["attrs"] = prefix.get("attrs", z.new_zeros((len(z), self.attrs_dim)))
        return output
