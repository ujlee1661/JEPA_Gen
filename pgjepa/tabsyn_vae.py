"""Token Transformer VAE for quantile-normalized mixed event observations.

The feature tokenizer, separate posterior-mean/logvariance Transformer branches,
reparameterization, and numeric-MSE/categorical-CE/KL objectives follow TabSyn's
design: https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/model.py
and https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py .
This independent implementation adds explicit observation-mask tokens and BCE
for missingness. Diffusion operates on posterior means, as in TabSyn. Numerical
decoding adds no Gaussian residual noise; the external fitted quantile inverse
restores original units. Train-fitted transformed support bounds canonicalize
inverse-transform endpoint saturation before history feedback.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TabSynVAE(nn.Module):
    """TabSyn-style feature-token VAE with the PGGenerator row-codec interface."""

    def __init__(self, n_num: int, cat_sizes: list[int], token_dim: int = 4,
                 layers: int = 2, heads: int = 1, beta: float = .01):
        super().__init__()
        if n_num < 0 or any(size < 1 for size in cat_sizes) or n_num + len(cat_sizes) < 1:
            raise ValueError('A nonempty mixed-event schema is required')
        if token_dim < 1 or layers < 1 or heads < 1 or token_dim % heads:
            raise ValueError('Positive dimensions and token_dim divisible by heads required')
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('KL beta must be finite and nonnegative')
        self.n_num, self.cat_sizes = n_num, list(cat_sizes)
        self.token_dim = token_dim
        self.n_fields = n_num + len(cat_sizes)
        self.n_tokens = 2 * self.n_fields
        self.latent_dim = self.n_tokens * token_dim
        self.numeric_weight = nn.Parameter(torch.empty(n_num, token_dim))
        self.feature_bias = nn.Parameter(torch.empty(self.n_fields, token_dim))
        self.mask_weight = nn.Parameter(torch.empty(self.n_fields, token_dim))
        self.mask_bias = nn.Parameter(torch.empty(self.n_fields, token_dim))
        self.missing_embedding = nn.Parameter(torch.empty(self.n_fields, token_dim))
        self.embeddings = nn.ModuleList([nn.Embedding(size, token_dim) for size in cat_sizes])
        for parameter in (self.numeric_weight, self.feature_bias, self.mask_weight,
                          self.mask_bias, self.missing_embedding):
            nn.init.normal_(parameter, std=.1)

        def transformer():
            layer = nn.TransformerEncoderLayer(token_dim, heads, token_dim * 32,
                                                dropout=0., activation='gelu', batch_first=True,
                                                norm_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        self.mean_encoder = transformer()
        self.logvar_encoder = transformer()
        self.mean_projection = nn.Linear(token_dim, token_dim)
        self.logvar_projection = nn.Linear(token_dim, token_dim)
        self.decoder = transformer()
        self.num_heads = nn.ModuleList([nn.Linear(token_dim, 1) for _ in range(n_num)])
        self.cat_heads = nn.ModuleList([nn.Linear(token_dim, size) for size in cat_sizes])
        self.mask_heads = nn.ModuleList([nn.Linear(token_dim, 1) for _ in range(self.n_fields)])
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('latent_mean', torch.zeros(self.latent_dim))
        self.register_buffer('latent_std', torch.ones(self.latent_dim))
        self.register_buffer('numeric_low', torch.zeros(n_num))
        self.register_buffer('numeric_high', torch.zeros(n_num))
        self.register_buffer('support_configured', torch.tensor(False))
        self.register_buffer('support_adjusted', torch.zeros(n_num, dtype=torch.long))
        self.register_buffer('support_total', torch.zeros(n_num, dtype=torch.long))

    def set_beta(self, beta: float) -> None:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('KL beta must be finite and nonnegative')
        self.beta.fill_(beta)

    def _tokens(self, batch: dict[str, Tensor]) -> Tensor:
        shape = batch['mask'].shape[:-1]
        mask = batch['mask'].reshape(-1, self.n_fields).float()
        numeric = batch['x_num'].reshape(len(mask), self.n_num).nan_to_num()
        parts = [numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
                 * mask[:, :self.n_num, None]]
        if self.cat_sizes:
            categorical = batch['x_cat'].reshape(-1, len(self.cat_sizes))
            parts.append(torch.stack([embedding(categorical[:, j])
                                      * mask[:, self.n_num + j, None]
                                      for j, embedding in enumerate(self.embeddings)], 1))
        features = torch.cat(parts, 1) + self.feature_bias[None]
        features = features + (1 - mask[..., None]) * self.missing_embedding[None]
        masks = mask[..., None] * self.mask_weight[None] + self.mask_bias[None]
        return torch.cat([features, masks], 1).reshape(*shape, self.n_tokens, self.token_dim)

    def posterior(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        tokens = self._tokens(batch)
        shape = tokens.shape[:-2]
        flat = tokens.reshape(-1, self.n_tokens, self.token_dim)
        mean = self.mean_projection(self.mean_encoder(flat))
        # A numerical variance guard; it does not clip observation outputs.
        logvar = self.logvar_projection(self.logvar_encoder(flat)).clamp(-12, 8)
        return mean.reshape(*shape, self.latent_dim), logvar.reshape(*shape, self.latent_dim)

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = self._tokens(batch)
        shape = tokens.shape[:-2]
        flat = tokens.reshape(-1, self.n_tokens, self.token_dim)
        mean = self.mean_projection(self.mean_encoder(flat))
        return mean.reshape(*shape, self.latent_dim)

    def _heads(self, latent: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        shape = latent.shape[:-1]
        hidden = self.decoder(latent.reshape(-1, self.n_tokens, self.token_dim))
        if self.n_num:
            numeric = torch.cat([head(hidden[:, j]) for j, head in enumerate(self.num_heads)], -1)
        else:
            numeric = hidden.new_zeros((len(hidden), 0))
        categories = [head(hidden[:, self.n_num + j]).reshape(*shape, self.cat_sizes[j])
                      for j, head in enumerate(self.cat_heads)]
        masks = torch.cat([head(hidden[:, self.n_fields + j])
                           for j, head in enumerate(self.mask_heads)], -1)
        return numeric.reshape(*shape, self.n_num), categories, masks.reshape(*shape, self.n_fields)

    def reconstruction_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        mean, logvar = self.posterior(batch)
        latent = mean + torch.randn_like(mean) * (.5 * logvar).exp() if self.training else mean
        numeric, categories, masks = self._heads(latent)
        observed = batch['mask'].float()
        valid = batch.get('t_valid', torch.ones_like(observed[..., 0], dtype=torch.bool)).float()
        weights = observed[..., :self.n_num] * valid.unsqueeze(-1)
        mse = ((numeric - batch['x_num'].nan_to_num()).square() * weights).sum() / weights.sum().clamp_min(1)
        ce, correct, count = mean.sum() * 0, mean.sum() * 0, mean.sum() * 0
        for j, logits in enumerate(categories):
            category_weights = observed[..., self.n_num + j] * valid
            labels = batch['x_cat'][..., j]
            errors = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                     reduction='none').reshape_as(category_weights)
            ce = ce + (errors * category_weights).sum()
            correct = correct + ((logits.argmax(-1) == labels).float() * category_weights).sum()
            count = count + category_weights.sum()
        ce = ce / count.clamp_min(1)
        bce = (F.binary_cross_entropy_with_logits(masks, observed, reduction='none').mean(-1)
               * valid).sum() / valid.sum().clamp_min(1)
        kl = (-.5 * (1 + logvar - mean.square() - logvar.exp()).mean(-1)
              * valid).sum() / valid.sum().clamp_min(1)
        accuracy = (((masks >= 0) == observed.bool()).float().mean(-1)
                    * valid).sum() / valid.sum().clamp_min(1)
        loss = mse + ce + bce + self.beta * kl
        return loss, {'numeric_mse': float(mse.detach()), 'categorical_ce': float(ce.detach()),
                      'categorical_accuracy': float((correct / count.clamp_min(1)).detach()),
                      'mask_bce': float(bce.detach()), 'mask_accuracy': float(accuracy.detach()),
                      'kl': float(kl.detach()), 'beta': float(self.beta)}

    def decode(self, latent: Tensor, sample: bool = True) -> dict[str, Tensor]:
        numeric, categories, masks = self._heads(latent)
        observed = torch.bernoulli(masks.sigmoid()) if sample else (masks >= 0).float()
        if bool(self.support_configured) and self.n_num:
            bounded = numeric.maximum(self.numeric_low).minimum(self.numeric_high)
            with torch.no_grad():
                axes = tuple(range(numeric.ndim - 1))
                self.support_adjusted.add_(((bounded != numeric) & observed[..., :self.n_num].bool()).sum(axes))
                self.support_total.add_(observed[..., :self.n_num].long().sum(axes))
            numeric = bounded
        categorical = (torch.stack([logits.argmax(-1) for logits in categories], -1) if categories else
                       torch.empty((*latent.shape[:-1], 0), dtype=torch.long, device=latent.device))
        return {'x_num': numeric, 'x_cat': categorical, 'mask': observed}

    @torch.no_grad()
    def fit_latent_stats(self, train_batch: dict[str, Tensor]) -> None:
        valid = train_batch.get('t_valid', torch.ones_like(train_batch['mask'][..., 0], dtype=torch.bool))
        valid = valid.reshape(-1).bool()
        rows = {key: value.reshape(len(valid), value.shape[-1]) for key, value in train_batch.items()
                if key in {'x_num', 'x_cat', 'mask'}}
        if int(valid.sum()) < 2:
            raise ValueError('At least two training rows required for latent statistics')
        total = self.latent_mean.new_zeros(self.latent_dim, dtype=torch.float64)
        squares = total.clone()
        count = 0
        for start in range(0, len(valid), 1024):
            keep = valid[start:start + 1024]
            if not bool(keep.any()):
                continue
            batch = {key: value[start:start + 1024][keep] for key, value in rows.items()}
            latent = self.encode(batch).double()
            total += latent.sum(0)
            squares += latent.square().sum(0)
            count += len(latent)
        average = total / count
        self.latent_mean.copy_(average)
        self.latent_std.copy_((squares / count - average.square()).clamp_min(0).sqrt().clamp_min(.01))
        for j in range(self.n_num):
            observed = valid & (rows['mask'][:, j] >= .5)
            values = rows['x_num'][:, j][observed]
            self.numeric_low[j] = values.min() if len(values) else 0.
            self.numeric_high[j] = values.max() if len(values) else 0.
        self.support_configured.fill_(True)
        self.reset_decoder_audit()

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.latent_std + self.latent_mean

    @torch.no_grad()
    def reset_decoder_audit(self) -> None:
        self.support_adjusted.zero_()
        self.support_total.zero_()

    def decoder_audit(self, reset: bool = False) -> dict:
        result = {'codec': 'TabSyn-style token Transformer VAE; explicit mask adaptation',
                  'numeric_sampling': 'deterministic decoder mean; external train-fitted quantile inverse',
                  'categorical_sampling': 'argmax', 'missingness_sampling': 'Bernoulli mask',
                  'support_policy': 'train-fitted transformed min/max; equivalent quantile endpoint saturation',
                  'configured': bool(self.support_configured),
                  'support_adjusted': self.support_adjusted.detach().cpu().tolist(),
                  'support_total': self.support_total.detach().cpu().tolist(),
                  'support_adjustment_rate': (self.support_adjusted / self.support_total.clamp_min(1)).detach().cpu().tolist(),
                  'beta': float(self.beta)}
        if reset:
            self.reset_decoder_audit()
        return result
