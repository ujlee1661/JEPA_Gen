"""Original event MLP codec with external quantile observation restoration."""
from __future__ import annotations

import torch
from torch import Tensor

from .models import EventAE


class QuantileEventAE(EventAE):
    """Keep EventAE's exact parameter architecture; change no network layers.

    Numerical values are decoder means in quantile coordinates. Additional
    residual Gaussian noise and signed-log output projection are inappropriate
    for this codec: its external train-fitted quantile inverse owns restoration.
    Categories/masks retain the original sampling policies.
    """

    def decode(self, latent: Tensor, sample: bool = True) -> dict[str, Tensor]:
        numeric, categories, mask_logits = self._heads(latent)
        if sample:
            cats = [torch.distributions.Categorical(logits=logits).sample() for logits in categories]
            mask = torch.bernoulli(mask_logits.sigmoid())
        else:
            cats = [logits.argmax(-1) for logits in categories]
            mask = (mask_logits >= 0).float()
        categorical = (torch.stack(cats, -1) if cats else
                       torch.empty((*latent.shape[:-1], 0), dtype=torch.long, device=latent.device))
        return {'x_num': numeric, 'x_cat': categorical, 'mask': mask}

    def decoder_audit(self, reset: bool = False) -> dict:
        return {'codec': 'original EventAE MLP, identical parameter shapes',
                'numeric_sampling': 'deterministic decoder mean in quantile coordinates',
                'numeric_restoration': 'external train-only inverse quantile; no negative-to-zero clipping',
                'categorical_sampling': 'Categorical logits', 'missingness_sampling': 'Bernoulli mask',
                'extra_numeric_noise': False, 'signed_log_projection': False}
