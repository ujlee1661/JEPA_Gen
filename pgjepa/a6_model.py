"""A6 joint marked-path diffusion with a probabilistic predictive-law prior.

The code is a finite global/regime description, not a lossless future path.
Cardinality is sampled, and event-level uncertainty comes from joint diffusion.
There is no autoregressive feedback of generated rows.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .stable_diffusion import v_target, v_to_x0_epsilon, zero_terminal_snr


def _log_sub_exp(large: Tensor, small: Tensor) -> Tensor:
    difference = (small-large).clamp_max(0)
    return large+torch.log(-torch.expm1(difference))


def discretized_count_log_mass(count: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Log mass for N=max(round(exp(U)-1),0), U~Normal(mean,scale).

    Batch count may be [B] and parameter matrices [B,K]. No population-specific
    minimum/maximum is imposed. The zero bin integrates all U<log(1.5).
    """
    if bool((count < 0).any()) or bool((count != count.round()).any()):
        raise ValueError('Nonnegative integer cardinalities required')
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError('Finite Gaussian parameters and positive scales required')
    dtype = mean.dtype
    count = count.to(device=mean.device, dtype=torch.float64)
    if count.ndim+1 == mean.ndim:
        count = count.unsqueeze(-1)
    mu, sigma = mean.double(), scale.double()
    lower = ((count+.5).log()-mu)/sigma
    upper = ((count+1.5).log()-mu)/sigma
    use_sf = lower > 0
    # Mask inputs to the unused subtraction too: an unused log(0) derivative
    # could otherwise poison gradients through torch.where.
    cdf_lower = torch.where(use_sf, torch.full_like(lower, -1), lower)
    cdf_upper = torch.where(use_sf, torch.zeros_like(upper), upper)
    sf_lower = torch.where(use_sf, lower, torch.zeros_like(lower))
    sf_upper = torch.where(use_sf, upper, torch.ones_like(upper))
    cdf = _log_sub_exp(torch.special.log_ndtr(cdf_upper), torch.special.log_ndtr(cdf_lower))
    sf = _log_sub_exp(torch.special.log_ndtr(-sf_lower), torch.special.log_ndtr(-sf_upper))
    positive = torch.where(use_sf, sf, cdf)
    result = torch.where(count == 0, torch.special.log_ndtr(upper), positive)
    return result.to(dtype)


class JointPathDenoiser(nn.Module):
    def __init__(self, token_dim: int, code_dim: int, attrs_dim: int,
                 width: int = 128, layers: int = 4, heads: int = 4):
        super().__init__()
        self.token_dim, self.width = token_dim, width
        self.input = nn.Linear(token_dim, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Sequential(nn.Linear(code_dim+attrs_dim+4, width), nn.SiLU(), nn.Linear(width, width))
        self.noise_time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_dim))

    def forward(self, noisy: Tensor, step: Tensor, valid: Tensor, code: Tensor,
                attrs: Tensor, calendar: Tensor, count: Tensor) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[-1] != self.token_dim or valid.shape != noisy.shape[:2]:
            raise ValueError('Noisy tokens and validity layout disagree')
        if step.shape != (len(noisy),) or count.shape != (len(noisy),):
            raise ValueError('One step and cardinality per path required')
        if not torch.equal(valid.sum(1), count.long()):
            raise ValueError('Validity mask must match supplied cardinality')
        if noisy.shape[1] == 0:
            return noisy*0+self.input.weight.sum()*0
        selected = torch.nonzero(valid.any(1), as_tuple=False).flatten()
        output = torch.zeros_like(noisy)
        if not len(selected):
            return output+self.input.weight.sum()*0
        noisy = torch.where(valid[..., None], noisy, torch.zeros_like(noisy))
        positions = torch.arange(noisy.shape[1], device=noisy.device, dtype=noisy.dtype)[None]
        rank = positions/(count.to(noisy.dtype)[:, None]-1).clamp_min(1)
        position = torch.stack([rank.expand(len(noisy), -1), torch.sin(math.pi*rank),
                                torch.cos(math.pi*rank)], -1)
        frequencies = torch.exp(torch.arange(32, device=noisy.device, dtype=noisy.dtype)*(-math.log(10000.)/31))
        angle = step.to(noisy.dtype)[:, None]*frequencies[None]
        noise_embedding = self.noise_time(torch.cat([angle.sin(), angle.cos()], -1))
        condition = self.condition(torch.cat([code, attrs, calendar,
                                              count.to(noisy.dtype).log1p()[:, None]], -1))+noise_embedding
        hidden = self.input(noisy)+self.position(position)+condition[:, None]
        transformed = self.transformer(hidden[selected], src_key_padding_mask=~valid[selected])
        predicted = self.output(transformed)
        output = output.index_copy(0, selected, predicted)
        return torch.where(valid[..., None], output, torch.zeros_like(output))


class A6Generator(nn.Module):
    model_name = 'A6 joint-path predictive-law generator'

    def __init__(self, token_dim: int, attrs_dim: int, code_dim: int,
                 observation_end: float, width: int = 128, layers: int = 4,
                 heads: int = 4, hidden: int = 128, components: int = 8,
                 diffusion_steps: int = 100):
        super().__init__()
        if min(token_dim, code_dim, components, hidden, diffusion_steps) < 1 or observation_end <= 0:
            raise ValueError('Positive dimensions, noise levels and observation end required')
        if attrs_dim < 0 or width % heads or diffusion_steps < 2:
            raise ValueError('Invalid attribute, attention or diffusion dimensions')
        self.token_dim, self.attrs_dim, self.code_dim = token_dim, attrs_dim, code_dim
        self.hidden_dim, self.components, self.diffusion_steps = hidden, components, diffusion_steps
        self.past_projection = nn.Sequential(nn.Linear(token_dim, 64), nn.SiLU())
        self.static_initial = nn.Sequential(nn.Linear(max(1, attrs_dim), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.past_gru = nn.GRU(64, hidden, batch_first=True)
        self.prior = nn.Sequential(nn.Linear(hidden+attrs_dim+3, 256), nn.SiLU(),
                                   nn.Linear(256, 256), nn.SiLU(),
                                   nn.Linear(256, components*(3+2*code_dim)))
        self.decoder = JointPathDenoiser(token_dim, code_dim, attrs_dim, width, layers, heads)
        self.register_buffer('observation_end', torch.tensor(observation_end, dtype=torch.float64))
        grid = torch.arange(diffusion_steps+1, dtype=torch.float64)/diffusion_steps
        alpha = torch.cos((grid+.008)/1.008*math.pi/2).square()
        alpha = alpha/alpha[0]
        beta = (1-alpha[1:]/alpha[:-1]).clamp(max=.1)
        self.register_buffer('alpha_bar', zero_terminal_snr(torch.cumprod(1-beta, 0).float()))

    def _calendar(self, attrs: Tensor, anchor: Tensor, delta: Tensor) -> Tensor:
        anchor = torch.as_tensor(anchor, device=attrs.device, dtype=attrs.dtype).reshape(-1)
        delta = torch.as_tensor(delta, device=attrs.device, dtype=attrs.dtype).reshape(-1)
        if len(anchor) != len(attrs) or len(delta) != len(attrs):
            raise ValueError('One calendar anchor and horizon per path required')
        if not bool(torch.isfinite(anchor).all()) or not bool(torch.isfinite(delta).all()) or bool((anchor < 0).any()) or bool((delta < 0).any()):
            raise ValueError('Finite nonnegative anchor and horizon required')
        end = self.observation_end.to(attrs)
        tolerance = 8*torch.finfo(attrs.dtype).eps*end
        if bool((anchor+delta > end+tolerance).any()):
            raise ValueError('Query exceeds the known finite observation window')
        return torch.stack([anchor.log1p()/end.log1p(), delta.log1p()/end.log1p(), delta/end], -1)

    def encode_past(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor) -> Tensor:
        if past_tokens.ndim != 3 or past_tokens.shape[-1] != self.token_dim or past_valid.shape != past_tokens.shape[:2]:
            raise ValueError('Past token and validity shapes disagree')
        if attrs.shape != (len(past_tokens), self.attrs_dim) or not bool(torch.isfinite(attrs).all()):
            raise ValueError('Static attributes differ from model schema')
        lengths = past_valid.long().sum(1)
        expected = torch.arange(past_tokens.shape[1], device=past_tokens.device)[None] < lengths[:, None]
        if not torch.equal(past_valid.bool(), expected):
            raise ValueError('Past histories must be contiguous prefixes')
        static = attrs if self.attrs_dim else attrs.new_zeros(len(attrs), 1)
        initial = self.static_initial(static)
        selected = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if not len(selected):
            return initial
        tokens = torch.where(past_valid[..., None], past_tokens, torch.zeros_like(past_tokens))
        embedded = self.past_projection(tokens[selected])
        packed = pack_padded_sequence(embedded, lengths[selected].cpu(), batch_first=True, enforce_sorted=False)
        _, final = self.past_gru(packed, initial[selected][None])
        return initial.index_copy(0, selected, final[0])

    def prior_parameters(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor,
                         anchor: Tensor, delta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state = self.encode_past(past_tokens, past_valid, attrs)
        calendar = self._calendar(attrs, anchor, delta)
        raw = self.prior(torch.cat([state, attrs, calendar], -1)).reshape(len(attrs), self.components, 3+2*self.code_dim)
        logits = raw[..., 0]
        count_mean, count_scale = raw[..., 1], F.softplus(raw[..., 2])+.03
        code_mean = raw[..., 3:3+self.code_dim]
        code_scale = F.softplus(raw[..., 3+self.code_dim:])+.03
        return logits, count_mean, count_scale, code_mean, code_scale

    def prior_terms(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor,
                    anchor: Tensor, delta: Tensor, count: Tensor, code: Tensor) -> Tensor:
        logits, count_mu, count_sigma, code_mu, code_sigma = self.prior_parameters(past_tokens, past_valid, attrs, anchor, delta)
        if code.shape != (len(attrs), self.code_dim) or not bool(torch.isfinite(code).all()):
            raise ValueError('Finite target future codes required')
        count_log = discretized_count_log_mass(count, count_mu, count_sigma)
        code_log = (-.5*((code[:, None]-code_mu)/code_sigma).square()-code_sigma.log()-.5*math.log(2*math.pi)).sum(-1)
        return -torch.logsumexp(F.log_softmax(logits, -1)+count_log+code_log, -1)/(self.code_dim+1)

    def nll_prior(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor,
                  anchor: Tensor, delta: Tensor, count: Tensor, code: Tensor) -> tuple[Tensor, dict[str, float]]:
        terms = self.prior_terms(past_tokens, past_valid, attrs, anchor, delta, count, code)
        loss = terms.mean()
        return loss, {'prior_nll_per_dimension': float(loss.detach()), 'prior_paths': float(len(attrs))}

    @torch.no_grad()
    def sample_prior(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor,
                     anchor: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        logits, count_mu, count_sigma, code_mu, code_sigma = self.prior_parameters(past_tokens, past_valid, attrs, anchor, delta)
        component = torch.distributions.Categorical(logits=logits).sample()
        rows = torch.arange(len(attrs), device=attrs.device)
        log_count = count_mu[rows, component]+count_sigma[rows, component]*torch.randn(len(attrs), device=attrs.device)
        # Upper numeric atom is far beyond the safety cap and is an invalid
        # generation, never a silently truncated valid cardinality.
        count = (log_count.clamp(-30, 30).exp()-.5).floor().clamp_min(0).long()
        code = code_mu[rows, component]+code_sigma[rows, component]*torch.randn(len(attrs), self.code_dim, device=attrs.device)
        return count, code

    def diffusion_terms(self, future_tokens: Tensor, valid: Tensor, code: Tensor,
                        attrs: Tensor, anchor: Tensor, delta: Tensor, count: Tensor,
                        *, noise: Tensor | None = None, steps: Tensor | None = None) -> Tensor:
        if not torch.equal(valid.sum(1), count.long()):
            raise ValueError('Future path cardinalities and validity differ')
        if steps is None:
            steps = torch.randint(self.diffusion_steps, (len(attrs),), device=attrs.device)
        if noise is None:
            noise = torch.randn_like(future_tokens)
        clean = torch.where(valid[..., None], future_tokens, torch.zeros_like(future_tokens))
        alpha = self.alpha_bar[steps][:, None, None]
        noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        prediction = self.decoder(noisy, steps, valid, code, attrs, self._calendar(attrs, anchor, delta), count)
        target = v_target(clean, noise, alpha)
        squared = torch.where(valid[..., None], (prediction-target).square(), torch.zeros_like(prediction))
        divisor = (count*self.token_dim).clamp_min(1).to(squared)
        return squared.sum((1, 2))/divisor

    def diffusion_loss(self, future_tokens: Tensor, valid: Tensor, code: Tensor,
                       attrs: Tensor, anchor: Tensor, delta: Tensor, count: Tensor,
                       *, noise: Tensor | None = None, steps: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        terms = self.diffusion_terms(future_tokens, valid, code, attrs, anchor, delta, count, noise=noise, steps=steps)
        loss = terms.mean()
        return loss, {'path_diffusion_loss': float(loss.detach()), 'nonempty_paths': float((count > 0).sum())}

    @torch.no_grad()
    def sample_paths(self, past_tokens: Tensor, past_valid: Tensor, attrs: Tensor,
                     anchor: Tensor, delta: Tensor, *, sample_steps: int = 50,
                     max_events: int = 4096, count: Tensor | None = None,
                     code: Tensor | None = None) -> dict[str, Tensor]:
        if sample_steps < 1 or max_events < 1:
            raise ValueError('Positive sampling steps and safety cap required')
        if (count is None) != (code is None):
            raise ValueError('Supply both count and code, or sample both from prior')
        if count is None:
            count, code = self.sample_prior(past_tokens, past_valid, attrs, anchor, delta)
        if code is None:
            raise RuntimeError('Missing future code')
        if bool((count > max_events).any()):
            raise RuntimeError(f'Predicted cardinality exceeds safety cap: {int(count.max())}>{max_events}')
        width = int(count.max()) if len(count) else 0
        valid = torch.arange(width, device=attrs.device)[None] < count[:, None]
        noisy = torch.randn(len(attrs), width, self.token_dim, device=attrs.device)
        calendar = self._calendar(attrs, anchor, delta)
        schedule = torch.linspace(self.diffusion_steps-1, 0, min(sample_steps, self.diffusion_steps), device=attrs.device).round().long()
        for index, step in enumerate(schedule):
            velocity = self.decoder(noisy, step.expand(len(attrs)), valid, code, attrs, calendar, count)
            clean, epsilon = v_to_x0_epsilon(noisy, velocity, self.alpha_bar[step])
            if index+1 < len(schedule):
                alpha = self.alpha_bar[schedule[index+1]]
                noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*epsilon
            else:
                noisy = clean
        return {'tokens':torch.where(valid[..., None], noisy, torch.zeros_like(noisy)), 'valid':valid,
                'count':count, 'code':code, 'cap_hit':torch.zeros(len(attrs),dtype=torch.bool,device=attrs.device)}

    def set_trainable(self, part: str) -> None:
        if part not in {'all', 'prior', 'decoder'}:
            raise ValueError('Trainable partition must be all/prior/decoder')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(part == 'all' or (name.startswith('decoder.') == (part == 'decoder')))

    def parameter_counts(self) -> dict[str, int]:
        return {'total':sum(p.numel() for p in self.parameters()),
                'prior':sum(p.numel() for name,p in self.named_parameters() if not name.startswith('decoder.')),
                'decoder':sum(p.numel() for p in self.decoder.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
