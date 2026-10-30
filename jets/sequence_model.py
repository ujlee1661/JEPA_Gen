"""Native spacing diffusion with an active predictive-prefix branch.

An observed-history GRU uses absolute-time tokens. A separate predictor GRU
sees noisy CLR-prefix tokens whose gauge is window-global. Frozen learned
suffix targets enter auxiliary scores only. Generation samples N, its simplex
face, and a joint path; generated rows are not fed back autoregressively.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .count import discretized_count_log_mass
from .spacing import face_log_mass, project_clr, sample_face
from .diffusion import v_target, v_to_x0_epsilon, zero_terminal_snr


class SequenceGenerator(nn.Module):
    model_name = 'JETS joint sequence diffusion'

    def __init__(self, token_dim: int, attrs_dim: int, observation_end: float,
                 time_scale: float = 1., code_dim: int = 64, width: int = 128,
                 layers: int = 4, heads: int = 4, hidden: int = 128,
                 components: int = 8, diffusion_steps: int = 100):
        super().__init__()
        if min(token_dim, code_dim, width, layers, heads, hidden, components) < 1:
            raise ValueError('Positive dimensions required')
        if attrs_dim < 0 or width % heads or diffusion_steps < 2:
            raise ValueError('Invalid static, attention or diffusion dimensions')
        if not math.isfinite(time_scale) or time_scale <= 0 or not math.isfinite(observation_end) or observation_end <= 0:
            raise ValueError('Finite positive time scale and observation end required')
        self.token_dim, self.attrs_dim, self.code_dim = token_dim, attrs_dim, code_dim
        self.hidden_dim, self.components, self.diffusion_steps = hidden, components, diffusion_steps
        self.register_buffer('time_scale', torch.tensor(time_scale, dtype=torch.float64))
        self.register_buffer('observation_end', torch.tensor(observation_end, dtype=torch.float64))
        self.noise_embedding = nn.Sequential(nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # attrs, known anchor/horizon, log N and N-known flag. Unknown N is
        # represented only by the flag; the target N never enters that branch.
        self.initial = nn.Sequential(nn.Linear(attrs_dim+5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # Actual observed history uses the observed absolute-time/mark
        # tokens, so it has a separate projection from noisy future CLR tokens.
        self.history_projection = nn.Sequential(nn.Linear(token_dim, 64), nn.SiLU())
        self.history_gru = nn.GRU(64, hidden, batch_first=True)
        self.history_condition = nn.Linear(hidden, hidden, bias=False)
        self.prefix_input = nn.Sequential(nn.Linear(token_dim+2, 64), nn.SiLU())
        self.prefix_gru = nn.GRU(64, hidden, batch_first=True)
        self.prediction = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2*code_dim))
        self.predictive_condition = nn.Sequential(nn.Linear(2*code_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.count_head = nn.Sequential(nn.Linear(2*code_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3*components))
        self.face_head = nn.Sequential(nn.Linear(2*code_dim+3, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.token_input = nn.Linear(token_dim+2, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.outer_condition = nn.Sequential(nn.Linear(attrs_dim+4, width), nn.SiLU(), nn.Linear(width, width))
        self.noise_condition = nn.Linear(hidden, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_dim))
        grid = torch.arange(diffusion_steps+1, dtype=torch.float64)/diffusion_steps
        alpha = torch.cos((grid+.008)/1.008*math.pi/2).square()
        alpha = alpha/alpha[0]
        beta = (1-alpha[1:]/alpha[:-1]).clamp(max=.1)
        self.register_buffer('alpha_bar', zero_terminal_snr(torch.cumprod(1-beta, 0).float()))

    def _calendar(self, attrs: Tensor, anchor: Tensor, delta: Tensor) -> Tensor:
        if attrs.ndim != 2 or attrs.shape[1] != self.attrs_dim or not bool(torch.isfinite(attrs).all()):
            raise ValueError('Finite static attributes with declared schema required')
        anchor = torch.as_tensor(anchor, device=attrs.device, dtype=torch.float64).reshape(-1)
        delta = torch.as_tensor(delta, device=attrs.device, dtype=torch.float64).reshape(-1)
        if len(anchor) != len(attrs) or len(delta) != len(attrs):
            raise ValueError('One anchor and horizon per window required')
        if not bool(torch.isfinite(anchor).all() and torch.isfinite(delta).all()) or bool((anchor < 0).any() or (delta < 0).any()):
            raise ValueError('Finite nonnegative window calendar required')
        end = self.observation_end
        if bool((anchor+delta > end+8*torch.finfo(torch.float64).eps*end).any()):
            raise ValueError('Window exceeds known observation end')
        return torch.stack([anchor.log1p()/end.log1p(), delta.log1p()/end.log1p(), delta/end], -1).to(attrs)

    def _noise(self, steps: Tensor, dtype: torch.dtype) -> Tensor:
        frequency = torch.exp(torch.arange(32, device=steps.device, dtype=dtype)*(-math.log(10000.)/31))
        angle = steps.to(dtype)[:, None]*frequency[None]
        return self.noise_embedding(torch.cat([angle.sin(), angle.cos()], -1))

    def _initial(self, attrs: Tensor, calendar: Tensor, steps: Tensor,
                 count: Tensor | None, history: Tensor) -> Tensor:
        if count is None:
            cardinality = attrs.new_zeros(len(attrs), 2)
        else:
            cardinality = torch.stack([count.to(attrs).log1p(), torch.ones_like(count).to(attrs)], -1)
        return (self.initial(torch.cat([attrs, calendar, cardinality], -1))
                +self._noise(steps, attrs.dtype)+self.history_condition(history))

    def encode_history(self, attrs: Tensor, past_tokens: Tensor | None = None,
                       past_valid: Tensor | None = None) -> Tensor:
        """Encode observed absolute-time tokens only; empty is the BOS case."""
        if (past_tokens is None) != (past_valid is None):
            raise ValueError('Observed history requires both tokens and valid mask')
        initial = attrs.new_zeros(len(attrs), self.hidden_dim)
        if past_tokens is None:
            return initial
        if past_tokens.ndim != 3 or past_tokens.shape[0] != len(attrs) or past_tokens.shape[-1] != self.token_dim:
            raise ValueError('Observed history token schema differs from model')
        if past_valid is None or past_valid.shape != past_tokens.shape[:2]:
            raise ValueError('Observed history mask shape differs')
        lengths = past_valid.long().sum(-1)
        index = torch.arange(past_tokens.shape[1], device=attrs.device)[None]
        if not torch.equal(past_valid.bool(), index < lengths[:, None]):
            raise ValueError('Observed histories must be contiguous valid prefixes')
        if not bool(torch.isfinite(torch.where(past_valid[..., None], past_tokens, 0.)).all()):
            raise ValueError('Observed history must contain finite valid tokens')
        selected = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if not len(selected):
            return initial
        clean = torch.where(past_valid[..., None], past_tokens, torch.zeros_like(past_tokens))
        embedded = self.history_projection(clean[selected])
        packed = pack_padded_sequence(embedded, lengths[selected].cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.history_gru(packed, initial[selected][None])
        return initial.index_copy(0, selected, state[0])

    def _distribution(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.prediction(state)
        mean, scale_raw = raw.split(self.code_dim, -1)
        scale = F.softplus(scale_raw)+.03
        return mean, scale, torch.cat([mean, scale], -1)

    def _position(self, count: Tensor, length: int, dtype: torch.dtype) -> Tensor:
        # Indices 0..N include the explicit terminal gap/token.
        rank = torch.arange(length, device=count.device, dtype=dtype)[None]/count[:, None].to(dtype).clamp_min(1)
        return torch.stack([rank, (math.pi*rank).sin(), (math.pi*rank).cos()], -1)

    def _validate(self, batch: dict, tokens: Tensor | None = None) -> None:
        tokens = batch['tokens'] if tokens is None else tokens
        count, valid, event, active = (batch[k] for k in ('count','valid','event_valid','active'))
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim or tokens.shape[1] < 1:
            raise ValueError('N+1 token layout required')
        if count.shape != (len(tokens),) or bool((count < 0).any()) or bool((count != count.round()).any()):
            raise ValueError('Nonnegative integer counts required')
        index = torch.arange(tokens.shape[1], device=tokens.device)[None]
        if not torch.equal(valid, index <= count[:, None]) or not torch.equal(event, index < count[:, None]):
            raise ValueError('Valid and event masks must encode N+1 and N contiguous positions')
        if active.shape != valid.shape or bool((active & ~valid).any()):
            raise ValueError('Active face extends outside valid gap positions')
        positive = torch.as_tensor(batch['delta'], device=tokens.device) > 0
        if bool((active.any(-1) != positive).any()):
            raise ValueError('Positive-duration faces nonempty; zero-duration faces empty')
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError('Finite tokens required')

    def project_tokens(self, tokens: Tensor, active: Tensor, event_valid: Tensor) -> Tensor:
        time = project_clr(tokens[..., 0], active)
        marks = torch.where(event_valid[..., None], tokens[..., 1:], torch.zeros_like(tokens[..., 1:]))
        return torch.cat([time[..., None], marks], -1)

    def count_parameters(self, attrs: Tensor, anchor: Tensor, delta: Tensor, *,
                         past_tokens: Tensor | None = None,
                         past_valid: Tensor | None = None) -> dict[str, Tensor]:
        calendar = self._calendar(attrs, anchor, delta)
        steps = torch.full((len(attrs),), self.diffusion_steps-1, device=attrs.device, dtype=torch.long)
        state = self._initial(attrs, calendar, steps, None, self.encode_history(attrs, past_tokens, past_valid))
        mean, scale, features = self._distribution(state)
        raw = self.count_head(features).reshape(len(attrs), self.components, 3)
        return {'logits':raw[..., 0], 'mean':raw[..., 1], 'scale':F.softplus(raw[..., 2])+.03,
                'prediction_mean':mean, 'prediction_scale':scale}

    def count_log_mass(self, count: Tensor, attrs: Tensor, anchor: Tensor, delta: Tensor, *,
                       past_tokens: Tensor | None = None,
                       past_valid: Tensor | None = None) -> Tensor:
        parameters = self.count_parameters(attrs, anchor, delta, past_tokens=past_tokens, past_valid=past_valid)
        component = discretized_count_log_mass(count, parameters['mean'], parameters['scale'])
        return torch.logsumexp(F.log_softmax(parameters['logits'], -1)+component, -1)

    def face_logits(self, attrs: Tensor, anchor: Tensor, delta: Tensor, count: Tensor,
                    length: int | None = None, *, past_tokens: Tensor | None = None,
                    past_valid: Tensor | None = None) -> Tensor:
        calendar = self._calendar(attrs, anchor, delta)
        step = torch.full((len(attrs),), self.diffusion_steps-1, device=attrs.device, dtype=torch.long)
        state = self._initial(attrs, calendar, step, count, self.encode_history(attrs, past_tokens, past_valid))
        _, _, feature = self._distribution(state)
        length = int(count.max())+1 if length is None else length
        positions = self._position(count, length, attrs.dtype)
        return self.face_head(torch.cat([feature[:, None].expand(-1, length, -1), positions], -1))[..., 0]

    def forward_velocity(self, noisy: Tensor, steps: Tensor, batch: dict) -> dict[str, Tensor]:
        self._validate(batch, noisy)
        if steps.shape != (len(noisy),) or bool((steps < 0).any() or (steps >= self.diffusion_steps).any()):
            raise ValueError('One valid diffusion step per window required')
        attrs, count = batch['attrs'], batch['count']
        calendar = self._calendar(attrs, batch['anchor'], batch['delta'])
        projected = self.project_tokens(noisy, batch['active'], batch['event_valid'])
        flags = torch.stack([batch['active'], batch['event_valid']], -1).to(noisy)
        inputs = torch.cat([projected, flags], -1)
        history = self.encode_history(attrs, batch.get('past_tokens'), batch.get('past_valid'))
        initial = self._initial(attrs, calendar, steps, count, history)
        states, _ = self.prefix_gru(self.prefix_input(inputs), initial[None])
        # Prediction at index i only reads noisy tokens strictly before i.
        prefix = torch.cat([initial[:, None], states[:, :-1]], 1)
        mean, scale, features = self._distribution(prefix)
        outer = self.outer_condition(torch.cat([attrs, calendar, count.to(attrs).log1p()[:, None]], -1))
        outer = outer+self.noise_condition(self._noise(steps, noisy.dtype))
        hidden = (self.token_input(inputs)+self.position(self._position(count, noisy.shape[1], noisy.dtype))
                  +self.predictive_condition(features)+outer[:, None])
        hidden = self.transformer(hidden, src_key_padding_mask=~batch['valid'])
        velocity = self.project_tokens(self.output(hidden), batch['active'], batch['event_valid'])
        return {'velocity':velocity, 'prefix_mean':mean, 'prefix_scale':scale,
                'prefix_state':prefix}

    def forward(self, noisy: Tensor, steps: Tensor, batch: dict) -> dict[str, Tensor]:
        return self.forward_velocity(noisy, steps, batch)

    @staticmethod
    def _normal_terms(target: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        if target.shape != mean.shape or not bool(torch.isfinite(target).all()):
            raise ValueError('Finite frozen teacher embeddings matching prefix layout required')
        return (.5*((target.detach()-mean)/scale).square()+scale.log()+.5*math.log(2*math.pi)).mean(-1)

    def loss(self, batch: dict, teacher_codes: Tensor | None = None,
             bos_teacher_codes: Tensor | None = None, aux_weight: float = 0.,
             *, noise: Tensor | None = None, steps: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
        self._validate(batch)
        if not math.isfinite(aux_weight) or aux_weight < 0:
            raise ValueError('Finite nonnegative auxiliary weight required')
        clean = self.project_tokens(batch['tokens'], batch['active'], batch['event_valid'])
        if steps is None:
            steps = torch.randint(self.diffusion_steps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError('Noise shape differs from clean token layout')
        noise = self.project_tokens(noise, batch['active'], batch['event_valid'])
        alpha = self.alpha_bar[steps][:, None, None]
        noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        result = self.forward_velocity(noisy, steps, batch)
        target = v_target(clean, noise, alpha)
        squared = (result['velocity']-target).square().sum((1, 2))
        dof = (batch['active'].sum(-1)-1).clamp_min(0)+batch['count']*(self.token_dim-1)
        continuous = (squared/dof.clamp_min(1).to(squared)).mean()
        history = {'past_tokens':batch.get('past_tokens'), 'past_valid':batch.get('past_valid')}
        count_parameters = self.count_parameters(batch['attrs'], batch['anchor'], batch['delta'], **history)
        count_terms = discretized_count_log_mass(batch['count'], count_parameters['mean'], count_parameters['scale'])
        count_loss = -torch.logsumexp(F.log_softmax(count_parameters['logits'], -1)+count_terms, -1).mean()
        logits = self.face_logits(batch['attrs'], batch['anchor'], batch['delta'], batch['count'], clean.shape[1], **history)
        positive = torch.as_tensor(batch['delta'], device=clean.device) > 0
        face_terms = clean.new_zeros(len(clean), dtype=torch.float64)
        if bool(positive.any()):
            face_terms[positive] = -face_log_mass(logits[positive], batch['active'][positive], batch['valid'][positive])
        face_loss = face_terms.mean().to(continuous)
        auxiliary = continuous*0
        bos_aux = continuous*0
        if teacher_codes is not None:
            terms = self._normal_terms(teacher_codes, result['prefix_mean'], result['prefix_scale'])
            auxiliary = ((terms*batch['valid']).sum(-1)/batch['valid'].sum(-1)).mean()
            if bos_teacher_codes is None:
                bos_teacher_codes = teacher_codes[:, 0]
        if bos_teacher_codes is not None:
            bos_aux = self._normal_terms(bos_teacher_codes, count_parameters['prediction_mean'],
                                         count_parameters['prediction_scale']).mean()
        if aux_weight and (teacher_codes is None or bos_teacher_codes is None):
            raise ValueError('Auxiliary-on training requires prefix and BOS frozen teacher targets')
        generation = continuous+count_loss+face_loss
        total = generation+aux_weight*(auxiliary+bos_aux)/2
        return total, {'loss':float(total.detach()), 'generation_loss':float(generation.detach()),
            'continuous_v_loss':float(continuous.detach()), 'count_nll':float(count_loss.detach()),
            'face_nll':float(face_loss.detach()), 'prefix_aux_nll_per_dim':float(auxiliary.detach()),
            'bos_aux_nll_per_dim':float(bos_aux.detach()), 'aux_weight':float(aux_weight),
            'mean_intrinsic_dimension':float(dof.float().mean())}

    @torch.no_grad()
    def sample_count(self, attrs: Tensor, anchor: Tensor, delta: Tensor, max_events: int = 4096, *,
                     past_tokens: Tensor | None = None,
                     past_valid: Tensor | None = None) -> Tensor:
        parameters = self.count_parameters(attrs, anchor, delta, past_tokens=past_tokens, past_valid=past_valid)
        component = torch.distributions.Categorical(logits=parameters['logits']).sample()
        row = torch.arange(len(attrs), device=attrs.device)
        value = parameters['mean'][row, component]+parameters['scale'][row, component]*torch.randn(len(attrs), device=attrs.device)
        if not bool(torch.isfinite(value).all()) or bool((value >= math.log(max_events+1.5)).any()):
            raise RuntimeError('Predicted cardinality exceeds safety cap; no clipping/resampling')
        return (value.exp()-.5).floor().clamp_min(0).long()

    @torch.no_grad()
    def sample_paths(self, attrs: Tensor, anchor: Tensor, delta: Tensor, *,
                     sample_steps: int = 50, max_events: int = 4096,
                     count: Tensor | None = None, active: Tensor | None = None,
                     noise: Tensor | None = None, past_tokens: Tensor | None = None,
                     past_valid: Tensor | None = None) -> dict[str, Tensor]:
        if sample_steps < 1 or max_events < 0 or not len(attrs):
            raise ValueError('Positive steps, nonnegative cap and nonempty batch required')
        self._calendar(attrs, anchor, delta)
        if count is None:
            if active is not None:
                raise ValueError('A supplied diagnostic face requires a supplied count')
            count = self.sample_count(attrs, anchor, delta, max_events,
                                      past_tokens=past_tokens, past_valid=past_valid)
        count = torch.as_tensor(count, device=attrs.device)
        if count.shape != (len(attrs),) or bool((count < 0).any() or (count != count.round()).any()):
            raise ValueError('One nonnegative integer count per path required')
        if bool((count > max_events).any()):
            raise RuntimeError('Predicted cardinality exceeds safety cap; no clipping/resampling')
        count = count.long()
        length = int(count.max())+1
        index = torch.arange(length, device=attrs.device)[None]
        valid, event = index <= count[:, None], index < count[:, None]
        positive = torch.as_tensor(delta, device=attrs.device) > 0
        if active is None:
            active = torch.zeros_like(valid)
            if bool(positive.any()):
                logits = self.face_logits(attrs, anchor, delta, count, length,
                                           past_tokens=past_tokens, past_valid=past_valid)
                active[positive] = sample_face(logits[positive], valid[positive])
        else:
            active = torch.as_tensor(active, device=attrs.device, dtype=torch.bool)
        shape = (len(attrs), length, self.token_dim)
        if noise is None:
            noise = torch.randn(shape, device=attrs.device, dtype=attrs.dtype)
        if noise.shape != shape:
            raise ValueError('Initial noise differs from sampled path layout')
        batch = {'tokens':noise, 'valid':valid, 'event_valid':event, 'active':active,
                 'attrs':attrs, 'count':count, 'anchor':anchor, 'delta':delta,
                 'past_tokens':past_tokens, 'past_valid':past_valid}
        self._validate(batch)
        noisy = self.project_tokens(noise, active, event)
        schedule = torch.linspace(self.diffusion_steps-1, 0, min(sample_steps, self.diffusion_steps),
                                  device=attrs.device).round().long()
        for offset, step in enumerate(schedule):
            velocity = self.forward_velocity(noisy, step.expand(len(attrs)), batch)['velocity']
            clean, epsilon = v_to_x0_epsilon(noisy, velocity, self.alpha_bar[step])
            clean = self.project_tokens(clean, active, event)
            epsilon = self.project_tokens(epsilon, active, event)
            if offset+1 < len(schedule):
                alpha = self.alpha_bar[schedule[offset+1]]
                noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*epsilon
            else:
                noisy = clean
            noisy = self.project_tokens(noisy, active, event)
        return {'tokens':noisy, 'valid':valid, 'event_valid':event, 'active':active,
                'count':count, 'anchor':torch.as_tensor(anchor, device=attrs.device),
                'delta':torch.as_tensor(delta, device=attrs.device), 'attrs':attrs,
                'cap_hit':torch.zeros(len(attrs), dtype=torch.bool, device=attrs.device)}

    def parameter_counts(self) -> dict[str, int]:
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
