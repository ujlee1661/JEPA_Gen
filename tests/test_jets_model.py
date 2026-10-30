"""Standalone interface and information-boundary tests for JETS."""
import pytest
import torch
from jets.models import JETSGenerator
from jets.encoders import TimeTargetEncoder, FeatureTargetEncoder


def model():
    torch.manual_seed(10)
    return JETSGenerator(5, 2, 10., code_dim=8, width=16, layers=1,
                         heads=2, hidden=16, components=2, diffusion_steps=10)


def batch():
    return {'tokens': torch.randn(2, 4, 5),
            'valid': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
            'event_valid': torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
            'active': torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
            'count': torch.tensor([3, 2]), 'attrs': torch.ones(2, 2),
            'anchor': torch.zeros(2, dtype=torch.float64),
            'delta': torch.ones(2, dtype=torch.float64),
            'past_tokens': torch.empty(2, 0, 5),
            'past_valid': torch.empty(2, 0, dtype=torch.bool)}


def test_sampling_does_not_accept_real_future_codes_and_replays():
    m = model().eval(); m.stochastic_condition = True
    b = batch(); attrs = b['attrs']; a = b['anchor']; d = b['delta']
    with pytest.raises(TypeError):
        m.sample_paths(attrs, a, d, teacher_codes=torch.zeros(2, 8))
    with torch.no_grad():
        torch.manual_seed(42); x = m.sample_paths(attrs, a, d, sample_steps=10)
        torch.manual_seed(42); y = m.sample_paths(attrs, a, d, sample_steps=10)
    assert torch.equal(x['tokens'], y['tokens'])
    assert torch.equal(x['count'], y['count'])
    assert bool(torch.isfinite(x['tokens']).all())


def test_disabled_feature_aux_ignores_teacher_target():
    m = model(); b = batch(); noise = torch.randn_like(b['tokens'])
    steps = torch.tensor([0, 9]); t = torch.randn(2, 4, 8)
    feature = torch.randn_like(t)
    torch.manual_seed(27)
    first, _ = m.loss(b, t, t[:, 0], 1., feature_teacher_codes=feature,
                      feature_aux_weight=0., noise=noise, steps=steps)
    torch.manual_seed(27)
    second, _ = m.loss(b, t, t[:, 0], 1., feature_teacher_codes=feature+100,
                       feature_aux_weight=0., noise=noise, steps=steps)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_time_target_encoder_does_not_read_future_marks():
    target_encoder = TimeTargetEncoder(5, 2, 10., code_dim=8, width=16, layers=1, heads=2).eval()
    original = batch()
    altered = dict(original)
    altered['tokens'] = original['tokens'].clone()
    altered['tokens'][..., 1:] += 100.
    with torch.no_grad():
        torch.testing.assert_close(target_encoder.encode(original), target_encoder.encode(altered), rtol=0, atol=0)
        first, _ = target_encoder.loss(original)
        second, _ = target_encoder.loss(altered)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_feature_target_decoder_does_not_receive_clean_marks():
    target_encoder = FeatureTargetEncoder(5, 2, 10., code_dim=8, width=16, layers=1,
                                          heads=2, decoder_layers=1).eval()
    original = batch()
    altered = dict(original)
    altered['tokens'] = original['tokens'].clone()
    altered['tokens'][..., 1:] = float('nan')
    z = torch.randn(2, 8)
    global_noise = torch.randn_like(z)
    local_noise = torch.randn(2, 4, 5)
    with torch.no_grad():
        first = target_encoder.sample_reconstruction(z, original, global_noise, local_noise)
        second = target_encoder.sample_reconstruction(z, altered, global_noise, local_noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_one_condition_draw_per_generated_path():
    m = model().eval(); m.stochastic_condition = True
    seen = []
    hook = m.feature_residual.register_forward_hook(
        lambda module, inputs, output: seen.append(m._condition_epsilon.clone()))
    b = batch()
    with torch.no_grad():
        torch.manual_seed(108)
        m.sample_paths(b['attrs'], b['anchor'], b['delta'], sample_steps=4)
    hook.remove()
    assert len(seen) == 4 and all(torch.equal(seen[0], value) for value in seen[1:])
    assert not torch.equal(seen[0][0], seen[0][1])
    assert m._condition_epsilon is None
