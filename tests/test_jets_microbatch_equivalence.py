"""Diagnostic only: fixed logical batch vs weighted microbatch gradients.

No training runner or optimizer is changed. Floating-point summation order can
produce small differences; this test is not a bitwise identity guarantee.
"""
import copy

import pytest
import torch

from jets.encoders import FeatureTargetEncoder, TimeTargetEncoder
from jets.models import JETSGenerator


def _batch():
    torch.manual_seed(19)
    return {'tokens': torch.randn(2, 4, 5),
            'valid': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
            'event_valid': torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
            'active': torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
            'count': torch.tensor([3, 2]), 'attrs': torch.ones(2, 2),
            'anchor': torch.zeros(2, dtype=torch.float64),
            'delta': torch.ones(2, dtype=torch.float64),
            'past_tokens': torch.empty(2, 0, 5), 'past_valid': torch.empty(2, 0, dtype=torch.bool)}


def _slice(batch, index):
    return {key: value[index:index+1] for key, value in batch.items()}


def _compare_gradients(full, micro):
    for (left_name, left), (right_name, right) in zip(full.named_parameters(), micro.named_parameters()):
        assert left_name == right_name
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize('kind', ['generator', 'time', 'feature'])
def test_fixed_logical_batch_microgradient_matches(kind):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    batch = _batch()
    if kind == 'generator':
        full = JETSGenerator(5, 2, 10., code_dim=8, width=16, layers=1, heads=2,
                             hidden=16, components=2, diffusion_steps=10)
        full.stochastic_condition = False
    elif kind == 'time':
        full = TimeTargetEncoder(5, 2, 10., code_dim=8, width=16, layers=1, heads=2)
    else:
        full = FeatureTargetEncoder(5, 2, 10., code_dim=8, width=16,
                                    layers=1, heads=2, decoder_layers=1)
    full.train()
    micro = copy.deepcopy(full)
    # Generate all randomness at logical-batch shape once, then slice; this is
    # essential for preserving input schedules when changing microbatch size.
    time_target = torch.randn(2, 4, 8)
    feature_target = torch.randn(2, 4, 8)
    steps = torch.tensor([0, 9])
    noise = torch.randn_like(batch['tokens'])
    global_pair = torch.randn(2, 2, 8)
    local_pair = torch.randn(2, 2, 4, 5)

    def loss(model, data, index=None):
        section = slice(None) if index is None else slice(index, index+1)
        if kind == 'generator':
            return model.loss(data, time_target[section], time_target[section, 0], 1.,
                feature_teacher_codes=feature_target[section], feature_aux_weight=1.,
                steps=steps[section], noise=noise[section])[0]
        if kind == 'time':
            return model.loss(data)[0]
        return model.loss(data, objective='energy', global_noise_pair=global_pair[:, section],
                          local_noise_pair=local_pair[:, section])[0]

    whole = loss(full, batch)
    whole.backward()
    total = sum((loss(micro, _slice(batch, i), i) / 2 for i in range(2)))
    total.backward()
    torch.testing.assert_close(whole, total, rtol=2e-5, atol=2e-6)
    _compare_gradients(full, micro)
