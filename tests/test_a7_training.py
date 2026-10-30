import copy

import numpy as np
import pytest
import torch

from pgjepa.a6_data import encode_window
from pgjepa.a7_data import collate_a7, prepare_a7
from pgjepa.a7_model import A7Generator
from pgjepa.a7_training import (a7_config, cache_suffix_codes, dense_targets, make_model,
    prepare_run, tensor_digest, train_generators, train_teacher)
from pgjepa.experiment import read_json, write_json
from pgjepa.paper_data import sha256


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    token_state = {'token_dim': 5, 'n_num': 2, 'cat_sizes': [], 'attrs_dim': 2,
        'numeric_slice': (1, 3), 'mask_slice': (3, 5), 'category_slices': [],
        'mean': np.zeros(5), 'std': np.ones(5), 'observation_end': 10., 'time_epsilon': 1e-7}
    records = [{'x_num': torch.tensor([[0., 0.], [1., 1.], [2., 2.]]),
        'x_cat': torch.empty(3, 0, dtype=torch.long), 'mask': torch.ones(3, 2),
        'delta': torch.tensor([[1.], [1.], [1.]]), 'calendar_times': torch.tensor([1., 2., 3.], dtype=torch.float64),
        'exposure_end': torch.tensor(10.), 'attrs': torch.ones(2)}]
    windows = []
    for cut, horizon in [(0, 10.), (1, 5.)]:
        item = encode_window(records[0], cut, horizon, token_state)
        item.update(entity=0, kind='bos_full' if cut == 0 else 'prefix_endpoint')
        windows.append(item)
    source = tmp_path/'source.pt'
    torch.save({'records': records, 'attrs': torch.ones(1, 2), 'token_state': token_state,
        'windows': windows, 'manifest': {'seed': 9, 'window_bank_size': 2}}, source)
    prepare_a7(source, tmp_path/'data')
    config = a7_config({'teacher_steps': 2, 'teacher_batch_size': 2, 'generator_steps': 2,
        'batch_size': 2, 'log_interval': 1, 'code_cache_batch_size': 4,
        'teacher': {'code_dim': 8, 'width': 16, 'layers': 1, 'heads': 2, 'count_components': 2},
        'model': {'code_dim': 8, 'width': 16, 'layers': 1, 'heads': 2, 'hidden': 16,
                  'components': 2, 'diffusion_steps': 10}})
    data = prepare_run(tmp_path/'data', tmp_path/'run', config)
    return data, config, tmp_path/'run'


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_teacher_calibrated_frozen_cache_and_matched_generator_noise(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    data, config, output = fixture(tmp_path)
    teacher = train_teacher(data, output, config, device)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    assert read_json(output/'teacher/complete.json')['calibrated_after_freeze']
    before = sha256(output/'teacher/final.pt')
    codes = cache_suffix_codes(teacher, data, output, config, device)
    assert [len(values) for values in codes] == [window['count']+1 for window in data['windows']]
    (output/'teacher_audit').mkdir()
    # Unit fixture gate: gate behavior itself is independently tested in its module.
    write_json(output/'teacher_audit/teacher_audit.json', {'decision': 'PASS', 'scope': 'test fixture only'})
    train_generators(data, output, config, codes, device)
    assert (output/'training_complete.json').exists()
    assert sha256(output/'teacher/final.pt') == before
    histories = [read_json(output/arm/'history.json') for arm in config['arms']]
    assert [r['diffusion_noise_signature'] for r in histories[0]] == [r['diffusion_noise_signature'] for r in histories[1]]
    assert read_json(output/config['arms'][0]/'complete.json')['initialization_digest'] == read_json(output/config['arms'][1]/'complete.json')['initialization_digest']
    saved = torch.load(output/config['arms'][0]/'final.pt', map_location='cpu', weights_only=False)
    assert saved['step'] == 2 and saved['selection'] == config['selection']
    before_checkpoint = sha256(output/config['arms'][0]/'final.pt')
    train_generators(data, output, config, codes, device)
    assert sha256(output/config['arms'][0]/'final.pt') == before_checkpoint


def test_teacher_code_targets_cannot_change_generator_forward(tmp_path):
    data, config, _ = fixture(tmp_path)
    model = make_model(data, config)
    batch = collate_a7(data['windows'])
    steps = torch.zeros(len(batch['count']), dtype=torch.long)
    noise = torch.randn_like(batch['tokens'])
    targets = torch.zeros(len(batch['count']), batch['tokens'].shape[1], 8)
    initial = tensor_digest(model.state_dict())
    first = model.forward_velocity(noise, steps, batch)['velocity']
    enriched = copy.deepcopy(batch)
    enriched['teacher_codes'] = targets+1000
    second = model.forward_velocity(noise, steps, enriched)['velocity']
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert tensor_digest(model.state_dict()) == initial


def test_gate_failure_blocks_generator_training(tmp_path):
    data, config, output = fixture(tmp_path)
    (output/'teacher_audit').mkdir()
    write_json(output/'teacher_audit/teacher_audit.json', {'decision': 'NEEDS_REVIEW'})
    with pytest.raises(RuntimeError, match='gate'):
        train_generators(data, output, config, [], 'cpu')
    assert not (output/'generator_initial.pt').exists()


def test_auxiliary_target_noise_is_independent_of_generator_rng():
    codes = [torch.ones(3, 8), torch.zeros(2, 8)]
    state = torch.get_rng_state().clone()
    left = dense_targets(codes, [0, 1], 3, 'cpu', torch.Generator().manual_seed(1), .03)
    assert torch.equal(state, torch.get_rng_state())
    right = dense_targets(codes, [0, 1], 3, 'cpu', torch.Generator().manual_seed(1), .03)
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_interrupted_generator_restores_optimizer_and_both_random_streams(tmp_path, monkeypatch):
    completed = []
    for name in ('uninterrupted', 'resumed'):
        data, config, output = fixture(tmp_path/name)
        teacher = train_teacher(data, output, config, 'cpu')
        codes = cache_suffix_codes(teacher, data, output, config, 'cpu')
        (output/'teacher_audit').mkdir()
        write_json(output/'teacher_audit/teacher_audit.json', {'decision': 'PASS', 'scope': 'test fixture only'})
        if name == 'resumed':
            original = A7Generator.loss
            calls = [0]
            def fail_after_one_saved_update(self, *args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('intentional interruption')
                return original(self, *args, **kwargs)
            monkeypatch.setattr(A7Generator, 'loss', fail_after_one_saved_update)
            with pytest.raises(RuntimeError, match='intentional interruption'):
                train_generators(data, output, config, codes, 'cpu', ['A7_AUX_ON'])
            monkeypatch.setattr(A7Generator, 'loss', original)
        train_generators(data, output, config, codes, 'cpu', ['A7_AUX_ON'])
        saved = torch.load(output/'A7_AUX_ON/final.pt', map_location='cpu', weights_only=False)
        completed.append((tensor_digest(saved['model']), read_json(output/'A7_AUX_ON/history.json')))
    assert completed[0][0] == completed[1][0]
    assert [row['diffusion_noise_signature'] for row in completed[0][1]] == [row['diffusion_noise_signature'] for row in completed[1][1]]
