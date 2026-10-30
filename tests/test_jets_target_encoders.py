"""Small end-to-end checks for target encoder training and suffix caches."""
import pytest
import torch

import experiments.target_encoders as target_encoders
from experiments.data import load_cache
from experiments.target_encoders import build_cache, target_encoder_config, train_target_encoder
from jets.data import _complete_window, build_suffix_window, collate_windows
from jets.spacing import encode_spacing


@pytest.fixture
def encoder_bank(tmp_path):
    windows = []
    for times, marks in [([1., 3., 6.], [[.1, .5], [.2, .6], [.3, .7]]),
                         ([2., 5.], [[.4, .8], [.5, .9]])]:
        ts = torch.tensor(times, dtype=torch.float64)
        spacing = encode_spacing(ts, 0., 10.)
        raw = {'count':len(times), 'anchor':0., 'delta':10., 'times':ts,
               'marks':torch.tensor(marks), 'attrs':torch.ones(2),
               'past_tokens':torch.empty(0, 3), **spacing}
        window = _complete_window(raw, 1.)
        window['past_extension_tokens'] = torch.zeros(len(times), 3)
        windows.append(window)
    bank = tmp_path/'data.pt'
    torch.save({'windows':windows, 'attrs':torch.ones(2, 2),
        'state':{'token_dim':3,'time_scale':1.,'mark_token_state':{'observation_end':10.}}}, bank)
    cfg = {'seed':9, 'teacher_steps':2, 'teacher_batch_size':2, 'code_cache_batch_size':2,
           'log_interval':1, 'teacher':{'code_dim':8,'width':16,'layers':1,'heads':2,'decoder_layers':1}}
    return bank, cfg, windows


@pytest.mark.parametrize('family', ['time', 'feature'])
def test_encoder_resume_calibration_and_suffix_cache(encoder_bank, tmp_path, family):
    bank, cfg, windows = encoder_bank
    interrupted = tmp_path/f'{family}_resume'
    assert train_target_encoder(bank, interrupted, family, cfg, stop_after=1) is None
    resumed = train_target_encoder(bank, interrupted, family, cfg)
    continuous_dir = tmp_path/f'{family}_continuous'
    continuous = train_target_encoder(bank, continuous_dir, family, cfg)
    assert not resumed.training and not any(p.requires_grad for p in resumed.parameters())
    assert resumed.state_dict().keys() == continuous.state_dict().keys()
    assert all(torch.equal(resumed.state_dict()[key], continuous.state_dict()[key])
               for key in resumed.state_dict())
    ckpt = interrupted/'final.pt'
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert saved['calibrated_training_only'] and saved['step'] == 2
    assert train_target_encoder(bank, interrupted, family, cfg) is not None
    cache_path = build_cache(bank, ckpt, interrupted/'suffix_codes.pt', family, cfg)
    codes = load_cache(cache_path, bank, windows, 8, ckpt)
    assert [list(c.shape) for c in codes] == [[4, 8], [3, 8]]
    # A suffix is independently re-centered, not sliced from the full CLR.
    suffix = build_suffix_window(windows[0], 1)
    with torch.no_grad():
        direct = resumed.encode(collate_windows([suffix]))[0]
    # Different batch padding can produce tiny floating-point kernel differences,
    # amplified by learned code calibration; the suffix coordinates are equal.
    torch.testing.assert_close(codes[0][1], direct, atol=1e-4, rtol=1e-4)
    original = cache_path.read_bytes()
    assert build_cache(bank, ckpt, cache_path, family, cfg) == cache_path
    assert cache_path.read_bytes() == original
    modified = torch.load(cache_path, map_location='cpu', weights_only=False)
    modified['codes'][0][0, 0] += 1
    torch.save(modified, cache_path)
    with pytest.raises(ValueError, match='digest mismatch'):
        build_cache(bank, ckpt, cache_path, family, cfg)
    with pytest.raises(ValueError, match='does not match'):
        build_cache(bank, ckpt, interrupted/'wrong.pt', 'feature' if family == 'time' else 'time', cfg)
    with ckpt.open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError, match='digest mismatch'):
        train_target_encoder(bank, interrupted, family, cfg)


def test_config_and_existing_output_identity_are_enforced(encoder_bank, tmp_path, monkeypatch):
    bank, cfg, _ = encoder_bank
    out = tmp_path/'safe'
    train_target_encoder(bank, out, 'time', cfg, stop_after=1)
    changed = {**cfg, 'seed': 10}
    with pytest.raises(ValueError, match='different bank or configuration'):
        train_target_encoder(bank, out, 'time', changed)
    assert not (out/'final.pt').exists()
    assert target_encoder_config({'model': {'code_dim': 8}})['teacher']['code_dim'] == 8
    with pytest.raises(ValueError, match='dimensions must match'):
        target_encoder_config({'model': {'code_dim': 8}, 'teacher': {'code_dim': 16}})
    monkeypatch.setattr(target_encoders, '_source_hashes', lambda: {'jets/changed.py': 'changed'})
    with pytest.raises(ValueError, match='different bank or configuration'):
        train_target_encoder(bank, out, 'time', cfg)


@pytest.mark.parametrize('failure_at', ['checkpoint_receipt', 'cache_receipt'])
def test_interrupted_publication_recovers_verified_artifact(encoder_bank, tmp_path, monkeypatch, failure_at):
    bank, cfg, windows = encoder_bank
    out = tmp_path/'recover'
    publish = target_encoders._publish_json_exclusive
    expected_name = 'complete.json' if failure_at == 'checkpoint_receipt' else 'suffix_codes.json'
    calls = {'injected': False}

    def fail_once(path, value):
        if path.name == expected_name and not calls['injected']:
            calls['injected'] = True
            raise OSError('simulated interruption after artifact publication')
        return publish(path, value)

    if failure_at == 'checkpoint_receipt':
        monkeypatch.setattr(target_encoders, '_publish_json_exclusive', fail_once)
        with pytest.raises(OSError, match='simulated interruption'):
            train_target_encoder(bank, out, 'time', cfg)
        final = out/'final.pt'
        assert final.is_file() and not (out/'complete.json').exists()
        old_bytes = final.read_bytes()
        monkeypatch.setattr(target_encoders, '_publish_json_exclusive', publish)
        model = train_target_encoder(bank, out, 'time', cfg)
        assert model is not None and final.read_bytes() == old_bytes
        assert (out/'complete.json').is_file()
        assert not (out/'.final.pt.journal.json').exists()
    else:
        train_target_encoder(bank, out, 'time', cfg)
        checkpoint, cache = out/'final.pt', out/'suffix_codes.pt'
        monkeypatch.setattr(target_encoders, '_publish_json_exclusive', fail_once)
        with pytest.raises(OSError, match='simulated interruption'):
            build_cache(bank, checkpoint, cache, 'time', cfg)
        assert cache.is_file() and not cache.with_suffix('.json').exists()
        old_bytes = cache.read_bytes()
        monkeypatch.setattr(target_encoders, '_publish_json_exclusive', publish)
        build_cache(bank, checkpoint, cache, 'time', cfg)
        assert cache.read_bytes() == old_bytes
        assert len(load_cache(cache, bank, windows, 8, checkpoint)) == len(windows)
        assert not (out/'.suffix_codes.pt.journal.json').exists()


def test_interrupted_cache_tampering_fails_without_overwrite(encoder_bank, tmp_path, monkeypatch):
    bank, cfg, _ = encoder_bank
    out = tmp_path/'tamper'
    train_target_encoder(bank, out, 'time', cfg)
    checkpoint, cache = out/'final.pt', out/'suffix_codes.pt'
    publish = target_encoders._publish_json_exclusive

    def interrupt_receipt(path, value):
        if path.name == 'suffix_codes.json':
            raise OSError('simulated interruption')
        return publish(path, value)

    monkeypatch.setattr(target_encoders, '_publish_json_exclusive', interrupt_receipt)
    with pytest.raises(OSError):
        build_cache(bank, checkpoint, cache, 'time', cfg)
    with cache.open('ab') as stream:
        stream.write(b'tampered')
    damaged = cache.read_bytes()
    monkeypatch.setattr(target_encoders, '_publish_json_exclusive', publish)
    with pytest.raises(ValueError, match='digest mismatch|differs'):
        build_cache(bank, checkpoint, cache, 'time', cfg)
    assert cache.read_bytes() == damaged
    assert not cache.with_suffix('.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('family', ['time', 'feature'])
def test_target_encoder_cuda_cache_smoke(encoder_bank, tmp_path, family):
    bank, cfg, windows = encoder_bank
    out = tmp_path/f'{family}_cuda'
    model = train_target_encoder(bank, out, family, cfg, device='cuda')
    assert model is not None and not model.training
    cache = build_cache(bank, out/'final.pt', out/'suffix_codes.pt', family, cfg, device='cuda')
    codes = load_cache(cache, bank, windows, 8, out/'final.pt')
    assert all(bool(torch.isfinite(item).all()) for item in codes)
