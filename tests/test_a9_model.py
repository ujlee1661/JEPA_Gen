import torch
from pgjepa.a7_data import collate_a7
from pgjepa.a7_model import A7Generator
from pgjepa.a9_model import A9Generator
from test_a7_training import fixture


def test_a9_initial_identity_and_feature_loss_control(tmp_path):
    data, config, _ = fixture(tmp_path)
    kwargs = dict(config['model'])
    args = (data['state']['token_dim'], data['attrs'].shape[1],
            data['state']['mark_token_state']['observation_end'])
    torch.manual_seed(10)
    old = A7Generator(*args, time_scale=data['state']['time_scale'], **kwargs)
    new = A9Generator(*args, time_scale=data['state']['time_scale'], **kwargs)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(k.startswith(('feature_prediction.', 'feature_residual.')) for k in missing)
    assert not unexpected
    batch = collate_a7(data['windows'][:2], 'cpu')
    noise = torch.randn_like(batch['tokens'])
    steps = torch.zeros(len(noise), dtype=torch.long)
    old_out = old.forward_velocity(noise, steps, batch)
    new_out = new.forward_velocity(noise, steps, batch)
    assert torch.equal(old_out['velocity'], new_out['velocity'])
    assert torch.equal(old.count_parameters(batch['attrs'], batch['anchor'], batch['delta'])['logits'],
                       new.count_parameters(batch['attrs'], batch['anchor'], batch['delta'])['logits'])
    target = torch.randn(*batch['valid'].shape, new.code_dim)
    off, _ = new.loss(batch, target, target[:, 0], 1., feature_teacher_codes=target,
                      feature_aux_weight=0., noise=noise, steps=steps)
    off_changed, _ = new.loss(batch, target, target[:, 0], 1., feature_teacher_codes=target+100,
                              feature_aux_weight=0., noise=noise, steps=steps)
    assert torch.equal(off, off_changed)
    on, stats = new.loss(batch, target, target[:, 0], 1., feature_teacher_codes=target,
                         feature_aux_weight=1., noise=noise, steps=steps)
    assert torch.isfinite(on) and stats['feature_aux_nll_per_dim'] > 0
    on.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.feature_prediction.parameters())
