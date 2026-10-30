import torch

from pgjepa.a9_conditional_time_teacher import A9ConditionalTimeTeacher


def test_marks_are_not_encoder_input_and_zero_duration_is_finite():
    torch.manual_seed(3)
    model = A9ConditionalTimeTeacher(4, 2, 100., code_dim=8, width=16, layers=1, heads=2)
    batch = {'tokens': torch.tensor([[[0., 1., 2., 3.]]]),
             'valid': torch.tensor([[True]]), 'active': torch.tensor([[False]]),
             'event_valid': torch.tensor([[False]]), 'count': torch.tensor([0]),
             'attrs': torch.zeros(1, 2), 'anchor': torch.tensor([0.]), 'delta': torch.tensor([0.])}
    first = model.encode(batch)
    altered = dict(batch); altered['tokens'] = batch['tokens'].clone(); altered['tokens'][..., 1:] = 91.
    assert torch.equal(first, model.encode(altered))
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_face_target_changes_loss_and_marks_do_not():
    torch.manual_seed(7)
    model = A9ConditionalTimeTeacher(3, 1, 100., code_dim=8, width=16, layers=1, heads=2)
    batch = {'tokens': torch.tensor([[[0.4, 1., 2.], [-0.4, 2., 3.], [0., 0., 0.]]]),
             'valid': torch.tensor([[True, True, True]]), 'active': torch.tensor([[True, True, False]]),
             'event_valid': torch.tensor([[True, True, False]]), 'count': torch.tensor([2]),
             'attrs': torch.zeros(1, 1), 'anchor': torch.tensor([1.]), 'delta': torch.tensor([10.])}
    a, _ = model.loss(batch)
    altered = dict(batch); altered['tokens'] = batch['tokens'].clone(); altered['tokens'][..., 1:] += 17
    b, _ = model.loss(altered)
    assert torch.equal(a, b)
    a.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
