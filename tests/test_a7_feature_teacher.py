import pytest
import torch

from pgjepa.a7_distribution_teacher import A7DistributionTeacher
from pgjepa.a7_feature_teacher import A7FeatureTeacher
from pgjepa.a7_spacing import project_clr


def model(kind=A7FeatureTeacher,device='cpu'):
    return kind(5,2,10.,code_dim=8,width=16,layers=1,heads=4,decoder_layers=1).to(device)


def batch(device='cpu'):
    n=torch.tensor([3,0,2],device=device);i=torch.arange(4,device=device)[None]
    active=torch.tensor([[1,0,1,1],[1,0,0,0],[0,0,0,0]],dtype=torch.bool,device=device)
    tokens=torch.randn(3,4,5,device=device);tokens[...,0]=project_clr(tokens[...,0],active)
    return {'tokens':tokens,'valid':i<=n[:,None],'event_valid':i<n[:,None],'active':active,
        'count':n,'attrs':torch.randn(3,2,device=device),'anchor':torch.tensor([0.,2.,4.],device=device),
        'delta':torch.tensor([10.,8.,0.],device=device)}


def test_encoder_is_exactly_unchanged_and_calibration_freeze_api_is_compatible():
    torch.manual_seed(9);base=model(A7DistributionTeacher).eval()
    torch.manual_seed(9);feature=model().eval()
    b=batch()
    torch.testing.assert_close(base.encode(b),feature.encode(b),rtol=0,atol=0)
    codes=feature.encode(b)
    feature.set_code_calibration(codes);feature.freeze()
    assert not feature.training and all(not p.requires_grad for p in feature.parameters())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_decoder_conditions_cannot_read_clean_marks_or_reencode_them(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=model(device=device).eval();b=batch(device)
    z=m.encode(b);gn=torch.randn(3,8,device=device);ln=torch.randn(3,4,5,device=device)
    conditions=m.decoder_conditions(b)
    assert set(conditions)=={'valid','active','event_valid','count','attrs','anchor','delta','future_time_clr'}
    first=m.sample_reconstruction(z,conditions,gn,ln)
    changed={**b,'tokens':b['tokens'].clone()};changed['tokens'][...,1:]=float('nan')
    def forbidden(*args,**kwargs):raise AssertionError('Decoder must not call the future encoder')
    m.encode=forbidden
    second=m.sample_reconstruction(z,changed,gn,ln)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(first[...,0],conditions['future_time_clr'],rtol=0,atol=0)
    assert not first[1,...,1:].any() and not first[2,:,0].any()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_feature_only_energy_loss_is_finite_backward_including_empty_ties_and_zero_duration(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=model(device=device);b=batch(device)
    gn,ln=m.draw_noise_pair(b)
    loss,stats=m.loss(b,objective='energy',global_noise_pair=gn,local_noise_pair=ln)
    loss.backward()
    assert torch.isfinite(loss) and stats['zero_mark_dof_paths']==1
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.future_time_input.weight.grad.abs().sum()>0
    assert m.input.weight.grad.abs().sum()>0 and m.coordinates[-1].weight.grad[1:].abs().sum()>0
    with pytest.raises(ValueError):m.loss(b,global_noise_pair=gn)


def test_objective_ignores_reconstructed_time_coordinates_entirely():
    m=model();b=batch();gn,ln=m.draw_noise_pair(b)
    def reconstruction(time):
        def f(z,layout,*args):
            output=z.new_zeros(3,4,5);output[...,0]=time
            return output
        return f
    m.sample_reconstruction=reconstruction(0.)
    a,stats_a=m.loss(b,'energy',gn,ln)
    m.sample_reconstruction=reconstruction(1e12)
    c,stats_c=m.loss(b,'energy',gn,ln)
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    assert stats_a['teacher_feature_energy']==stats_c['teacher_feature_energy']
