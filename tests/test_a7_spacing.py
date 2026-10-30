import itertools

import pytest
import torch

from pgjepa.a7_spacing import (decode_spacing, encode_record_window,
    encode_spacing, face_log_mass, first_active_log_probs, project_clr, sample_face)


@pytest.mark.parametrize('times,anchor,horizon',[
    ([],0.,4.), ([],3.,0.), ([3.,3.],3.,0.),
    ([2.,2.,2.5,3.,3.],2.,1.), ([2.,2.,2.],2.,1.),
    ([3.,3.],2.,1.), ([2.2,2.2,2.200000000001,2.9],2.,1.),
    ([0.,1e-12,1e-8,1.],0.,1.),
])
def test_roundtrip_boundaries_ties_and_empty(times,anchor,horizon):
    source = torch.tensor(times,dtype=torch.float64)
    code = encode_spacing(source,anchor,horizon)
    restored = decode_spacing(code['clr'],code['active'],anchor,horizon)
    torch.testing.assert_close(restored['times'],source,rtol=1e-14,atol=2e-15)
    torch.testing.assert_close(restored['gaps'],code['gaps'],rtol=1e-13,atol=2e-15)
    assert torch.equal(restored['gaps']==0,~code['active'])
    assert len(restored['gaps']) == len(source)+1
    assert abs(float(code['clr'].sum())) < 1e-12
    if times and times[-1] == anchor+horizon:
        assert float(restored['times'][-1]) == anchor+horizon


def test_clr_gauge_and_invalid_values():
    active = torch.tensor([True,False,True])
    first = decode_spacing(torch.tensor([-.7,0.,.7]),active,0.,5.)
    second = decode_spacing(torch.tensor([99.3,27.,100.7]),active,0.,5.)
    torch.testing.assert_close(first['times'],second['times'],rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError,match='ordered'):
        encode_spacing(torch.tensor([.5,.1]),0.,1.)
    with pytest.raises(ValueError,match='outside'):
        encode_spacing(torch.tensor([-1.]),0.,1.)
    with pytest.raises(ValueError,match='nonempty'):
        decode_spacing(torch.zeros(3),torch.zeros(3,dtype=torch.bool),0.,1.)
    with pytest.raises(ValueError,match='empty timing'):
        decode_spacing(torch.zeros(1),torch.ones(1,dtype=torch.bool),0.,0.)
    with pytest.raises(FloatingPointError,match='underflowed'):
        decode_spacing(torch.tensor([-1000.,1000.]),torch.ones(2,dtype=torch.bool),0.,1.)


def test_face_mass_enumerates_and_sampler_matches():
    torch.manual_seed(190)
    logits = torch.tensor([-.7,.3,1.2],dtype=torch.float64,requires_grad=True)
    masks = torch.tensor(list(itertools.product([False,True],repeat=3)))
    expanded = logits.expand(8,-1)
    masses = face_log_mass(expanded,masks).exp()
    torch.testing.assert_close(masses.sum(),torch.tensor(1.,dtype=torch.float64))
    assert masses[0] == 0
    samples = sample_face(logits.detach().expand(60000,-1))
    assert bool(samples.any(-1).all())
    empirical = torch.stack([(samples==mask).all(-1).double().mean() for mask in masks])
    torch.testing.assert_close(empirical,masses.detach(),rtol=0,atol=.007)
    first_probs = first_active_log_probs(logits).exp()
    first_empirical = torch.nn.functional.one_hot(samples.long().argmax(-1),3).double().mean(0)
    torch.testing.assert_close(first_empirical,first_probs.detach(),rtol=0,atol=.007)
    (-face_log_mass(logits,torch.tensor([True,False,True]))).backward()
    assert bool(torch.isfinite(logits.grad).all())


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_extreme_faces_gradients_padding_and_singleton(device):
    logits = torch.tensor([[-10000.,-10001.,-9999.,4.],[10000.,-10000.,1.,2.]],
                          dtype=torch.float64,device=device,requires_grad=True)
    valid = torch.tensor([[True,True,True,False],[True,False,False,False]],device=device)
    support = torch.tensor([[False,False,True,False],[True,False,False,False]],device=device)
    logmass = face_log_mass(logits,support,valid)
    assert bool(torch.isfinite(logmass).all())
    assert logmass[1] == 0
    (-logmass.sum()).backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert bool((logits.grad[~valid]==0).all())
    samples = sample_face(logits.detach().expand(1000,-1,-1),valid.expand(1000,-1,-1))
    assert bool(samples.any(-1).all())
    assert not bool((samples & ~valid).any())
    projected = project_clr(torch.randn(2,4,device=device),support)
    torch.testing.assert_close(projected.sum(-1),torch.zeros(2,device=device))


def test_projection_covariance_intrinsic_dimension_and_inactive_gradient():
    torch.manual_seed(89)
    face = torch.tensor([True,False,True,True])
    noise = torch.randn(80000,4,dtype=torch.float64)
    projected = project_clr(noise,face)
    covariance = projected.T@projected/len(projected)
    expected = torch.zeros(4,4,dtype=torch.float64)
    idx = torch.nonzero(face).flatten()
    for i in idx:
        for j in idx:
            expected[i,j] = float(i==j)-1/3
    torch.testing.assert_close(covariance,expected,rtol=0,atol=.012)
    assert abs(float(projected.square().sum(-1).mean())-2) < .025
    assert float(projected.sum(-1).abs().max()) < 2e-15
    x = torch.randn(4,dtype=torch.float64,requires_grad=True)
    project_clr(x,face).square().sum().backward()
    assert x.grad[1] == 0
    assert float(x.grad.sum().abs()) < 1e-14
    empty = project_clr(x,torch.zeros(4,dtype=torch.bool))
    assert bool((empty==0).all())


def test_record_adapter_preserves_marks_and_tie_order():
    state = {'token_dim':3,'numeric_slice':(1,2),'category_slices':[],
             'mask_slice':(2,3),'observation_end':2.,'time_epsilon':1e-7,
             'mean':torch.zeros(3).numpy(),'std':torch.ones(3).numpy()}
    record = {'calendar_times':torch.tensor([0.,1.,1.,2.],dtype=torch.float64),
              'delta':torch.tensor([0.,1.,0.,1.]),'attrs':torch.tensor([1.]),
              'x_num':torch.tensor([[10.],[20.],[30.],[40.]]),
              'x_cat':torch.empty(4,0,dtype=torch.long),'mask':torch.ones(4,1)}
    result = encode_record_window(record,2,1.,state)
    assert result['count'] == 2
    torch.testing.assert_close(result['marks'][:,0],torch.tensor([30.,40.]))
    torch.testing.assert_close(result['times'],torch.tensor([1.,2.],dtype=torch.float64))
    assert torch.equal(result['active'],torch.tensor([False,True,False]))
    zero = encode_record_window(record,2,0.,state)
    assert zero['count'] == 1
    assert not bool(zero['active'].any())
    torch.testing.assert_close(zero['marks'][:,0],torch.tensor([30.]))
