import torch
import pytest
from pgjepa.a7_data import collate_a7
from pgjepa.a9_model import A9Generator
from pgjepa.a9_generic_model import A9GenericGenerator
from test_a7_training import fixture


def test_generic_initial_identity_and_fixed_noise(tmp_path):
    data, config, _ = fixture(tmp_path)
    args = (data['state']['token_dim'], data['attrs'].shape[1],
            data['state']['mark_token_state']['observation_end'])
    kwargs = dict(time_scale=data['state']['time_scale'], **config['model'])
    torch.manual_seed(10)
    reference = A9Generator(*args, **kwargs)
    model = A9GenericGenerator(*args, **kwargs)
    model.load_state_dict(reference.state_dict(), strict=True)
    batch = collate_a7(data['windows'][:2], 'cpu')
    noise = torch.randn_like(batch['tokens'])
    steps = torch.zeros(len(noise), dtype=torch.long)
    model.stochastic_condition = True
    torch.manual_seed(23)
    model._draw_condition(len(noise), noise.device, noise.dtype)
    first = model.forward_velocity(noise, steps, batch)['velocity']
    second = model.forward_velocity(noise, steps, batch)['velocity']
    assert torch.equal(first, second)
    # Zero residual initialization preserves the old output for any epsilon.
    assert torch.equal(first, reference.forward_velocity(noise, steps, batch)['velocity'])
    model._condition_epsilon = None


def test_rng_matched_and_feature_targets_not_generation_input(tmp_path):
    data, config, _ = fixture(tmp_path)
    args=(data['state']['token_dim'],data['attrs'].shape[1],data['state']['mark_token_state']['observation_end'])
    model=A9GenericGenerator(*args,time_scale=data['state']['time_scale'],**config['model'])
    batch=collate_a7(data['windows'][:2], 'cpu')
    target=torch.randn(*batch['valid'].shape,model.code_dim)
    noise=torch.randn_like(batch['tokens']); steps=torch.zeros(len(noise),dtype=torch.long)
    states=[]
    for stochastic in (False,True):
        model.stochastic_condition=stochastic
        torch.manual_seed(83)
        loss,_=model.loss(batch,target,target[:,0],1.,feature_teacher_codes=target,
                          feature_aux_weight=1.,noise=noise,steps=steps)
        assert torch.isfinite(loss)
        states.append(torch.get_rng_state())
        assert model._condition_epsilon is None
    assert torch.equal(*states)
    # The generation entry point accepts no teacher code argument.
    try:
        model.sample_paths(batch['attrs'],batch['anchor'],batch['delta'],teacher_codes=target)
        assert False, 'teacher code accepted at generation'
    except TypeError:
        pass


def test_native_sampling_shared_noise_and_resume_rng(tmp_path):
    data, config, _ = fixture(tmp_path)
    args=(data['state']['token_dim'],data['attrs'].shape[1],data['state']['mark_token_state']['observation_end'])
    model=A9GenericGenerator(*args,time_scale=data['state']['time_scale'],**config['model'])
    model.stochastic_condition=True
    model.eval()
    with torch.no_grad():
        model.feature_residual[-1].weight.fill_(.01)
    eps=[]
    def observe(module, inp, out):
        eps.append(model._condition_epsilon.detach().clone())
    hook=model.feature_residual.register_forward_hook(observe)
    attrs=data['attrs'][:1].repeat(2,1)
    anchors=torch.zeros(2,dtype=torch.float64)
    horizon=torch.ones(2,dtype=torch.float64)
    torch.manual_seed(108)
    state=torch.get_rng_state()
    result=model.sample_paths(attrs,anchors,horizon,sample_steps=2,max_events=4096)
    final_rng=torch.get_rng_state()
    assert len(eps)==2 and torch.equal(eps[0],eps[1])
    assert not torch.equal(eps[0][0],eps[0][1])
    assert model._condition_epsilon is None
    eps.clear()
    torch.set_rng_state(state)
    replay=model.sample_paths(attrs,anchors,horizon,sample_steps=2,max_events=4096)
    assert torch.equal(result['tokens'],replay['tokens'])
    assert torch.equal(result['count'],replay['count'])
    assert torch.equal(final_rng,torch.get_rng_state())
    hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_rng_replay_feature_condition(tmp_path):
    data, config, _ = fixture(tmp_path)
    args=(data['state']['token_dim'],data['attrs'].shape[1],data['state']['mark_token_state']['observation_end'])
    model=A9GenericGenerator(*args,time_scale=data['state']['time_scale'],**config['model']).cuda()
    model.stochastic_condition=True
    batch=collate_a7(data['windows'][:2], 'cuda')
    target=torch.zeros(*batch['valid'].shape,model.code_dim,device='cuda')
    torch.manual_seed(46);torch.cuda.manual_seed_all(46)
    cpu_state=torch.get_rng_state();gpu_state=torch.cuda.get_rng_state_all()
    first,_=model.loss(batch,target,target[:,0],1.,feature_teacher_codes=target,feature_aux_weight=1.)
    torch.set_rng_state(cpu_state);torch.cuda.set_rng_state_all(gpu_state)
    second,_=model.loss(batch,target,target[:,0],1.,feature_teacher_codes=target,feature_aux_weight=1.)
    assert torch.equal(first,second)
    second.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.feature_prediction.parameters())
