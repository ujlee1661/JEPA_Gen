"""Future feature-information teacher conditional on the true future timeline.

The encoder is the unchanged distribution-teacher encoder. Only teacher-side
reconstruction sees the actual future CLR/time support and metadata; its proper
energy score concerns marks only. This oracle conditional reconstruction is
not native generation. Native generators receive predicted features, never the
true future timeline, marks or encoded future teacher target.
"""
from __future__ import annotations

import torch
from torch import Tensor,nn

from .distribution import DistributionTargetEncoder


class FeatureTargetEncoder(DistributionTargetEncoder):
    teacher_kind='feature_only_energy_given_true_future_time'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.future_time_input=nn.Linear(1,self.cls.shape[-1])

    @staticmethod
    def decoder_conditions(batch:dict) -> dict:
        """Copy only privileged time/layout/static inputs, never future marks."""
        result={key:batch[key] for key in ('valid','active','event_valid','count','attrs','anchor','delta')}
        time=batch['future_time_clr'] if 'future_time_clr' in batch else batch['tokens'][...,0]
        if time.shape!=batch['valid'].shape or not bool(torch.isfinite(time[batch['valid']]).all()):
            raise ValueError('Finite actual future CLR condition matching the layout required')
        # This is an observed condition, not a generated time estimate. Keep
        # the existing normalized CLR values exactly; repeated gauge projection
        # would introduce unnecessary float32 roundoff across API forms.
        result['future_time_clr']=torch.where(batch['valid'],time,torch.zeros_like(time))
        return result

    def sample_reconstruction(self,z:Tensor,batch_layout:dict,global_noise:Tensor|None=None,
                              local_noise:Tensor|None=None) -> Tensor:
        layout=self.decoder_conditions(batch_layout)
        valid,active,events,_,condition,position=self._layout(layout)
        if z.shape!=(len(valid),self.code_dim):raise ValueError('One code per feature reconstruction required')
        if global_noise is None:global_noise=torch.randn_like(z)
        if local_noise is None:local_noise=z.new_empty(*valid.shape,self.token_dim).normal_()
        if global_noise.shape!=z.shape or local_noise.shape!=(*valid.shape,self.token_dim):
            raise ValueError('Independent global/local noise shapes differ')
        if not bool(torch.isfinite(global_noise).all() and torch.isfinite(local_noise).all()):
            raise ValueError('Finite teacher-side independent reconstruction noise required')
        raw_code=z*self.code_std+self.code_mean
        flags=torch.stack([active,events],-1).to(z)
        time=layout['future_time_clr'].to(z)
        hidden=(self.position(position)+self.condition(condition)[:,None]+self.from_code(raw_code)[:,None]
            +self.global_noise_input(global_noise)[:,None]+self.local_noise_input(local_noise)
            +self.face_input(flags)+self.future_time_input(time[...,None]))
        hidden=self.decoder(hidden,src_key_padding_mask=~valid)
        raw=self.coordinates(hidden)
        marks=torch.where(events[...,None],raw[...,1:],torch.zeros_like(raw[...,1:]))
        return torch.cat([time[...,None],marks],-1)

    def loss(self,batch:dict,objective:str='energy',global_noise_pair:Tensor|None=None,
             local_noise_pair:Tensor|None=None):
        if objective not in ('energy','mse'):raise ValueError('Choose feature energy or matched MSE diagnostic')
        valid,_,events,count,_,_=self._layout(batch)
        if (global_noise_pair is None)!=(local_noise_pair is None):raise ValueError('Supply both independent noise pairs')
        if global_noise_pair is None:global_noise_pair,local_noise_pair=self.draw_noise_pair(batch)
        if local_noise_pair is None or global_noise_pair.shape!=(2,len(valid),self.code_dim) or local_noise_pair.shape!=(2,*valid.shape,self.token_dim):
            raise ValueError('Exactly two complete independent noise draws required')
        z=self.encode(batch)
        conditions=self.decoder_conditions(batch)
        first=self.sample_reconstruction(z,conditions,global_noise_pair[0],local_noise_pair[0])
        second=self.sample_reconstruction(z,conditions,global_noise_pair[1],local_noise_pair[1])
        target=torch.where(events[...,None],batch['tokens'][...,1:],torch.zeros_like(batch['tokens'][...,1:]))
        a=(first[...,1:]-target).flatten(1);b=(second[...,1:]-target).flatten(1)
        pair=(first[...,1:]-second[...,1:]).flatten(1)
        dimension=count*(self.token_dim-1);denominator=dimension.clamp_min(1).to(z)
        one_norm=torch.linalg.vector_norm(a,dim=-1);two_norm=torch.linalg.vector_norm(b,dim=-1)
        pair_norm=torch.linalg.vector_norm(pair,dim=-1)
        energy=.5*(one_norm+two_norm-pair_norm)/denominator.sqrt()
        mse=.5*(a.square().sum(-1)+b.square().sum(-1))/denominator
        loss=(energy if objective=='energy' else mse).mean()
        return loss,{'teacher_loss':float(loss.detach()),'teacher_feature_energy':float(energy.detach().mean()),
            'teacher_feature_mse':float(mse.detach().mean()),'teacher_energy':float(energy.detach().mean()),
            'teacher_mse':float(mse.detach().mean()),'sample_pair_distance':float((pair_norm/denominator.sqrt()).detach().mean()),
            'zero_mark_dof_paths':float((dimension==0).sum()),'teacher_code_variance':float(z.detach().var(0,unbiased=False).mean())}
