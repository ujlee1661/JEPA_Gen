"""Oracle time-pattern teacher, conditioned on N and calendar at decoding.

Teacher encoder receives future timing only, not event marks. Layout still
reveals N; this model does not promise an N-independent code. The decoder
predicts the face without receiving it, then scores CLR conditional on face.
"""
from __future__ import annotations

import torch
from torch import Tensor
from .teacher_base import FutureTeacher
from .spacing import face_log_mass, project_clr


class TimeTeacher(FutureTeacher):
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        valid, active, events, count, _, position = self._layout(batch)
        tokens = batch['tokens']
        if tokens.shape != (*valid.shape, self.token_dim) or not bool(torch.isfinite(tokens[..., 0][valid]).all()):
            raise ValueError('Finite future CLR channel required')
        time = torch.where(valid, tokens[..., 0], torch.zeros_like(tokens[..., 0]))
        timing_only = torch.zeros_like(tokens)
        timing_only[..., 0] = time
        inputs = torch.cat([timing_only, active[..., None].to(tokens), events[..., None].to(tokens)], -1)
        hidden = self.input(inputs)+self.position(position)
        cls = self.cls.expand(len(tokens), -1, -1)+(count == 0).to(tokens)[:, None, None]*self.empty[None, None]
        mask = torch.cat([torch.ones(len(tokens), 1, dtype=torch.bool, device=tokens.device), valid], -1)
        raw = self.to_code(self.encoder(torch.cat([cls, hidden], 1), src_key_padding_mask=~mask)[:, 0])
        return (raw-self.code_mean)/self.code_std

    def loss(self, batch: dict[str, Tensor]):
        valid, active, _, _, condition, position = self._layout(batch)
        z = self.encode(batch)
        raw_z = z*self.code_std+self.code_mean
        queries = self.position(position)+self.condition(condition)[:, None]+self.from_code(raw_z)[:, None]
        hidden = self.decoder(queries, src_key_padding_mask=~valid)
        logits = self.face(hidden).squeeze(-1)
        positive = batch['delta'] > 0
        face = z.new_zeros(len(z))
        if bool(positive.any()):
            face[positive] = (-face_log_mass(logits[positive], active[positive], valid[positive])).to(z)
        target = project_clr(batch['tokens'][..., 0], active)
        prediction = project_clr(self.coordinates(hidden)[..., 0], active)
        dof = (active.sum(-1)-1).clamp_min(0)
        # Unit-variance Gaussian on the active CLR subspace (constants omitted)
        # plus the exact categorical face NLL; zero degrees give zero cost.
        error = (0.5*(prediction-target).square().sum(-1)).to(z)
        loss = (face+error).mean()
        return loss, {'teacher_face_nll':float(face.mean().detach()), 'teacher_clr_mse':float(error.mean().detach()),
                      'teacher_loss':float(loss.detach()), 'zero_time_dof':int((dof == 0).sum())}
