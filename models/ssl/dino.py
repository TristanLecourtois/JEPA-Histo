

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.vit import VisionTransformer, build_vit
from models.heads.projection_mlp import DINOHead


class DINO(nn.Module):
    """DINO model: student / teacher pair with centering.

    Args:
        encoder_cfg:     ``model.encoder`` config sub-dict.
        head_cfg:        ``model.projection_head`` config sub-dict.
        dino_cfg:        ``dino`` config sub-dict (temperatures, centering).
        ema_momentum:    Initial EMA momentum for the teacher encoder.
    """

    def __init__(
        self,
        encoder_cfg: dict,
        head_cfg: dict,
        dino_cfg: dict,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()

        embed_dim = encoder_cfg.get("embed_dim", 384)

        # Student
        self.student_encoder: VisionTransformer = build_vit(encoder_cfg)
        self.student_head = DINOHead(
            in_dim         = embed_dim,
            out_dim        = head_cfg.get("out_dim", 65536),
            hidden_dim     = head_cfg.get("hidden_dim", 2048),
            bottleneck_dim = head_cfg.get("bottleneck_dim", 256),
            nlayers        = head_cfg.get("nlayers", 3),
            norm_last_layer= head_cfg.get("norm_last_layer", True),
        )

        # Teacher (EMA, no gradient)
        self.teacher_encoder: VisionTransformer = copy.deepcopy(self.student_encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

        self.dino_cfg     = dino_cfg
        self.ema_momentum = ema_momentum

        # Running centre (registered as buffer to move to device automatically)
        out_dim = head_cfg.get("out_dim", 65536)
        self.register_buffer("center", torch.zeros(1, out_dim))


    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Update teacher encoder and head via EMA."""
        for param_s, param_t in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)
        for param_s, param_t in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

 
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update the running centring vector.

        Args:
            teacher_output: Teacher logits of shape ``(B*N_views, D)``.
        """
        m = self.dino_cfg.get("center_momentum", 0.9)
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * m + batch_center * (1 - m)


    @staticmethod
    def _softmax_with_temp(x: torch.Tensor, temp: float) -> torch.Tensor:
        return F.softmax(x / temp, dim=-1)

    def _dino_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        student_temp: float,
        teacher_temp: float,
        n_crops: int,
        n_global: int = 2,
    ) -> torch.Tensor:
        """Compute cross-entropy between all student crops and global teacher crops.

        Args:
            student_out:  Concatenated student logits ``(B * n_crops, D)``.
            teacher_out:  Concatenated teacher logits ``(B * n_global, D)``.
            student_temp: Student temperature.
            teacher_temp: Teacher temperature.
            n_crops:      Total number of crops (global + local).
            n_global:     Number of global crops.

        Returns:
            Scalar mean cross-entropy loss.
        """
        B_n = student_out.shape[0]
        B   = B_n // n_crops

        s_chunks = student_out.chunk(n_crops)
        t_chunks = (teacher_out - self.center).chunk(n_global)

        t_probs = [self._softmax_with_temp(t, teacher_temp) for t in t_chunks]

        total_loss = 0.0
        n_pairs    = 0
        for i, s in enumerate(s_chunks):
            s_log_prob = F.log_softmax(s / student_temp, dim=-1)
            for j, t_p in enumerate(t_probs):
                if i == j and i < n_global:
                    continue  # skip same view
                total_loss += -(t_p * s_log_prob).sum(dim=-1).mean()
                n_pairs    += 1

        return total_loss / n_pairs

    def forward(
        self,
        crops: List[torch.Tensor],
        teacher_temp: float,
        n_global_crops: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """Compute the DINO self-distillation loss.

        Args:
            crops:          List of image tensors:
                            ``[global_1, global_2, local_1, ..., local_N]``.
                            All tensors have shape ``(B, C, H, W)``.
            teacher_temp:   Current teacher temperature (annealed externally).
            n_global_crops: Number of global crops at the head of ``crops``.

        Returns:
            Dictionary with keys:
              - ``"loss"``:        Scalar DINO loss.
              - ``"student_out"``: Student logits.
              - ``"teacher_out"``: Teacher logits (global crops).
        """
        student_temp = self.dino_cfg.get("student_temp", 0.1)

        # Student forward — all crops
        all_student_feats = torch.cat(
            [self.student_encoder.forward_features(c) for c in crops], dim=0
        )
        student_out = self.student_head(all_student_feats)

        # Teacher forward — global crops only (no gradient)
        with torch.no_grad():
            all_teacher_feats = torch.cat(
                [self.teacher_encoder.forward_features(c) for c in crops[:n_global_crops]],
                dim=0,
            )
            teacher_out = self.teacher_head(all_teacher_feats)

        loss = self._dino_loss(
            student_out, teacher_out,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            n_crops=len(crops),
            n_global=n_global_crops,
        )

        # Update centre for next iteration
        self.update_center(teacher_out)

        return {
            "loss":        loss,
            "student_out": student_out,
            "teacher_out": teacher_out,
        }

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token features using the teacher encoder.

        Using the teacher (EMA) encoder at evaluation time generally yields
        slightly better representations than the student.

        Args:
            images: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, D)``.
        """
        return self.teacher_encoder.forward_features(images)


class TeacherTempScheduler:
    """Warm-up then constant schedule for the DINO teacher temperature.

    Args:
        temp_start:     Initial teacher temperature.
        temp_end:       Final teacher temperature after warm-up.
        warmup_epochs:  Number of warm-up epochs.
        total_epochs:   Total training epochs.
        steps_per_epoch: Number of optimisation steps per epoch.
    """

    def __init__(
        self,
        temp_start: float = 0.04,
        temp_end: float = 0.07,
        warmup_epochs: int = 30,
        total_epochs: int = 300,
        steps_per_epoch: int = 500,
    ) -> None:
        self.temp_start      = temp_start
        self.temp_end        = temp_end
        self.warmup_steps    = warmup_epochs * steps_per_epoch
        self.total_steps     = total_epochs * steps_per_epoch
        self._step           = 0

    def get_temp(self) -> float:
        if self._step < self.warmup_steps:
            return self.temp_start + (self.temp_end - self.temp_start) * (
                self._step / self.warmup_steps
            )
        return self.temp_end

    def step(self) -> float:
        temp = self.get_temp()
        self._step += 1
        return temp
