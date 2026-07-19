from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer


@dataclass
class DistillationSchedule:
    alpha_start: float = 0.8
    alpha_end: float = 0.3
    temperature_start: float = 6.0
    temperature_end: float = 2.0
    feature_loss_weight: float = 0.1

    def value_at(self, progress: float) -> tuple[float, float]:
        progress = min(max(progress, 0.0), 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        temperature = self.temperature_start + (self.temperature_end - self.temperature_start) * progress
        return alpha, temperature


class DistillationTrainer(Trainer):
    """Trainer that distills a teacher classifier into a smaller student.

    Loss = alpha * KL(student_logits, teacher_logits)
         + (1 - alpha) * CE(student_logits, labels)
         + feature_loss_weight * MSE(projected_student_hidden, teacher_hidden)

    The projection layer handles hidden-size mismatch, e.g. 128-dim student
    features versus 768-dim teacher features.
    """

    def __init__(
        self,
        *args: Any,
        teacher_model: torch.nn.Module,
        schedule: DistillationSchedule,
        feature_projection: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.args.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.schedule = schedule
        self.feature_projection = feature_projection
        if self.feature_projection is not None:
            self.feature_projection.to(self.args.device)

    def create_optimizer(self) -> None:
        super().create_optimizer()
        if self.feature_projection is None or self.optimizer is None:
            return
        existing = {id(param) for group in self.optimizer.param_groups for param in group["params"]}
        projection_params = [param for param in self.feature_projection.parameters() if id(param) not in existing]
        if projection_params:
            self.optimizer.add_param_group({"params": projection_params})

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs["labels"]

        student_outputs = model(**inputs, output_hidden_states=True)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)

        max_steps = max(int(self.state.max_steps), 1)
        progress = float(self.state.global_step) / float(max_steps)
        alpha, temperature = self.schedule.value_at(progress)

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        ce_loss = F.cross_entropy(student_logits, labels)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)

        feature_loss = torch.zeros((), device=student_logits.device)
        if self.feature_projection is not None:
            student_hidden = student_outputs.hidden_states[-1][:, 0, :]
            teacher_hidden = teacher_outputs.hidden_states[-1][:, 0, :]
            projected_student = self.feature_projection(student_hidden)
            feature_loss = F.mse_loss(projected_student, teacher_hidden)

        loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
        loss = loss + self.schedule.feature_loss_weight * feature_loss
        return (loss, student_outputs) if return_outputs else loss
