"""RoboDojo numeric adapter for original LoLA-v07 encoders and five-stream DiT.

This is the action core, not a complete VLM policy. Callers must supply
post-bridge visual/language tokens and their global condition explicitly.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lerobot.policies.lola.modeling_lola import LoLADiT
from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config
from lerobot.policies.lola_v07.modeling_lola_v07 import (
    LolaV07ActionEncoder,
    LolaV07StateEncoder,
)

from numeric_dataset import ARM_INDICES, GRIPPER_INDICES, sha256


def joint_config(*, small_test: bool = False) -> LoLAV07Config:
    options: dict[str, Any] = {}
    if small_test:
        options = dict(
            dit_hidden_size=64, dit_num_heads=4,
            dit_double_layers=1, dit_single_layers=1,
            action_bottleneck_dim=32, grip_bottleneck_dim=16,
            state_bottleneck_dim=32, state_grip_bottleneck_dim=16,
        )
    return LoLAV07Config(
        action_dim=14, state_dim=14, gripper_dim_indices=(6, 13),
        action_chunk_size=8, pred_chunk_size=16,
        history_type="state", state_encoder_mode="unified",
        load_full_history=True, completed_tasks_history_len=4,
        hist_action_token_drop_rate=0, transition_mask_rate=0,
        transition_summary_drop_rate=0, task_summary_drop_rate=0,
        history_architecture_version=1, history_tokenization_mode="segment_summary",
        vlm_backbone="cosmos3_nano",
        use_special_tokens=True, use_previous_task_end=True,
        use_state_condition=True, device="cpu", dtype="float32",
        vlm_unfreeze_v_loss_threshold=0, num_inference_steps=3,
        **options,
    )


class JointNormalizer(nn.Module):
    def __init__(self, statistics: dict[str, Any]) -> None:
        super().__init__()
        if statistics["partition"] != "train":
            raise ValueError("Expected train-only normalization statistics")
        for key, prefix in (("observation.state", "state"), ("action", "action")):
            mean = torch.tensor(statistics["features"][key]["mean"], dtype=torch.float32)
            scale = torch.tensor(statistics["features"][key]["std"], dtype=torch.float32)
            if mean.shape != (14,) or scale.shape != (14,):
                raise ValueError("Expected fourteen normalization dimensions")
            if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Normalization requires finite positive standard deviations")
            mean[list(GRIPPER_INDICES)] = 0
            scale[list(GRIPPER_INDICES)] = 1
            self.register_buffer(f"{prefix}_mean", mean)
            self.register_buffer(f"{prefix}_scale", scale)

    def state(self, value: Tensor) -> Tensor:
        return (value - self.state_mean) / self.state_scale

    def action(self, value: Tensor) -> Tensor:
        return (value - self.action_mean) / self.action_scale

    def raw_action(self, value: Tensor) -> Tensor:
        return value * self.action_scale + self.action_mean


def numeric_batch(samples: list[dict[str, Any]]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    keys = ("observation.state", "history.state", "history.valid_mask",
            "action", "action.valid_mask", "frame_index")
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples])) for key in keys}


class LoLAJointCore(nn.Module):
    def __init__(self, config: LoLAV07Config, statistics: dict[str, Any]) -> None:
        super().__init__()
        if (config.action_dim != 14 or config.state_dim != 14
                or config.gripper_dim_indices_abs != (6, 13)
                or config.state_encoder_mode != "unified"
                or config.history_tokenization_mode != "segment_summary"
                or config.pred_chunk_size != 16 or config.action_chunk_size != 8):
            raise ValueError("Incompatible RoboDojo joint configuration")
        self.config = config
        self.normalizer = JointNormalizer(statistics)
        self.action_encoder = LolaV07ActionEncoder(config).float()
        self.state_encoder = LolaV07StateEncoder(config).float()
        # Match the raw-zero history convention using exactly our transform.
        with torch.no_grad():
            self.state_encoder.history_null_state.copy_(
                self.normalizer.state(torch.zeros(14))
            )
        self.dit = LoLADiT(config).float()
        self.arm_dit_to_latent = nn.Linear(config.dit_hidden_size, config.action_bottleneck_dim)
        self.grip_dit_to_latent = nn.Linear(config.dit_hidden_size, config.grip_bottleneck_dim)
        with torch.no_grad():
            for projection, decoder in (
                (self.arm_dit_to_latent, self.action_encoder.arm_dec),
                (self.grip_dit_to_latent, self.action_encoder.grip_dec),
            ):
                projection.weight.copy_(torch.linalg.pinv(decoder.weight, rcond=1e-3))
                projection.bias.zero_()

    def conditions(
        self, batch: dict[str, Tensor], visual: Tensor, global_condition: Tensor,
        visual_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state = batch["observation.state"]
        history = batch["history.state"]
        mask = batch["history.valid_mask"]
        frame_index = batch["frame_index"]
        b, width = state.shape[0], self.config.dit_hidden_size
        if (state.shape != (b, 14) or history.shape != (b, 32, 14)
                or mask.shape != (b, 32) or mask.dtype != torch.bool
                or not mask[:, -1].all() or (mask[:, :-1] & ~mask[:, 1:]).any()
                or not torch.equal(history[:, -1], state)):
            raise ValueError("History must be left-padded and end at the current state")
        if (frame_index.shape != (b,) or frame_index.dtype != torch.int64
                or (frame_index < 0).any()
                or not torch.equal(mask.sum(dim=1), (frame_index + 1).clamp_max(32))):
            raise ValueError("Frame indices and history lengths disagree")
        if (visual.ndim != 3 or visual.shape[0] != b or visual.shape[2] != width
                or global_condition.shape != (b, width)
                or visual_mask.shape != visual.shape[:2] or visual_mask.dtype != torch.bool
                or not visual_mask.any(dim=1).all()):
            raise ValueError("Explicit post-bridge visual conditions and valid masks are required")
        if not all(torch.isfinite(x).all() for x in (state, history, visual, global_condition)):
            raise ValueError("Non-finite model inputs")
        normalized = self.normalizer.state(history)
        null = self.state_encoder.history_null_state.view(1, 1, 14)
        normalized = torch.where(mask[..., None], normalized, null)
        summaries = self.state_encoder.encode_segment_summaries(
            null.expand(b, 32, 14), torch.zeros_like(mask),
            torch.zeros(b, dtype=torch.long, device=state.device),
            normalized, mask, frame_index + 1,
        )
        def pack(transition: Tensor, task: Tensor) -> Tensor:
            return torch.cat((
                self.dit.hist_start_emb.expand(b, -1, -1),
                transition[:, None], self.dit.previous_task_end_emb.expand(b, -1, -1),
                task[:, None], self.dit.hist_end_emb.expand(b, -1, -1),
            ), dim=1)
        arm = pack(summaries["transition_arm"], summaries["task_arm"])
        grip = pack(summaries["transition_grip"], summaries["task_grip"])
        visual = torch.cat((self.dit.vlm_start_emb.expand(b, -1, -1),
                            visual, self.dit.vlm_end_emb.expand(b, -1, -1)), dim=1)
        ends = torch.ones(b, 1, dtype=torch.bool, device=state.device)
        visual_mask = torch.cat((ends, visual_mask, ends), dim=1)
        missing = torch.zeros_like(ends)
        history_mask = torch.cat((ends, missing, missing, ends, ends), dim=1)
        # DiT mask order: visual, gripper history, arm history, gripper target, arm target.
        full_mask = torch.cat((visual_mask, history_mask, history_mask, ends.expand(-1, 4)), dim=1)
        return (torch.cat((arm, grip), dim=1), visual, global_condition,
                self.normalizer.state(state), full_mask)

    def denoise(
        self, arm: Tensor, grip: Tensor, time: Tensor,
        condition: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        history, visual, global_condition, state, mask = condition
        tokens = self.action_encoder.decode(arm, grip, add_modality_emb=False)
        clean = self.dit(
            tokens, history, visual, global_condition, time,
            state_emb=state, return_chunks=True,
            joint_attention_kwargs={"attention_mask": mask},
        )
        arm_tokens, grip_tokens = clean.chunk(2, dim=1)
        return self.arm_dit_to_latent(arm_tokens), self.grip_dit_to_latent(grip_tokens)

    def readout(self, arm: Tensor, grip: Tensor) -> Tensor:
        arm_tokens, grip_tokens = self.action_encoder.decode(arm, grip).chunk(2, dim=1)
        b = arm.shape[0]
        result = arm.new_zeros(b, 16, 14)
        result[..., list(ARM_INDICES)] = self.dit.arm_out_proj(arm_tokens).reshape(b, 16, 12)
        result[..., list(GRIPPER_INDICES)] = torch.sigmoid(
            self.dit.gripper_out_proj(grip_tokens).reshape(b, 16, 2)
        )
        return result

    def forward(
        self, batch: dict[str, Tensor], visual: Tensor, global_condition: Tensor,
        visual_mask: Tensor, time: Tensor,
        noise: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        actions, valid = batch["action"], batch["action.valid_mask"]
        b = actions.shape[0]
        if (actions.shape != (b, 16, 14) or valid.shape != (b, 16)
                or valid.dtype != torch.bool or not valid[:, 0].all()
                or (~valid[:, :-1] & valid[:, 1:]).any()
                or not torch.isfinite(actions[valid]).all()):
            raise ValueError("Expected finite targets with right-padded valid masks")
        grips = actions[..., list(GRIPPER_INDICES)][valid]
        # Published data contain tiny negative roundoff around raw zero.
        if (grips < -1e-6).any() or (grips > 1 + 1e-6).any():
            raise ValueError("Continuous gripper targets outside the dataset range")
        if time.shape != (b,) or not torch.isfinite(time).all() or (time <= 0).any() or (time > 1).any():
            raise ValueError("Flow time must be finite and in (0, 1]")
        condition = self.conditions(batch, visual, global_condition, visual_mask)
        normalized = self.normalizer.action(actions)
        normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
        _, target_arm, target_grip = self.action_encoder(normalized, return_latent=True)
        if noise is None:
            noise = (torch.randn_like(target_arm), torch.randn_like(target_grip))
        if any(n.shape != z.shape or not torch.isfinite(n).all()
               for n, z in zip(noise, (target_arm, target_grip), strict=True)):
            raise ValueError("Invalid latent noise")
        t = time[:, None, None]
        noisy = tuple((1 - t) * z + t * n for z, n in zip((target_arm, target_grip), noise))
        clean_arm, clean_grip = self.denoise(*noisy, time, condition)
        chunk_valid = valid.reshape(b, 2, 8).all(dim=-1)
        flow_losses = []
        for z, n, noisy_z, clean in zip(
            (target_arm, target_grip), noise, noisy, (clean_arm, clean_grip), strict=True,
        ):
            error = ((noisy_z - clean) / t.clamp_min(1e-5) - (n - z)).square().mean(-1)
            # A partial chunk's latent mixes real and padded steps, so it has no
            # unambiguous per-step flow mask. Supervise flow only on full chunks.
            flow_losses.append((error * chunk_valid).sum() / chunk_valid.sum().clamp_min(1))
        prediction = self.readout(clean_arm, clean_grip)
        def masked_mean(error: Tensor) -> Tensor:
            return (error * valid[..., None]).sum() / (valid.sum() * error.shape[-1])
        arm_loss = masked_mean(F.huber_loss(
            prediction[..., list(ARM_INDICES)], normalized[..., list(ARM_INDICES)], reduction="none",
        ))
        grip_loss = masked_mean((
            prediction[..., list(GRIPPER_INDICES)] - normalized[..., list(GRIPPER_INDICES)]
        ).square())
        flow = (flow_losses[0] + flow_losses[1]) / 2
        loss = flow + self.config.action_loss_weight * arm_loss + self.config.gripper_loss_weight * grip_loss
        return {"loss": loss, "flow_loss": flow, "arm_loss": arm_loss, "gripper_loss": grip_loss,
                "prediction": prediction, "full_flow_chunks": chunk_valid.sum()}

    @torch.no_grad()
    def sample(
        self, batch: dict[str, Tensor], visual: Tensor, global_condition: Tensor,
        visual_mask: Tensor, noise: tuple[Tensor, Tensor],
    ) -> Tensor:
        condition = self.conditions(batch, visual, global_condition, visual_mask)
        b = batch["observation.state"].shape[0]
        for value, width in zip(noise, (self.config.action_bottleneck_dim, self.config.grip_bottleneck_dim)):
            if value.shape != (b, 2, width) or not torch.isfinite(value).all():
                raise ValueError("Invalid sampling noise")
        arm, grip = (value.clone() for value in noise)
        for step in range(self.config.num_inference_steps):
            time = arm.new_full((b,), 1 - step / self.config.num_inference_steps)
            clean_arm, clean_grip = self.denoise(arm, grip, time, condition)
            scale = time[:, None, None] * self.config.num_inference_steps
            arm = arm - (arm - clean_arm) / scale
            grip = grip - (grip - clean_grip) / scale
        return self.normalizer.raw_action(self.readout(arm, grip))


def load_train_statistics(path: Path, split: Path) -> dict[str, Any]:
    stats = json.loads(path.read_text())
    if stats["split_sha256"] != sha256(split):
        raise ValueError("Normalization belongs to a different split")
    return stats
