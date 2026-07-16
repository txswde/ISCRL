"""Core ISCRL components defined in the accompanying manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureAugmenter(nn.Module):
    """Create SimCLR views from pre-extracted 1024-D frame features.

    The manuscript uses Gaussian feature noise (sigma=0.03), random feature
    masking (p=0.20), and local temporal-index jitter within +/-1 sampled
    frame. Temporal jitter is applied before noise and masking.
    """

    def __init__(
        self,
        noise_std: float = 0.03,
        mask_prob: float = 0.20,
        temporal_jitter: int = 1,
    ) -> None:
        super().__init__()
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 <= mask_prob < 1:
            raise ValueError("mask_prob must be in [0, 1)")
        if temporal_jitter < 0:
            raise ValueError("temporal_jitter must be non-negative")
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.temporal_jitter = temporal_jitter

    def _temporal_index_jitter(self, features: Tensor) -> Tensor:
        # features: (B, T, D)
        if self.temporal_jitter == 0 or features.size(1) <= 1:
            return features
        batch, steps, dim = features.shape
        offsets = torch.randint(
            -self.temporal_jitter,
            self.temporal_jitter + 1,
            (batch, steps),
            device=features.device,
        )
        indices = torch.arange(steps, device=features.device).unsqueeze(0)
        indices = (indices + offsets).clamp_(0, steps - 1)
        gather_index = indices.unsqueeze(-1).expand(batch, steps, dim)
        return torch.gather(features, dim=1, index=gather_index)

    def forward(self, features: Tensor) -> Tensor:
        squeeze_batch = features.dim() == 2
        if squeeze_batch:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError("features must have shape (T, D) or (B, T, D)")

        augmented = self._temporal_index_jitter(features)
        if self.noise_std:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std
        if self.mask_prob:
            mask = torch.rand_like(augmented).ge(self.mask_prob)
            augmented = augmented * mask.to(augmented.dtype)
        return augmented.squeeze(0) if squeeze_batch else augmented


class SimCLRProjector(nn.Module):
    """The 1024-512-128 projection head reported in Table 1."""

    def __init__(
        self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class SimCLRObjective(nn.Module):
    """Feature-level dual-view InfoNCE objective.

    Augmentation is applied to the original 1024-D features before each view
    is mapped through the shared projection head.
    """

    def __init__(
        self,
        projector: SimCLRProjector,
        temperature: float = 0.5,
        noise_std: float = 0.03,
        mask_prob: float = 0.20,
        temporal_jitter: int = 1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.projector = projector
        self.temperature = temperature
        self.augmenter = FeatureAugmenter(
            noise_std=noise_std,
            mask_prob=mask_prob,
            temporal_jitter=temporal_jitter,
        )

    def forward(self, original_features: Tensor) -> Tensor:
        if original_features.dim() == 2:
            original_features = original_features.unsqueeze(0)
        if original_features.dim() != 3:
            raise ValueError("original_features must have shape (T, D) or (B, T, D)")

        view_i = self.augmenter(original_features)
        view_j = self.augmenter(original_features)
        z_i = F.normalize(self.projector(view_i).flatten(0, 1), dim=-1)
        z_j = F.normalize(self.projector(view_j).flatten(0, 1), dim=-1)

        if z_i.size(0) < 2:
            raise ValueError("InfoNCE requires at least two frame samples")

        representations = torch.cat((z_i, z_j), dim=0)
        logits = torch.matmul(representations, representations.t()) / self.temperature
        sample_count = z_i.size(0)
        self_mask = torch.eye(2 * sample_count, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
        targets = torch.arange(2 * sample_count, device=logits.device)
        targets = (targets + sample_count) % (2 * sample_count)
        return F.cross_entropy(logits, targets)


@dataclass
class AIMState:
    """Per-video state required by Algorithm 1."""

    intervention: float
    previous_intervention: float
    previous_reward: float = 0.0
    baseline: float = 0.0
    gain: float = 0.0


class AIMController:
    """Reward-feedback-driven AIM controller from Eqs. (10)-(12)."""

    def __init__(
        self,
        intervention_min: float = 0.10,
        intervention_max: float = 0.50,
        penalty: float = 0.10,
        increase_factor: float = 1.10,
        decrease_factor: float = 0.90,
        base_clip: float = 5.0,
        baseline_decay: float = 0.90,
    ) -> None:
        if not 0 < intervention_min <= intervention_max:
            raise ValueError("invalid intervention range")
        if not 0 < decrease_factor <= 1:
            raise ValueError("decrease_factor must be in (0, 1]")
        if increase_factor < 1:
            raise ValueError("increase_factor must be >= 1")
        if not 0 <= baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        self.intervention_min = intervention_min
        self.intervention_max = intervention_max
        self.penalty = penalty
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.base_clip = base_clip
        self.baseline_decay = baseline_decay
        self._states: Dict[str, AIMState] = {}

    def state_for(self, video_id: str) -> AIMState:
        if video_id not in self._states:
            self._states[video_id] = AIMState(
                intervention=self.intervention_min,
                previous_intervention=self.intervention_min,
            )
        return self._states[video_id]

    def advantage(self, video_id: str, reward: float) -> float:
        return reward - self.state_for(video_id).baseline

    def update(self, video_id: str, reward: float) -> Tuple[float, float, AIMState]:
        state = self.state_for(video_id)
        delta_reward = reward - state.previous_reward
        delta_intervention = abs(state.intervention - state.previous_intervention)
        gain = delta_reward - self.penalty * delta_intervention

        previous_intervention = state.intervention
        if gain > 0:
            state.intervention = min(
                state.intervention * self.increase_factor, self.intervention_max
            )
        else:
            state.intervention = max(
                state.intervention * self.decrease_factor, self.intervention_min
            )

        state.previous_intervention = previous_intervention
        state.previous_reward = reward
        state.gain = gain

        clip_threshold = self.base_clip / (1.0 + state.intervention)
        learning_rate_scale = 1.0 / (1.0 + state.intervention)
        return clip_threshold, learning_rate_scale, state

    def update_baseline(self, video_id: str, reward: float) -> float:
        state = self.state_for(video_id)
        state.baseline = (
            self.baseline_decay * state.baseline
            + (1.0 - self.baseline_decay) * reward
        )
        return state.baseline

    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return {key: vars(value).copy() for key, value in self._states.items()}

    def load_state_dict(self, values: Optional[Dict[str, Dict[str, float]]]) -> None:
        self._states.clear()
        if values:
            self._states.update({key: AIMState(**value) for key, value in values.items()})


# Backward-compatible names used by earlier scripts.
SimCLRLoss = SimCLRObjective
AIMOptimizer = AIMController
