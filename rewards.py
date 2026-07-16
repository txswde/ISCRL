"""Dual-space semantic reward from Eqs. (2)-(6)."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def single_space_reward(
    features: Tensor,
    selected_indices: Tensor,
    temporal_distance_threshold: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return diversity, representativeness, and their sum in one space."""
    if features.dim() == 3:
        if features.size(0) != 1:
            raise ValueError("reward currently expects one video at a time")
        features = features.squeeze(0)
    selected_indices = selected_indices.reshape(-1).long()
    if selected_indices.numel() == 0:
        zero = features.new_zeros(())
        return zero, zero, zero

    selected = features.index_select(0, selected_indices)
    count = selected_indices.numel()

    if count == 1:
        diversity = features.new_zeros(())
    else:
        normalized = F.normalize(selected, p=2, dim=-1)
        dissimilarity = 1.0 - torch.matmul(normalized, normalized.t())
        temporal_distance = (
            selected_indices[:, None] - selected_indices[None, :]
        ).abs()
        dissimilarity = torch.where(
            temporal_distance > temporal_distance_threshold,
            torch.ones_like(dissimilarity),
            dissimilarity,
        )
        diagonal_mask = ~torch.eye(count, device=features.device, dtype=torch.bool)
        diversity = dissimilarity.masked_select(diagonal_mask).sum() / (count * (count - 1))

    distances = torch.cdist(features, selected, p=2)
    representativeness = torch.exp(-distances.min(dim=1).values.mean())
    total = diversity + representativeness
    return diversity, representativeness, total


def compute_reward(
    original_features: Tensor,
    invariant_features: Tensor,
    actions: Tensor,
    beta: float = 0.10,
    temporal_distance_threshold: int = 20,
    **_: object,
) -> Tensor:
    """Compute (1-beta) * R_orig + beta * R_inv.

    beta controls the invariant semantic-space contribution as stated in Eq.
    (6); beta=0.10 is the manuscript setting.
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    selected_indices = torch.nonzero(actions.detach().reshape(-1) > 0.5, as_tuple=False).flatten()
    if selected_indices.numel() == 0:
        return original_features.new_zeros(())

    _, _, original_reward = single_space_reward(
        original_features.detach(), selected_indices, temporal_distance_threshold
    )
    _, _, invariant_reward = single_space_reward(
        invariant_features.detach(), selected_indices, temporal_distance_threshold
    )
    return (1.0 - beta) * original_reward + beta * invariant_reward
