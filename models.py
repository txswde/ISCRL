"""Dual-branch projection-fusion policy network for ISCRL."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from iscrl_components import SimCLRProjector

__all__ = ["ISCRLPolicy", "DSRRL", "SingleHeadSelfAttention"]


class SingleHeadSelfAttention(nn.Module):
    """Single-head scaled dot-product attention over sampled frames."""

    def __init__(self, feature_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        # features: (B, T, D)
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        logits = torch.matmul(q, k.transpose(-1, -2)) / self.feature_dim**0.5
        attention = F.softmax(logits, dim=-1)
        context = torch.matmul(self.dropout(attention), v)
        return self.output(context), attention


class TemporalBranch(nn.Module):
    """Branch-specific projection, bidirectional GRU, attention, and residual."""

    def __init__(self, input_dim: int, state_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        if state_dim % 2:
            raise ValueError("state_dim must be even for a bidirectional GRU")
        self.projection = nn.Linear(input_dim, state_dim)
        self.rnn = nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = SingleHeadSelfAttention(state_dim, dropout=dropout)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        projected = F.relu(self.projection(features), inplace=False)
        recurrent, _ = self.rnn(projected)
        attended, weights = self.attention(recurrent)
        return self.norm(projected + recurrent + attended), weights


class ISCRLPolicy(nn.Module):
    """ISCRL policy with original and invariant semantic branches.

    The branches are encoded independently and fused by a learnable linear
    layer. The returned attention matrix is the equal-weight mean of the two
    single-head branch matrices and is used only for temporal explanation.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        state_dim: int = 512,
        invariant_dim: int = 128,
        projector_hidden_dim: int = 512,
        attention_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.invariant_dim = invariant_dim
        self.simclr_projector = SimCLRProjector(
            input_dim=input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=invariant_dim,
        )
        self.original_branch = TemporalBranch(
            input_dim=input_dim, state_dim=state_dim, dropout=attention_dropout
        )
        self.invariant_branch = TemporalBranch(
            input_dim=invariant_dim, state_dim=state_dim, dropout=attention_dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(state_dim),
        )
        self.policy_head = nn.Linear(state_dim, 1)

    def invariant_features(self, original_features: Tensor, normalize: bool = True) -> Tensor:
        invariant = self.simclr_projector(original_features)
        return F.normalize(invariant, dim=-1) if normalize else invariant

    def set_projector_trainable(self, trainable: bool) -> None:
        for parameter in self.simclr_projector.parameters():
            parameter.requires_grad_(trainable)

    def forward(
        self,
        original_features: Tensor,
        invariant_features: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        if original_features.dim() == 2:
            original_features = original_features.unsqueeze(0)
        if original_features.dim() != 3:
            raise ValueError("original_features must have shape (T, D) or (B, T, D)")
        if invariant_features is None:
            invariant_features = self.invariant_features(original_features)
        elif invariant_features.dim() == 2:
            invariant_features = invariant_features.unsqueeze(0)

        original_state, original_attention = self.original_branch(original_features)
        invariant_state, invariant_attention = self.invariant_branch(invariant_features)
        fused_state = self.fusion(torch.cat((original_state, invariant_state), dim=-1))
        probabilities = torch.sigmoid(self.policy_head(fused_state))
        temporal_attention = 0.5 * (original_attention + invariant_attention)
        branch_attention = {
            "original": original_attention,
            "invariant": invariant_attention,
        }
        return (
            probabilities,
            fused_state,
            temporal_attention,
            invariant_features,
            branch_attention,
        )


# Preserve the original public class name for existing checkpoints/scripts.
DSRRL = ISCRLPolicy
