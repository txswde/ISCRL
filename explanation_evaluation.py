"""Utilities for the explanation protocols reported in the manuscript.

These functions deliberately contain no training logic. They construct the
perturbed/deleted feature sequences and compute overlap or metric drops; the
caller then evaluates those sequences with the same checkpoint and dataset
evaluation path used for the unmodified video.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def gaussian_feature_noise(
    features: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the sigma=0.01/0.03 feature-noise stability perturbation."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    generator = rng or np.random.default_rng()
    return np.asarray(features) + generator.normal(0.0, sigma, size=features.shape)


def temporal_index_jitter(
    features: np.ndarray,
    radius: int = 1,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply local temporal-index jitter and return the sampled source indices."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    features = np.asarray(features)
    generator = rng or np.random.default_rng()
    offsets = generator.integers(-radius, radius + 1, size=len(features))
    indices = np.clip(np.arange(len(features)) + offsets, 0, len(features) - 1)
    return features[indices], indices


def frame_dropout(
    features: np.ndarray,
    fraction: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop 5% of candidate frames and return the retained original indices."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0,1)")
    features = np.asarray(features)
    generator = rng or np.random.default_rng()
    keep_count = max(1, len(features) - int(np.ceil(len(features) * fraction)))
    retained = np.sort(generator.choice(len(features), size=keep_count, replace=False))
    return features[retained], retained


def top_fraction_indices(scores: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Indices of the highest-scoring fraction (at least one frame)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    scores = np.asarray(scores).reshape(-1)
    valid = np.flatnonzero(np.isfinite(scores))
    count = max(1, int(np.ceil(len(valid) * fraction)))
    return valid[np.argsort(scores[valid])[::-1][:count]]


def top_fraction_overlap(
    reference_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    fraction: float = 0.10,
    perturbed_to_reference: np.ndarray | None = None,
) -> float:
    """Top-10% overlap, optionally mapping a shortened sequence to the original."""
    reference = set(top_fraction_indices(reference_scores, fraction).tolist())
    perturbed = top_fraction_indices(perturbed_scores, fraction)
    if perturbed_to_reference is not None:
        perturbed = np.asarray(perturbed_to_reference)[perturbed]
    return len(reference.intersection(perturbed.tolist())) / float(len(reference))


def deletion_indices(
    scores: np.ndarray,
    fraction: float,
    strategy: str = "high",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select high-, low-, or random-score frames for the deletion protocol."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0,1)")
    scores = np.asarray(scores).reshape(-1)
    count = int(np.ceil(len(scores) * fraction))
    if strategy == "high":
        return np.argsort(scores)[::-1][:count]
    if strategy == "low":
        return np.argsort(scores)[:count]
    if strategy == "random":
        generator = rng or np.random.default_rng()
        return np.sort(generator.choice(len(scores), size=count, replace=False))
    raise ValueError("strategy must be 'high', 'low', or 'random'")


def delete_frames(
    features: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    strategy: str = "high",
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the retained candidate sequence and its original indices."""
    removed = deletion_indices(scores, fraction, strategy, rng)
    keep = np.ones(len(features), dtype=bool)
    keep[removed] = False
    retained = np.flatnonzero(keep)
    return np.asarray(features)[retained], retained


def metric_drops(
    original: Dict[str, float],
    after_deletion: Dict[str, float],
) -> Dict[str, float]:
    """Compute F1, Spearman, and Kendall drops using matching metric names."""
    if original.keys() != after_deletion.keys():
        raise ValueError("metric dictionaries must contain identical keys")
    return {name: original[name] - after_deletion[name] for name in original}
