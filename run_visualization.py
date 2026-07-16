"""Generate temporal and spatial ISCRL explanations for one video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import ISCRLPolicy
from visualization.grad_cam import SmoothedPolicyGradCAMPP


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ISCRL explanation pipeline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--frame-dir", required=True, help="directory containing raw frames")
    parser.add_argument("--output", default="visualization_results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--explain-all",
        action="store_true",
        help="run spatial explanations for every sampled frame (slow)",
    )
    parser.add_argument("--smooth-samples", type=int, default=10)
    parser.add_argument("--smooth-noise", type=float, default=0.15)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _strip_parallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_model(checkpoint_path: str, device: torch.device) -> ISCRLPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = ISCRLPolicy(
        input_dim=int(saved_args.get("input_dim", 1024)),
        state_dim=int(saved_args.get("hidden_dim", 512)),
        invariant_dim=int(saved_args.get("invariant_dim", 128)),
        projector_hidden_dim=int(saved_args.get("projector_hidden_dim", 512)),
        attention_dropout=float(saved_args.get("attention_dropout", 0.5)),
    )
    model.load_state_dict(_strip_parallel_prefix(state_dict))
    return model.to(device).eval()


def max_normalize(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(output)
    if not valid.any():
        return output
    maximum = np.max(output[valid])
    output[valid] = output[valid] / maximum if maximum > 0 else 0.0
    return output


def resolve_frame(
    frame_dir: Path,
    feature_index: int,
    original_index: int,
    sequence_length: int,
) -> Path:
    candidates = (
        f"frame_{original_index:04d}.jpg",
        f"frame_{original_index:06d}.jpg",
        f"img_{original_index + 1:05d}.jpg",
        f"img_{original_index:05d}.jpg",
    )
    for filename in candidates:
        candidate = frame_dir / filename
        if candidate.exists():
            return candidate
    images = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) == sequence_length:
        return images[feature_index]
    if 0 <= original_index < len(images):
        return images[original_index]
    raise FileNotFoundError(
        f"could not map sampled frame {feature_index} (original index {original_index}) "
        f"inside {frame_dir}"
    )


def save_temporal_plots(
    probabilities: np.ndarray,
    temporal_scores: np.ndarray,
    attention: np.ndarray,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(probabilities, color="green", label="Frame-selection probability")
    plt.plot(temporal_scores, color="blue", label="Temporal explanation score")
    plt.xlabel("Sampled frame index")
    plt.ylabel("Normalized score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_scores.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 7))
    plt.imshow(attention, cmap="hot", interpolation="nearest", aspect="auto")
    plt.xlabel("Key frame")
    plt.ylabel("Query frame")
    plt.colorbar(label="Attention")
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_attention_matrix.png", dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.omega <= 1:
        raise ValueError("--omega must be in [0,1]")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    with h5py.File(args.dataset, "r") as dataset:
        group = dataset[args.key]
        features = np.asarray(group["features"], dtype=np.float32)
        picks = (
            np.asarray(group["picks"], dtype=np.int64)
            if "picks" in group
            else np.arange(len(features), dtype=np.int64)
        )
    sequence = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities, _, attention, _, _ = model(sequence)
    probabilities_np = probabilities[0, :, 0].cpu().numpy()
    attention_np = attention[0].cpu().numpy()
    # Eq. (13): mean attention received by each frame (column mean).
    temporal_raw = attention_np.mean(axis=0)
    temporal_normalized = max_normalize(temporal_raw)

    output_dir = Path(args.output) / args.key.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_temporal_plots(
        probabilities_np,
        temporal_normalized,
        attention_np,
        output_dir,
    )

    if args.explain_all:
        indices = np.arange(len(features))
    else:
        indices = np.argsort(probabilities_np)[::-1][: min(args.top_k, len(features))]
    explainer = SmoothedPolicyGradCAMPP(
        model,
        samples=args.smooth_samples,
        noise_std=args.smooth_noise,
        device=device,
    )
    frame_dir = Path(args.frame_dir)
    spatial_raw = np.full(len(features), np.nan, dtype=np.float64)
    frame_paths: Dict[int, str] = {}
    for rank, feature_index in enumerate(indices, start=1):
        original_index = int(picks[min(feature_index, len(picks) - 1)])
        image_path = resolve_frame(
            frame_dir,
            int(feature_index),
            original_index,
            len(features),
        )
        overlay, heatmap, spatial_score = explainer.explain(
            image_path,
            sequence,
            int(feature_index),
        )
        spatial_raw[feature_index] = spatial_score
        frame_paths[int(feature_index)] = str(image_path)
        cv2.imwrite(str(output_dir / f"spatial_rank{rank:03d}_frame{feature_index:05d}.jpg"), overlay)
        cv2.imwrite(
            str(output_dir / f"heatmap_rank{rank:03d}_frame{feature_index:05d}.png"),
            np.uint8(255.0 * heatmap),
        )

    # Eqs. (14)-(15): separate max normalization, followed by weighted fusion.
    spatial_normalized = max_normalize(spatial_raw)
    combined = args.omega * temporal_normalized + (1.0 - args.omega) * spatial_normalized
    np.savez_compressed(
        output_dir / "explanation_scores.npz",
        probabilities=probabilities_np,
        temporal_attention=attention_np,
        temporal_raw=temporal_raw,
        temporal_normalized=temporal_normalized,
        spatial_raw=spatial_raw,
        spatial_normalized=spatial_normalized,
        combined=combined,
        omega=np.asarray(args.omega),
    )
    with (output_dir / "explained_frames.json").open("w", encoding="utf-8") as file:
        json.dump(frame_paths, file, indent=2)
    print(f"Saved explanations to {output_dir}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
