"""Policy-conditioned smoothed Grad-CAM++ for ISCRL explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import GoogLeNet_Weights, googlenet

from models import ISCRLPolicy


class FrozenGoogLeNetPool5(nn.Module):
    """Expose the last GoogLeNet convolutional maps and 1024-D pool5 feature."""

    def __init__(self, weights: GoogLeNet_Weights = GoogLeNet_Weights.IMAGENET1K_V1) -> None:
        super().__init__()
        self.weights = weights
        # TorchVision loads the published auxiliary-head weights as part of the
        # ImageNet checkpoint; the heads are never called by this feature path.
        self.network = googlenet(weights=weights)
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    @property
    def preprocess(self):
        return self.weights.transforms()

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        net = self.network
        x = net.conv1(image)
        x = net.maxpool1(x)
        x = net.conv2(x)
        x = net.conv3(x)
        x = net.maxpool2(x)
        x = net.inception3a(x)
        x = net.inception3b(x)
        x = net.maxpool3(x)
        x = net.inception4a(x)
        x = net.inception4b(x)
        x = net.inception4c(x)
        x = net.inception4d(x)
        x = net.inception4e(x)
        x = net.maxpool4(x)
        x = net.inception5a(x)
        activations = net.inception5b(x)
        pool5 = net.avgpool(activations).flatten(1)
        return activations, pool5


class SmoothedPolicyGradCAMPP:
    """Explain a policy probability using the same fixed GoogLeNet feature path.

    The target is the ISCRL probability assigned to one sampled frame, not an
    ImageNet class. For every noisy view, the frame's freshly computed pool5
    feature replaces the corresponding pre-extracted feature in the video
    context before the policy forward pass.
    """

    def __init__(
        self,
        policy: ISCRLPolicy,
        feature_extractor: FrozenGoogLeNetPool5 | None = None,
        samples: int = 10,
        noise_std: float = 0.15,
        device: torch.device | str | None = None,
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = policy.to(self.device).eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.extractor = (feature_extractor or FrozenGoogLeNetPool5()).to(self.device).eval()
        self.samples = samples
        self.noise_std = noise_std

    @staticmethod
    def _gradcam_pp(activations: Tensor, gradients: Tensor) -> Tensor:
        gradient_2 = gradients.pow(2)
        gradient_3 = gradient_2 * gradients
        activation_sum = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * gradient_2 + activation_sum * gradient_3
        denominator = torch.where(
            denominator.abs() > 1e-8,
            denominator,
            torch.ones_like(denominator),
        )
        alpha = gradient_2 / denominator
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        return F.relu((weights * activations).sum(dim=1, keepdim=True))

    def _load_image(self, image_path: str | Path) -> Tuple[np.ndarray, Tensor]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # The official weight transform accepts a NumPy-compatible PIL image.
        from PIL import Image

        tensor = self.extractor.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return bgr, tensor

    def explain(
        self,
        image_path: str | Path,
        context_features: Tensor,
        frame_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return BGR overlay, normalized heatmap, and raw spatial score."""
        bgr, clean_input = self._load_image(image_path)
        context = context_features.to(self.device).detach()
        if context.dim() == 2:
            context = context.unsqueeze(0)
        if context.dim() != 3 or context.size(0) != 1:
            raise ValueError("context_features must have shape (T,D) or (1,T,D)")
        if not 0 <= frame_index < context.size(1):
            raise IndexError("frame_index is outside the sampled sequence")

        cam_sum: Tensor | None = None
        for sample in range(self.samples):
            if sample == 0 or self.noise_std == 0:
                image = clean_input.clone()
            else:
                scale = clean_input.detach().amax() - clean_input.detach().amin()
                image = clean_input + torch.randn_like(clean_input) * self.noise_std * scale
            image.requires_grad_(True)
            activations, pool5 = self.extractor(image)
            policy_input = torch.cat(
                (
                    context[:, :frame_index],
                    pool5.unsqueeze(1),
                    context[:, frame_index + 1 :],
                ),
                dim=1,
            )
            probability = self.policy(policy_input)[0][0, frame_index, 0]
            gradients = torch.autograd.grad(probability, activations, retain_graph=False)[0]
            cam = self._gradcam_pp(activations, gradients).detach()
            cam_sum = cam if cam_sum is None else cam_sum + cam

        assert cam_sum is not None
        raw_cam = cam_sum / float(self.samples)
        raw_spatial_score = float(raw_cam.mean().cpu())
        resized = F.interpolate(
            raw_cam,
            size=(bgr.shape[0], bgr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        resized = resized - resized.min()
        resized = resized / resized.max().clamp_min(1e-8)
        heatmap = resized.cpu().numpy()
        color = cv2.applyColorMap(np.uint8(255.0 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.6, color, 0.4, 0.0)
        return overlay, heatmap, raw_spatial_score


# Compatibility alias used by the previous visualization entry point.
SmoothGradCAMpp = SmoothedPolicyGradCAMPP
