"""
Grad-CAM (Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization," ICCV 2017) against the
MobileNetV3-Small branch of HybridQualityModel, for per-head visual
explanations ("why did the model think this was blurry" vs "why noisy",
localized spatially on the image) -- this project's explainability
mechanism (BUILD_SPEC.md §10 / the assessment's Explainability requirement)
alongside the classical feature values themselves.

Hooks the layer returned by HybridQualityModel.get_gradcam_target_layer()
(the last Conv2dNormActivation block in the backbone's `features`, i.e. the
final spatial feature map before global pooling), backprops from a single
chosen output head, and produces a class-activation heatmap overlaid on the
original image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import functional as TF

from .classical_features import extract_feature_vector
from .cnn_model import HybridQualityModel

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_SCALER_PATH = WEIGHTS_DIR / "feature_scaler.joblib"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HEATMAP_ALPHA = 0.45  # overlay opacity: 0 = original image only, 1 = heatmap only


class _GradCAMHooks:
    """Captures the target layer's forward activations and backward gradients."""

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _compute_cam(model: HybridQualityModel, image_tensor: torch.Tensor, classical_tensor: torch.Tensor, head_name: str) -> np.ndarray:
    """Returns a (H, W) float32 array in [0, 1] at the target layer's spatial resolution."""
    hooks = _GradCAMHooks(model.get_gradcam_target_layer())
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(image_tensor, classical_tensor)
        score = outputs[head_name].sum()
        score.backward()

        # Grad-CAM: channel-wise weight = global-average-pooled gradient,
        # then a weighted sum of activation channels, then ReLU (only
        # positive influence on the target head is visualized).
        weights = hooks.gradients.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        cam = torch.relu((weights * hooks.activations).sum(dim=1, keepdim=True))  # (N, 1, h, w)
        cam = cam[0, 0].detach().cpu().numpy()

        cam -= cam.min()
        max_val = cam.max()
        if max_val > 1e-8:
            cam /= max_val
        return cam
    finally:
        hooks.remove()


def generate_gradcam(
    model: HybridQualityModel,
    image: Image.Image,
    head_name: str,
    scaler: Optional[StandardScaler] = None,
) -> Image.Image:
    """
    Produces a Grad-CAM heatmap overlay for a single PIL image and a single
    output head (one of cnn_model.ISSUE_HEADS or cnn_model.QUALITY_SCORE_HEAD).

    scaler: the fitted StandardScaler for classical features (see
    ml_training/dataset.py's fit_feature_scaler). If omitted, loads
    backend/app/ml/weights/feature_scaler.joblib -- the artifact
    ml_training/train.py fits and saves.
    """
    model.eval()
    device = next(model.parameters()).device

    if scaler is None:
        import joblib

        scaler = joblib.load(DEFAULT_SCALER_PATH)

    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    raw_features = extract_feature_vector(resized)
    scaled_features = scaler.transform(raw_features.reshape(1, -1))[0]
    classical_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)

    image_tensor = TF.normalize(TF.to_tensor(resized), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    cam = _compute_cam(model, image_tensor, classical_tensor, head_name)

    cam_resized = cv2.resize(cam, (resized.width, resized.height))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = np.array(resized, dtype=np.float32)
    overlay = (1 - HEATMAP_ALPHA) * base + HEATMAP_ALPHA * heatmap_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


if __name__ == "__main__":
    # Demo: 2-3 sample images per issue category from the KADID-10k test
    # split, Grad-CAM'd against their own positive head, saved for visual
    # review before this gets wired into the API.
    _BACKEND_DIR = Path(__file__).resolve().parents[2]
    _REPO_ROOT = _BACKEND_DIR.parent
    sys.path.insert(0, str(_REPO_ROOT / "ml_training"))
    sys.path.insert(0, str(_BACKEND_DIR))

    import joblib

    from dataset import IMAGES_DIR, load_split_dataframe  # noqa: E402
    from app.ml.cnn_model import ISSUE_HEADS  # noqa: E402

    OUT_DIR = _REPO_ROOT / "ml_training" / "notebooks" / "gradcam_samples"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "mobilenetv3_iqa.pt"
    model = HybridQualityModel.from_pretrained(MODEL_WEIGHTS_PATH)
    scaler = joblib.load(DEFAULT_SCALER_PATH)

    test_df = load_split_dataframe("test")

    SAMPLES_PER_HEAD = 3
    for head in ISSUE_HEADS:
        positives = test_df[test_df[head] == 1]
        chosen = positives.sample(n=min(SAMPLES_PER_HEAD, len(positives)), random_state=42) if len(positives) else positives
        print(f"{head}: {len(chosen)} sample(s) (from {len(positives)} positive test images)")
        for _, row in chosen.iterrows():
            img_path = IMAGES_DIR / row["filename"]
            pil_image = Image.open(img_path).convert("RGB")
            overlay = generate_gradcam(model, pil_image, head, scaler=scaler)
            out_path = OUT_DIR / f"{head}_{Path(row['filename']).stem}.png"
            overlay.save(out_path)
            print(f"  saved {out_path.name}")

    print(f"\nSaved Grad-CAM overlays to {OUT_DIR}")
