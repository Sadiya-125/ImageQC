"""
Classical (non-learned) image-quality features, computed with OpenCV/NumPy
only. Each function is a pure function: it takes one image and returns a
float or small dict, with no shared state and no side effects, so each is
independently unit-testable.

These same functions feed two consumers that both require the exact same
feature order:
  - cnn_model.py's classical-features MLP branch, and
  - anomaly.py's Isolation Forest.
extract_feature_vector() is the single place that fixes that order, via the
module-level FEATURE_NAMES constant. Never reorder FEATURE_NAMES without
retraining everything downstream of it.
"""

from __future__ import annotations

from typing import Dict, Union

import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.stats import skew

ImageLike = Union[np.ndarray, Image.Image]

# Pixels at/near pure black or pure white are treated as "clipped" --
# crushed shadows or blown highlights respectively -- rather than requiring
# an exact 0 or 255 match, since real sensor/JPEG noise rarely lands on the
# exact extreme value even when a region is effectively clipped.
CLIP_LOW_THRESHOLD = 5
CLIP_HIGH_THRESHOLD = 250

# Immerkaer's fast noise estimator kernel (see noise_estimate() docstring).
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

# JPEG encodes images in independent 8x8 blocks.
JPEG_BLOCK_SIZE = 8


def _to_gray_uint8(img: ImageLike) -> np.ndarray:
    """
    Normalizes a PIL Image or numpy array (RGB, RGBA, or already-grayscale)
    into a single 2D uint8 luma array, using OpenCV's standard RGB->gray
    weights. All feature functions in this module operate on luma, so this
    conversion is the shared entry point.
    """
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    else:
        arr = np.asarray(img)
        if arr.ndim == 2:
            return arr.astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def laplacian_variance(img: ImageLike) -> float:
    """
    Sharpness/blur signal: the variance of the image's Laplacian response.

    A sharp image has strong high-frequency edge content, which the
    Laplacian (a second-derivative operator) responds to strongly and
    inconsistently across the image, producing high variance. A blurred
    image suppresses high frequencies, so the Laplacian response is small
    and uniform, producing low variance. This is the standard no-reference
    blur metric from Pech-Pacheco et al., "Diatom autofocusing in
    brightfield microscopy: a comparative study," ICPR 2000.
    """
    gray = _to_gray_uint8(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(img: ImageLike) -> Dict[str, float]:
    """
    Exposure/brightness signal derived from the luma histogram:
      - mean_luma: overall brightness, 0 (black) - 255 (white).
      - pct_clipped_low: fraction of pixels at/near black (<= CLIP_LOW_THRESHOLD),
        i.e. crushed shadows -- a signal for underexposure.
      - pct_clipped_high: fraction of pixels at/near white (>= CLIP_HIGH_THRESHOLD),
        i.e. blown highlights -- a signal for overexposure.
      - luma_skew: skewness of the luma distribution. A well-exposed image's
        luma histogram is roughly symmetric; positive skew indicates a long
        tail toward bright values from a mostly-dark image (consistent with
        underexposure), negative skew the reverse (consistent with
        overexposure).
    """
    gray = _to_gray_uint8(img)
    flat = gray.astype(np.float64).ravel()
    return {
        "mean_luma": float(flat.mean()),
        "pct_clipped_low": float(np.mean(flat <= CLIP_LOW_THRESHOLD)),
        "pct_clipped_high": float(np.mean(flat >= CLIP_HIGH_THRESHOLD)),
        "luma_skew": float(skew(flat)),
    }


def noise_estimate(img: ImageLike) -> float:
    """
    Fast noise-level estimator, implementing Immerkaer's method (J.
    Immerkaer, "Fast Noise Variance Estimation," Computer Vision and Image
    Understanding, 64(2), 1996):

        sigma = sqrt(pi/2) * (1 / (6*(W-2)*(H-2))) * sum(|I * N|)

    where N = [[1,-2,1],[-2,4,-2],[1,-2,1]] and "*" is 2D valid convolution.
    N is constructed so that it has zero response on both flat regions and
    regions that vary linearly (a linear ramp's discrete second derivative
    is zero), so on a noise-free image the response is close to zero
    everywhere; the residual energy that remains is attributable almost
    entirely to noise rather than to genuine image structure.
    """
    gray = _to_gray_uint8(img).astype(np.float64)
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return 0.0
    convolved = convolve2d(gray, _NOISE_KERNEL, mode="valid")
    sigma = np.sqrt(np.pi / 2) * np.sum(np.abs(convolved)) / (6 * (w - 2) * (h - 2))
    return float(sigma)


def contrast_std(img: ImageLike) -> float:
    """Global contrast, measured as the standard deviation of the luma channel."""
    gray = _to_gray_uint8(img)
    return float(gray.astype(np.float64).std())


def jpeg_blockiness(img: ImageLike) -> float:
    """
    8x8 block-edge discontinuity measure, in the spirit of Wang, Sheikh &
    Bovik's no-reference blockiness metric ("No-reference perceptual
    quality assessment of JPEG compressed images," ICIP 2002).

    JPEG quantizes each 8x8 block independently, which tends to introduce a
    small intensity discontinuity exactly at block boundaries that isn't
    present at other pixel offsets. This computes the mean absolute
    pixel-to-pixel difference at 8-pixel-aligned boundaries minus the mean
    absolute difference at non-aligned positions, averaged over the
    horizontal and vertical directions -- a value near zero indicates no
    block structure, a clearly positive value indicates block artifacts.
    """
    gray = _to_gray_uint8(img).astype(np.float64)
    h, w = gray.shape

    h_diff = np.abs(np.diff(gray, axis=1))  # shape (h, w-1): columns 1..w-1
    col_is_boundary = (np.arange(1, w) % JPEG_BLOCK_SIZE == 0)
    h_boundary = float(h_diff[:, col_is_boundary].mean()) if col_is_boundary.any() else 0.0
    h_nonboundary = float(h_diff[:, ~col_is_boundary].mean()) if (~col_is_boundary).any() else 0.0

    v_diff = np.abs(np.diff(gray, axis=0))  # shape (h-1, w): rows 1..h-1
    row_is_boundary = (np.arange(1, h) % JPEG_BLOCK_SIZE == 0)
    v_boundary = float(v_diff[row_is_boundary, :].mean()) if row_is_boundary.any() else 0.0
    v_nonboundary = float(v_diff[~row_is_boundary, :].mean()) if (~row_is_boundary).any() else 0.0

    return ((h_boundary - h_nonboundary) + (v_boundary - v_nonboundary)) / 2.0


# Fixed, documented order for the combined feature vector. This order is a
# contract shared between training (ml_training/) and inference
# (backend/app/ml/) -- changing it requires re-fitting anomaly.py's
# IsolationForest and retraining cnn_model.py's classical-features branch.
FEATURE_NAMES = [
    "laplacian_variance",
    "mean_luma",
    "pct_clipped_low",
    "pct_clipped_high",
    "luma_skew",
    "noise_estimate",
    "contrast_std",
    "jpeg_blockiness",
]


def extract_feature_vector(img: ImageLike) -> np.ndarray:
    """
    Combines all classical image-quality features into a single
    fixed-length float64 vector, ordered per FEATURE_NAMES.
    """
    exposure = exposure_stats(img)
    values = [
        laplacian_variance(img),
        exposure["mean_luma"],
        exposure["pct_clipped_low"],
        exposure["pct_clipped_high"],
        exposure["luma_skew"],
        noise_estimate(img),
        contrast_std(img),
        jpeg_blockiness(img),
    ]
    return np.array(values, dtype=np.float64)
