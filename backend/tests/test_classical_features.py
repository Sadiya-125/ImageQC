"""
Unit tests for app/ml/classical_features.py against known synthetic inputs
-- pure functions, no DB/model/network involved.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.classical_features import (  # noqa: E402
    FEATURE_NAMES,
    contrast_std,
    exposure_stats,
    extract_feature_vector,
    jpeg_blockiness,
    laplacian_variance,
    noise_estimate,
)

RNG = np.random.default_rng(42)


def _sharp_checkerboard(size: int = 128, cell: int = 8) -> np.ndarray:
    """A high-frequency-content image: lots of hard edges, no blur."""
    grid = (np.indices((size, size)).sum(axis=0) // cell) % 2
    img = (grid * 255).astype(np.uint8)
    return np.stack([img] * 3, axis=-1)


def _blurred(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _solid(value: int, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _noisy(img: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    noise = RNG.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


class TestLaplacianVariance:
    def test_blur_reduces_sharpness(self):
        sharp = _sharp_checkerboard()
        blurred = _blurred(sharp, ksize=15)
        assert laplacian_variance(sharp) > laplacian_variance(blurred)

    def test_heavier_blur_reduces_it_further(self):
        sharp = _sharp_checkerboard()
        mild = _blurred(sharp, ksize=5)
        heavy = _blurred(sharp, ksize=21)
        assert laplacian_variance(mild) > laplacian_variance(heavy)

    def test_flat_image_has_near_zero_variance(self):
        assert laplacian_variance(_solid(128)) == pytest.approx(0.0, abs=1e-6)


class TestExposureStats:
    def test_all_black_image_reads_as_underexposed(self):
        """
        classical_features.py itself doesn't assign the "underexposed"
        label (that's the trained model's job) -- but an all-black image's
        raw stats should be an unambiguous underexposure signal: nearly
        every pixel clipped at black, near-zero mean luma.
        """
        stats = exposure_stats(_solid(0))
        assert stats["pct_clipped_low"] > 0.95
        assert stats["mean_luma"] < 5.0

    def test_all_white_image_reads_as_overexposed(self):
        stats = exposure_stats(_solid(255))
        assert stats["pct_clipped_high"] > 0.95
        assert stats["mean_luma"] > 250.0

    def test_mid_gray_image_has_no_clipping(self):
        stats = exposure_stats(_solid(128))
        assert stats["pct_clipped_low"] == pytest.approx(0.0)
        assert stats["pct_clipped_high"] == pytest.approx(0.0)
        assert stats["mean_luma"] == pytest.approx(128.0, abs=1.0)


class TestNoiseEstimate:
    def test_noisy_image_has_higher_noise_than_clean(self):
        clean = _solid(128)
        noisy = _noisy(clean, sigma=40.0)
        assert noise_estimate(noisy) > noise_estimate(clean)

    def test_flat_image_has_near_zero_noise(self):
        assert noise_estimate(_solid(128)) == pytest.approx(0.0, abs=1e-6)


class TestContrastStd:
    def test_high_contrast_beats_flat(self):
        half_and_half = np.zeros((128, 128, 3), dtype=np.uint8)
        half_and_half[:, 64:] = 255
        assert contrast_std(half_and_half) > contrast_std(_solid(128))


class TestJpegBlockiness:
    def test_synthetic_block_grid_scores_higher_than_smooth_gradient(self):
        size = 64
        blocky = np.zeros((size, size, 3), dtype=np.uint8)
        for by in range(0, size, 8):
            for bx in range(0, size, 8):
                shade = RNG.integers(0, 255)
                blocky[by : by + 8, bx : bx + 8] = shade

        ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
        smooth_gradient = np.stack([ramp] * 3, axis=-1)

        assert jpeg_blockiness(blocky) > jpeg_blockiness(smooth_gradient)


class TestExtractFeatureVector:
    def test_shape_and_order_match_feature_names(self):
        vector = extract_feature_vector(_sharp_checkerboard())
        assert vector.shape == (len(FEATURE_NAMES),)
        assert vector.dtype == np.float64

    def test_values_are_finite(self):
        vector = extract_feature_vector(_noisy(_sharp_checkerboard()))
        assert np.all(np.isfinite(vector))
