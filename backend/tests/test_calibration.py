"""Unit tests for app/ml/calibration.py's temperature-scaling math."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.calibration import apply_temperature  # noqa: E402


class TestApplyTemperature:
    def test_identity_temperature_is_a_noop(self):
        assert apply_temperature(0.73, 1.0) == pytest.approx(0.73, abs=1e-6)

    def test_softens_toward_half_when_temperature_above_one(self):
        original = 0.9
        softened = apply_temperature(original, 2.0)
        assert 0.5 < softened < original

    def test_sharpens_away_from_half_when_temperature_below_one(self):
        original = 0.9
        sharpened = apply_temperature(original, 0.5)
        assert sharpened > original

    def test_preserves_which_side_of_half_the_probability_is_on(self):
        for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
            for t in [0.2, 0.5, 1.0, 2.0, 5.0]:
                calibrated = apply_temperature(p, t)
                assert (calibrated >= 0.5) == (p >= 0.5), f"p={p}, t={t}"

    def test_stays_in_valid_probability_range(self):
        for p in [0.0001, 0.5, 0.9999]:
            for t in [0.1, 1.0, 10.0]:
                calibrated = apply_temperature(p, t)
                assert 0.0 <= calibrated <= 1.0
