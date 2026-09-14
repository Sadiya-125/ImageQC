"""
Post-hoc confidence calibration for the 5 issue heads, via temperature
scaling (Guo et al., "On Calibration of Modern Neural Networks," ICML 2017).

HybridQualityModel.forward() (cnn_model.py) applies sigmoid internally and
returns probabilities directly, not raw logits -- that's the right call for
the model's own API (every consumer wants a probability), but temperature
scaling is normally applied to logits. Rather than changing the model's
output contract, calibration works here by recovering an approximate logit
from the output probability (`logit = ln(p / (1-p))`), rescaling that by a
per-head temperature T fit on the validation split (ml_training/calibrate.py),
and re-applying sigmoid: `p_calibrated = sigmoid(logit(p_raw) / T)`.
T > 1 softens overconfident probabilities toward 0.5; T < 1 sharpens them;
T == 1 is a no-op. This is an approximation (rounding at the float32
boundary near p=0 or p=1 loses some precision converting back to a logit),
but negligible in practice since well-calibrated confidence values rarely
sit at the extremes anyway.
"""

import json
import math
from pathlib import Path
from typing import Dict

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
CALIBRATION_PATH = WEIGHTS_DIR / "calibration.json"

# Numerical safety margin so ln(p / (1-p)) never hits +/-inf.
_PROB_EPSILON = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _PROB_EPSILON), 1.0 - _PROB_EPSILON)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def apply_temperature(p: float, temperature: float) -> float:
    if temperature <= 0 or temperature == 1.0:
        return p
    return _sigmoid(_logit(p) / temperature)


def read_temperatures(path: Path = CALIBRATION_PATH) -> Dict[str, float]:
    """Returns {head_name: temperature}, or {} if no calibration has been fit yet (all heads default to T=1, a no-op)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_temperatures(temperatures: Dict[str, float], path: Path = CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(temperatures, indent=2), encoding="utf-8")
