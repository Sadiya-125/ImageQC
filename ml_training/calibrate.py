"""
Fits per-head temperature-scaling parameters (see backend/app/ml/calibration.py
for the method) on the KADID-10k validation split, using the best checkpoint
from ml_training/train.py. Run this after training, before evaluate.py, so
evaluate.py's reported metrics reflect the calibrated confidences the API
actually serves.

Usage: python ml_training/calibrate.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dataset import KadidQualityDataset  # noqa: E402
from app.ml.calibration import CALIBRATION_PATH, write_temperatures  # noqa: E402
from app.ml.classical_features import FEATURE_NAMES  # noqa: E402
from app.ml.cnn_model import ISSUE_HEADS, HybridQualityModel  # noqa: E402

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "backend" / "app" / "ml" / "weights"
MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "mobilenetv3_iqa.pt"
SCALER_PATH = WEIGHTS_DIR / "feature_scaler.joblib"

BATCH_SIZE = 32
_PROB_EPSILON = 1e-6


def _collect_val_predictions(model: HybridQualityModel, device: torch.device):
    scaler = joblib.load(SCALER_PATH)
    val_dataset = KadidQualityDataset("val", scaler=scaler, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probs = {name: [] for name in ISSUE_HEADS}
    trues = {name: [] for name in ISSUE_HEADS}

    model.eval()
    with torch.no_grad():
        for image_tensor, classical_tensor, labels in val_loader:
            outputs = model(image_tensor.to(device), classical_tensor.to(device))
            for name in ISSUE_HEADS:
                probs[name].extend(outputs[name].cpu().numpy().tolist())
                trues[name].extend(labels[name].numpy().astype(int).tolist())

    return probs, trues


def _nll_for_temperature(temperature: float, probs: np.ndarray, trues: np.ndarray) -> float:
    """Negative log-likelihood of the true labels under sigmoid(logit(p)/T)."""
    p = np.clip(probs, _PROB_EPSILON, 1.0 - _PROB_EPSILON)
    logits = np.log(p / (1.0 - p))
    scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
    scaled = np.clip(scaled, _PROB_EPSILON, 1.0 - _PROB_EPSILON)
    return float(-np.mean(trues * np.log(scaled) + (1 - trues) * np.log(1 - scaled)))


def fit_temperature(probs: list, trues: list) -> float:
    probs_arr = np.array(probs)
    trues_arr = np.array(trues, dtype=np.float64)
    if len(np.unique(trues_arr)) < 2:
        # No positive (or no negative) examples for this head in the val
        # split -- NLL-based fitting is meaningless, leave uncalibrated.
        return 1.0
    result = minimize_scalar(
        lambda t: _nll_for_temperature(t, probs_arr, trues_arr),
        bounds=(0.05, 10.0),
        method="bounded",
    )
    return float(result.x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {MODEL_WEIGHTS_PATH}")
    model = HybridQualityModel.from_pretrained(MODEL_WEIGHTS_PATH, num_classical_features=len(FEATURE_NAMES)).to(device)

    print("Collecting validation-split predictions...")
    probs, trues = _collect_val_predictions(model, device)

    temperatures = {}
    print("\nFitted temperatures (T=1.0 means no change; T>1 softens, T<1 sharpens):")
    for name in ISSUE_HEADS:
        t = fit_temperature(probs[name], trues[name])
        temperatures[name] = round(t, 4)
        n_pos = int(sum(trues[name]))
        print(f"  {name:15s} T={t:.4f}  ({n_pos}/{len(trues[name])} positive in val)")

    write_temperatures(temperatures)
    print(f"\nWrote calibration temperatures to {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
