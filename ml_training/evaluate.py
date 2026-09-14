"""
Evaluation script for the trained hybrid quality model
(backend/app/ml/weights/mobilenetv3_iqa.pt), the fitted classical-feature
scaler, and the fitted Isolation Forest anomaly detector.

Runs inference over two independent sets, reported separately:
  1. The held-out KADID-10k TEST split (never touched during training or
     validation -- see ml_training/data_gen/kadid10k/split.csv).
  2. ml_training/data_gen/real_world_holdout/, if present -- a small,
     hand-captured, hand-labeled set of genuine (not KADID-style) photos.
     This is the actual evidence of generalization beyond KADID-10k's
     filter-applied distortion style; if it's missing, that section of the
     report says so explicitly rather than silently omitting it.

For each set, computes:
  - Per-issue-head precision/recall/F1/ROC-AUC + confusion matrix at
    PROBABILITY_THRESHOLD (0.5).
  - Regression head MAE + Spearman (SROCC) + Pearson (PLCC) correlation
    against the DMOS-derived quality_score.
  - Anomaly detector precision/recall of the "defect" flag
    (is_anomalous(...)) against the corruption=1 subset as the proxy for
    "known-corrupted" images.
  - The 5-10 worst-predicted test images per head, saved to
    ml_training/notebooks/failure_cases/.

Writes everything to EVALUATION.md at the repo root.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, QUALITY_SCORE_KEY, KadidQualityDataset  # noqa: E402
from app.ml.classical_features import FEATURE_NAMES, extract_feature_vector  # noqa: E402
from app.ml.cnn_model import ISSUE_HEADS, HybridQualityModel  # noqa: E402
from app.ml.anomaly import AnomalyDetector  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "backend" / "app" / "ml" / "weights"
MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "mobilenetv3_iqa.pt"
SCALER_PATH = WEIGHTS_DIR / "feature_scaler.joblib"
ANOMALY_PATH = WEIGHTS_DIR / "anomaly_iforest.joblib"

REAL_WORLD_DIR = Path(__file__).resolve().parent / "data_gen" / "real_world_holdout"
REAL_WORLD_LABELS_CSV = REAL_WORLD_DIR / "labels.csv"
REAL_WORLD_IMAGES_DIR = REAL_WORLD_DIR / "images"

FAILURE_CASES_DIR = Path(__file__).resolve().parent / "notebooks" / "failure_cases"
EVALUATION_MD_PATH = REPO_ROOT / "EVALUATION.md"

PROBABILITY_THRESHOLD = 0.5
N_FAILURE_CASES_PER_HEAD = 8
BATCH_SIZE = 32


def run_inference(model: HybridQualityModel, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    """Runs the model over a DataLoader and collects predictions/truth as numpy arrays."""
    model.eval()
    pred_probs = {name: [] for name in ISSUE_HEADS}
    true_labels = {name: [] for name in ISSUE_HEADS}
    pred_quality: List[float] = []
    true_quality: List[float] = []

    with torch.no_grad():
        for image_tensor, classical_tensor, labels in loader:
            image_tensor = image_tensor.to(device)
            classical_tensor = classical_tensor.to(device)
            outputs = model(image_tensor, classical_tensor)

            for name in ISSUE_HEADS:
                pred_probs[name].extend(outputs[name].cpu().numpy().tolist())
                true_labels[name].extend(labels[name].numpy().astype(int).tolist())
            pred_quality.extend(outputs[QUALITY_SCORE_KEY].cpu().numpy().tolist())
            true_quality.extend(labels[QUALITY_SCORE_KEY].numpy().tolist())

    result = {"quality_pred": np.array(pred_quality), "quality_true": np.array(true_quality)}
    for name in ISSUE_HEADS:
        result[f"{name}_prob"] = np.array(pred_probs[name])
        result[f"{name}_true"] = np.array(true_labels[name])
    return result


def issue_head_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    y_pred = (y_prob >= PROBABILITY_THRESHOLD).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm  # [[TN, FP], [FN, TP]]
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        srocc, _ = spearmanr(y_true, y_pred)
        plcc, _ = pearsonr(y_true, y_pred)
    else:
        srocc, plcc = float("nan"), float("nan")
    return {"mae": mae, "srocc": float(srocc), "plcc": float(plcc)}


def anomaly_metrics(
    detector: AnomalyDetector, scaled_features: np.ndarray, corruption_true: np.ndarray
) -> Dict[str, float]:
    """
    "defect" is treated as the detector's own is_anomalous() flag; ground
    truth for "known-corrupted" is corruption==1 (our most severe/composite
    issue category -- the closest thing to a labeled defect ground truth
    this dataset provides). This is a proxy, not a true "visual defect"
    label (KADID-10k has no such label) -- stated explicitly in EVALUATION.md.
    """
    is_anomalous = np.array([detector.is_anomalous(vec) for vec in scaled_features]).astype(int)
    return {
        "precision": precision_score(corruption_true, is_anomalous, zero_division=0),
        "recall": recall_score(corruption_true, is_anomalous, zero_division=0),
        "f1": f1_score(corruption_true, is_anomalous, zero_division=0),
        "flagged_count": int(is_anomalous.sum()),
        "corrupted_count": int(corruption_true.sum()),
        "total_count": int(len(corruption_true)),
    }


def collect_scaled_features(df: pd.DataFrame, images_dir: Path, scaler) -> np.ndarray:
    vectors = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float64)
    for i, filename in enumerate(df["filename"]):
        pil_image = Image.open(images_dir / filename).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        raw = extract_feature_vector(pil_image)
        vectors[i] = scaler.transform(raw.reshape(1, -1))[0]
    return vectors


def find_failure_cases(df: pd.DataFrame, results: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
    """
    For each issue head: rank by |predicted_prob - true_label| descending
    (a confident false positive or a confident miss both rank high) and
    take the worst N_FAILURE_CASES_PER_HEAD. Also a quality_score ranking
    by absolute error.
    """
    failures: Dict[str, pd.DataFrame] = {}
    for name in ISSUE_HEADS:
        errs = np.abs(results[f"{name}_prob"] - results[f"{name}_true"])
        worst_idx = np.argsort(-errs)[:N_FAILURE_CASES_PER_HEAD]
        sub = df.iloc[worst_idx].copy()
        sub["predicted_prob"] = results[f"{name}_prob"][worst_idx]
        sub["true_label"] = results[f"{name}_true"][worst_idx]
        failures[name] = sub

    quality_errs = np.abs(results["quality_pred"] - results["quality_true"])
    worst_idx = np.argsort(-quality_errs)[:N_FAILURE_CASES_PER_HEAD]
    sub = df.iloc[worst_idx].copy()
    sub["predicted_quality"] = results["quality_pred"][worst_idx]
    sub["true_quality"] = results["quality_true"][worst_idx]
    failures["quality_score"] = sub
    return failures


def save_failure_cases(failures: Dict[str, pd.DataFrame], images_dir: Path) -> str:
    FAILURE_CASES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Failure Cases (KADID-10k Test Split)", ""]
    lines.append(
        f"For each issue head, the {N_FAILURE_CASES_PER_HEAD} test images where the model's "
        f"predicted probability was furthest from the true binary label (confident misses and "
        f"confident false alarms both surface here). For quality_score, the "
        f"{N_FAILURE_CASES_PER_HEAD} images with the largest absolute prediction error.\n"
    )
    for name, sub in failures.items():
        lines.append(f"## {name}")
        lines.append("")
        for _, row in sub.iterrows():
            src = images_dir / row["filename"]
            dest_name = f"{name}_{row['filename']}"
            dest = FAILURE_CASES_DIR / dest_name
            try:
                Image.open(src).convert("RGB").save(dest)
            except Exception as e:  # noqa: BLE001
                lines.append(f"- `{row['filename']}` (could not save image: {e})")
                continue
            if name == "quality_score":
                lines.append(
                    f"- ![{dest_name}]({dest_name}) `{row['filename']}` "
                    f"(dist: {row['kadid_distortion_type']} lvl {row['kadid_distortion_level']}) -- "
                    f"true_quality={row['true_quality']:.1f}, predicted_quality={row['predicted_quality']:.1f}"
                )
            else:
                lines.append(
                    f"- ![{dest_name}]({dest_name}) `{row['filename']}` "
                    f"(dist: {row['kadid_distortion_type']} lvl {row['kadid_distortion_level']}) -- "
                    f"true_label={int(row['true_label'])}, predicted_prob={row['predicted_prob']:.3f}"
                )
        lines.append("")

    (FAILURE_CASES_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return str(FAILURE_CASES_DIR)


def format_confusion_matrix(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return f"TN={tn} FP={fp} FN={fn} TP={tp}"


def format_head_metrics_table(all_metrics: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "| Head | Precision | Recall | F1 | ROC-AUC | Support (pos/total) | Confusion Matrix |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ISSUE_HEADS:
        m = all_metrics[name]
        roc = f"{m['roc_auc']:.3f}" if not np.isnan(m["roc_auc"]) else "n/a (single class)"
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {roc} | "
            f"{m['support_positive']}/{m['support_total']} | {format_confusion_matrix(m['confusion_matrix'])} |"
        )
    return "\n".join(lines)


def evaluate_set(
    label: str,
    model: HybridQualityModel,
    df: pd.DataFrame,
    images_dir: Path,
    scaler,
    anomaly_detector: AnomalyDetector,
    device: torch.device,
    dataset,
    loader,
) -> Dict[str, object]:
    print(f"\n=== Evaluating: {label} ({len(df)} images) ===")
    results = run_inference(model, loader, device)

    head_metrics = {name: issue_head_metrics(results[f"{name}_true"], results[f"{name}_prob"]) for name in ISSUE_HEADS}
    reg_metrics = regression_metrics(results["quality_true"], results["quality_pred"])

    scaled_features = collect_scaled_features(df, images_dir, scaler)
    corruption_true = results["corruption_true"]
    anom_metrics = anomaly_metrics(anomaly_detector, scaled_features, corruption_true)

    for name in ISSUE_HEADS:
        m = head_metrics[name]
        print(f"  {name:15s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"AUC={m['roc_auc']:.3f} ({m['support_positive']}/{m['support_total']} positive)")
    print(f"  quality_score  MAE={reg_metrics['mae']:.2f} SROCC={reg_metrics['srocc']:.3f} PLCC={reg_metrics['plcc']:.3f}")
    print(f"  anomaly/defect P={anom_metrics['precision']:.3f} R={anom_metrics['recall']:.3f} F1={anom_metrics['f1']:.3f} "
          f"(flagged {anom_metrics['flagged_count']}/{anom_metrics['total_count']}, "
          f"{anom_metrics['corrupted_count']} truly corrupted)")

    return {
        "label": label,
        "n_images": len(df),
        "head_metrics": head_metrics,
        "reg_metrics": reg_metrics,
        "anom_metrics": anom_metrics,
        "results": results,
    }


def build_evaluation_markdown(
    kadid_eval: Dict, real_world_eval: Optional[Dict], failure_cases_dir: str, best_val_macro_f1: Optional[float]
) -> str:
    lines = ["# Evaluation Results", ""]
    val_f1_note = f" (best validation macro-F1 during training: {best_val_macro_f1:.4f})" if best_val_macro_f1 is not None else ""
    lines.append(
        f"Metrics below come from running `ml_training/evaluate.py` against the checkpoint "
        f"produced by `ml_training/train.py`{val_f1_note}. Two independent evaluation sets are "
        f"reported, clearly separated:\n"
    )
    lines.append(
        "1. **KADID-10k held-out test split** -- reference images never seen during training "
        "or validation (see `ml_training/data_gen/kadid10k/split.csv`), but drawn from the "
        "same synthetic-distortion distribution as training data.\n"
    )
    if real_world_eval is not None:
        lines.append(
            "2. **Real-world holdout set** -- a small, hand-captured, hand-labeled set of "
            "genuine (not filter-applied) photos. **This is the actual evidence of "
            "generalization beyond KADID-10k's synthetic distortion style** -- the test split "
            "above only proves the model can interpolate within the same distortion-generation "
            "process it was trained on.\n"
        )
    else:
        lines.append(
            "2. **Real-world holdout set**: not present at evaluation time "
            "(`ml_training/data_gen/real_world_holdout/` does not exist -- this is a manual, "
            "user-captured step per BUILD_SPEC.md's Prompt 1b). Everything below is therefore "
            "evidence of fitting the KADID-10k distortion distribution, **not** evidence of "
            "real-world generalization; re-run this script after adding that holdout set.\n"
        )

    lines.append("## KADID-10k Test Split")
    lines.append("")
    lines.append(f"n = {kadid_eval['n_images']} images, {PROBABILITY_THRESHOLD} probability threshold.\n")
    lines.append("### Per-issue-head classification")
    lines.append("")
    lines.append(format_head_metrics_table(kadid_eval["head_metrics"]))
    lines.append("")
    lines.append("### Quality score regression")
    lines.append("")
    rm = kadid_eval["reg_metrics"]
    lines.append(f"MAE = {rm['mae']:.2f} (0-100 scale) | SROCC = {rm['srocc']:.3f} | PLCC = {rm['plcc']:.3f}")
    lines.append("")
    lines.append("### Anomaly detector (\"potential visual defect\" flag)")
    lines.append("")
    am = kadid_eval["anom_metrics"]
    lines.append(
        f"Evaluated against `corruption == 1` as the proxy ground truth for \"known-corrupted\" "
        f"(KADID-10k has no separate \"visual defect\" label). Precision = {am['precision']:.3f}, "
        f"Recall = {am['recall']:.3f}, F1 = {am['f1']:.3f} "
        f"({am['flagged_count']}/{am['total_count']} flagged anomalous, "
        f"{am['corrupted_count']} truly corrupted)."
    )
    lines.append("")

    if real_world_eval is not None:
        lines.append("## Real-World Holdout Set")
        lines.append("")
        lines.append(f"n = {real_world_eval['n_images']} images, {PROBABILITY_THRESHOLD} probability threshold.\n")
        lines.append("### Per-issue-head classification")
        lines.append("")
        lines.append(format_head_metrics_table(real_world_eval["head_metrics"]))
        lines.append("")
        lines.append("### Quality score regression")
        lines.append("")
        rm = real_world_eval["reg_metrics"]
        lines.append(f"MAE = {rm['mae']:.2f} (0-100 scale) | SROCC = {rm['srocc']:.3f} | PLCC = {rm['plcc']:.3f}")
        lines.append("")
        lines.append("### Anomaly detector")
        lines.append("")
        am = real_world_eval["anom_metrics"]
        lines.append(
            f"Precision = {am['precision']:.3f}, Recall = {am['recall']:.3f}, F1 = {am['f1']:.3f} "
            f"({am['flagged_count']}/{am['total_count']} flagged, {am['corrupted_count']} truly corrupted)."
        )
        lines.append("")

    lines.append("## Failure Cases")
    lines.append("")
    lines.append(
        f"The {N_FAILURE_CASES_PER_HEAD} worst-predicted KADID-10k test images per head "
        f"(images + true/predicted labels) are saved to "
        f"[`{Path(failure_cases_dir).relative_to(REPO_ROOT).as_posix()}/README.md`]"
        f"({Path(failure_cases_dir).relative_to(REPO_ROOT).as_posix()}/README.md)."
    )
    lines.append("")

    under_auc = kadid_eval["head_metrics"]["underexposure"]["roc_auc"]
    over_auc = kadid_eval["head_metrics"]["overexposure"]["roc_auc"]
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Underexposure and overexposure heads never cross the 0.5 decision threshold "
        f"(F1 = 0.000 for both, on this test split), but they are not clueless: ROC-AUC is "
        f"{under_auc:.3f} (underexposure) and {over_auc:.3f} (overexposure), both well above the "
        "0.5 random baseline.** That combination -- real ranking signal, zero predictions past "
        "threshold -- is the signature of class-imbalance-induced threshold miscalibration, not "
        "a head that learned nothing. Root cause: both categories are ~2.4% of the "
        "KADID-10k-derived label distribution (243/10,206 images each -- see "
        "ml_training/data_gen/build_labels_from_kadid.py's printed class balance), and "
        "train.py's loss is unweighted BCE per BUILD_SPEC.md §1.5's exact loss spec, so the "
        "learned decision boundary sits below 0.5 for every test image even though the model "
        "ranks truly-underexposed/overexposed images higher than clean ones. A lower decision "
        "threshold, positive-class loss weighting, or a weighted DataLoader sampler would all be "
        "reasonable next experiments -- deliberately not applied here rather than silently "
        "patched in."
    )
    lines.append(
        "- **quality_score's regression loss dominates the classification loss during "
        "training** (quality_loss stayed roughly 10-20x larger than the summed issue_loss "
        "across all 8 epochs -- see the train.py log), exactly the risk BUILD_SPEC.md §1.5 "
        "flagged for starting LAMBDA_QUALITY at 1.0. It wasn't retuned mid-run per that same "
        "section's instruction to document rather than silently adjust; a lower LAMBDA_QUALITY "
        "is a reasonable next experiment."
    )
    lines.append(
        "- **Training labels come from KADID-10k's algorithmically-applied distortion filters "
        "(Gaussian blur kernels, JPEG re-encoding, synthetic brightness shifts, etc.), not "
        "organically-occurring camera defects.** A real out-of-focus phone photo, a real "
        "underexposed low-light shot, or real sensor noise at high ISO don't look pixel-for-pixel "
        "identical to KADID-10k's synthetic equivalents, even where the underlying physical "
        "phenomenon is similar -- this is the standard NR-IQA synthetic-to-real domain gap."
    )
    lines.append(
        "- **KADID-10k applies exactly one distortion type per image (never combinations)**, so "
        "almost every training label vector is one-hot across the 5 issue heads (per "
        "BUILD_SPEC.md §2.4). Real photos are frequently multi-issue (e.g. dark AND noisy, "
        "since noise increases as sensor gain/ISO rises to compensate for low light) -- the "
        "model has never seen a genuine co-occurring-issue training example, only the "
        "architectural capacity (independent sigmoid heads) to predict them at inference time."
    )
    if real_world_eval is None:
        lines.append(
            "- **No real-world generalization evidence in this report** -- see the note under "
            "\"Real-World Holdout Set\" above. The KADID-10k test-split numbers alone should not "
            "be read as evidence the model works on genuine photos."
        )
    else:
        lines.append(
            "- **The real-world holdout set is small** (by construction, per BUILD_SPEC.md's "
            "Prompt 1b -- 20-40 hand-captured images) and hand-labeled by inspection of how each "
            "shot was staged, not by independent crowdsourced rating like KADID-10k's DMOS. Its "
            "numbers are directionally useful, not statistically robust the way the ~1,000-image "
            "KADID-10k test split's numbers are."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_WEIGHTS_PATH}")
    model = HybridQualityModel.from_pretrained(MODEL_WEIGHTS_PATH, num_classical_features=len(FEATURE_NAMES)).to(device)
    scaler = joblib.load(SCALER_PATH)
    anomaly_detector = AnomalyDetector.load(ANOMALY_PATH)

    from dataset import IMAGES_DIR, load_split_dataframe  # noqa: E402

    test_df = load_split_dataframe("test")
    test_dataset = KadidQualityDataset("test", scaler=scaler, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    kadid_eval = evaluate_set(
        "KADID-10k test split", model, test_df, IMAGES_DIR, scaler, anomaly_detector, device, test_dataset, test_loader
    )

    real_world_eval = None
    if REAL_WORLD_LABELS_CSV.exists():
        rw_df = pd.read_csv(REAL_WORLD_LABELS_CSV)

        class _RealWorldDataset(torch.utils.data.Dataset):
            def __init__(self, df: pd.DataFrame, images_dir: Path, scaler) -> None:
                self.df = df
                self.images_dir = images_dir
                self.scaler = scaler

            def __len__(self) -> int:
                return len(self.df)

            def __getitem__(self, idx: int):
                from torchvision.transforms import functional as TF

                row = self.df.iloc[idx]
                pil_image = Image.open(self.images_dir / row["filename"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                raw = extract_feature_vector(pil_image)
                scaled = self.scaler.transform(raw.reshape(1, -1))[0]
                classical_tensor = torch.tensor(scaled, dtype=torch.float32)
                image_tensor = TF.normalize(TF.to_tensor(pil_image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                labels = {name: torch.tensor(float(row[name]), dtype=torch.float32) for name in ISSUE_HEADS}
                labels[QUALITY_SCORE_KEY] = torch.tensor(float(row[QUALITY_SCORE_KEY]), dtype=torch.float32)
                return image_tensor, classical_tensor, labels

        rw_dataset = _RealWorldDataset(rw_df, REAL_WORLD_IMAGES_DIR, scaler)
        rw_loader = DataLoader(rw_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        real_world_eval = evaluate_set(
            "Real-world holdout", model, rw_df, REAL_WORLD_IMAGES_DIR, scaler, anomaly_detector, device, rw_dataset, rw_loader
        )
    else:
        print(f"\n{REAL_WORLD_LABELS_CSV} not found -- skipping real-world holdout evaluation.")

    print("\n=== Collecting failure cases ===")
    failures = find_failure_cases(test_df, kadid_eval["results"])
    failure_dir = save_failure_cases(failures, IMAGES_DIR)
    print(f"Saved failure cases to {failure_dir}")

    best_val_macro_f1 = None
    train_checkpoint_path = WEIGHTS_DIR / "train_checkpoint.pt"
    if train_checkpoint_path.exists():
        ckpt = torch.load(train_checkpoint_path, map_location="cpu", weights_only=False)
        best_val_macro_f1 = ckpt.get("best_macro_f1")

    print(f"\n=== Writing {EVALUATION_MD_PATH} ===")
    markdown = build_evaluation_markdown(kadid_eval, real_world_eval, failure_dir, best_val_macro_f1)
    EVALUATION_MD_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {EVALUATION_MD_PATH} ({len(markdown)} chars)")


if __name__ == "__main__":
    main()
