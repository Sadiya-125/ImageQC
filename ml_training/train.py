"""
Training script for the hybrid MobileNetV3-Small + classical-features
quality model (backend/app/ml/cnn_model.py) and the Isolation Forest
anomaly detector (backend/app/ml/anomaly.py), on the KADID-10k-derived
labels from ml_training/data_gen/.

HYPERPARAMETERS (documented per BUILD_SPEC.md's Prompt 3 requirements):

  Two-phase training:
    Phase 1 (PHASE1_EPOCHS=3): MobileNetV3-Small backbone frozen entirely.
      Only the classical branch + fusion trunk + 6 heads train, at
      lr=PHASE1_LR=1e-3, so these newly-initialized layers stabilize before
      any gradient touches the pretrained backbone.
    Phase 2 (PHASE2_EPOCHS=5): the last UNFREEZE_LAST_N_BLOCKS=3 modules of
      the backbone's `features` (its final two InvertedResidual blocks plus
      the closing 1x1 Conv2dNormActivation -- see cnn_model.py's
      get_gradcam_target_layer docstring for why these are the last
      spatially-meaningful layers) are unfrozen and fine-tuned at
      lr=PHASE2_BACKBONE_LR=1e-5, while the rest of the network continues
      training at lr=PHASE2_HEAD_LR=1e-4.
    8 total epochs by default. This is deliberately modest for a 48-hour
    assessment window with an ~8.2k-image train split on a 4GB GPU: it's
    enough to demonstrate the two-phase fine-tuning strategy working
    end-to-end and to produce a real, checkpointed, evaluable model, while
    leaving the rest of the 48 hours for the backend/frontend/deployment
    work the assessment also grades. Override via --phase1-epochs /
    --phase2-epochs for a longer run.

  Optimizer/schedule: AdamW, weight_decay=WEIGHT_DECAY=1e-4, CosineAnnealingLR
    within each phase (T_max = that phase's epoch count).

  Loss (exact, per BUILD_SPEC.md §1.5):
    total_loss = sum(BCELoss(pred_i, true_i) for i in the 5 issue heads)
               + LAMBDA_QUALITY * SmoothL1Loss(quality_pred, quality_true)
    LAMBDA_QUALITY=1.0 to start (the 5 BCE terms are SUMMED, not averaged,
    so the combined classification signal is roughly comparable in scale to
    the regression term -- summing rather than averaging was the explicit
    instruction). Both loss components are logged separately every epoch
    specifically so it's visible if one dominates the other and LAMBDA needs
    revisiting later -- deliberately not silently retuned here.

  Batch size: BATCH_SIZE=32 (within the specified 16-32 range), mixed
    precision via torch.cuda.amp (autocast + GradScaler) to fit comfortably
    in the RTX 3050's 4GB VRAM.

  Checkpointing: the best-so-far model (by val macro-F1 across the 5 issue
    heads) is saved as a plain state_dict to MODEL_WEIGHTS_PATH (loadable
    directly via HybridQualityModel.from_pretrained). A separate, fuller
    TRAIN_CHECKPOINT_PATH is saved after every epoch (model + optimizer +
    global epoch + best-so-far metric) to support --resume. Resuming
    rebuilds the current phase's optimizer/scheduler from scratch rather
    than restoring their internal state -- a deliberate simplification
    (AdamW's momentum re-warms quickly; exact resume-time LR continuity
    wasn't worth the added bookkeeping for a 48-hour assessment).

CUDA fallback: if torch.cuda.is_available() is False, this script prints a
clear warning, switches to CPU, and automatically shrinks both the epoch
counts and the dataset (via --max-train-samples/--max-val-samples) so a
real, complete, if smaller, run still finishes -- rather than attempting the
full 8-epoch/8.2k-image run on CPU, which would not complete in a reasonable
time.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dataset import (  # noqa: E402
    IMAGES_DIR,
    IMAGE_SIZE,
    QUALITY_SCORE_KEY,
    KadidQualityDataset,
    fit_feature_scaler,
    load_split_dataframe,
)
from app.ml.classical_features import FEATURE_NAMES, extract_feature_vector  # noqa: E402
from app.ml.cnn_model import ISSUE_HEADS, HybridQualityModel  # noqa: E402
from app.ml.anomaly import AnomalyDetector  # noqa: E402

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "backend" / "app" / "ml" / "weights"
MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "mobilenetv3_iqa.pt"
SCALER_PATH = WEIGHTS_DIR / "feature_scaler.joblib"
ANOMALY_PATH = WEIGHTS_DIR / "anomaly_iforest.joblib"
TRAIN_CHECKPOINT_PATH = WEIGHTS_DIR / "train_checkpoint.pt"

PHASE1_LR = 1e-3
PHASE2_BACKBONE_LR = 1e-5
PHASE2_HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
UNFREEZE_LAST_N_BLOCKS = 3
LAMBDA_QUALITY = 1.0

BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for Windows-safe reliability; bump up on Linux for speed.


def print_device_info(device: torch.device) -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.2f} GB VRAM)")
    else:
        print("No CUDA GPU detected -- using CPU.")
    print(f"Device: {device}\n")


def set_phase1_trainable(model: HybridQualityModel) -> None:
    for p in model.cnn_features.parameters():
        p.requires_grad = False
    for p in model.cnn_pool.parameters():
        p.requires_grad = False


def set_phase2_trainable(model: HybridQualityModel, n_blocks: int = UNFREEZE_LAST_N_BLOCKS) -> None:
    for p in model.cnn_features.parameters():
        p.requires_grad = False
    for block in list(model.cnn_features.children())[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def build_optimizer(model: HybridQualityModel, phase: int) -> torch.optim.Optimizer:
    if phase == 1:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return AdamW(trainable, lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    backbone_params = [p for p in model.cnn_features.parameters() if p.requires_grad]
    other_params = (
        list(model.classical_branch.parameters())
        + list(model.trunk.parameters())
        + list(model.issue_heads.parameters())
        + list(model.quality_head.parameters())
    )
    return AdamW(
        [
            {"params": backbone_params, "lr": PHASE2_BACKBONE_LR},
            {"params": other_params, "lr": PHASE2_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def compute_loss(
    outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], bce: nn.Module, huber: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    issue_loss = sum(bce(outputs[name], labels[name]) for name in ISSUE_HEADS)
    quality_loss = huber(outputs[QUALITY_SCORE_KEY], labels[QUALITY_SCORE_KEY])
    total = issue_loss + LAMBDA_QUALITY * quality_loss
    return total, issue_loss, quality_loss


def run_epoch(
    model: HybridQualityModel,
    loader: DataLoader,
    device: torch.device,
    bce: nn.Module,
    huber: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_issue_loss = total_quality_loss = 0.0
    n_batches = 0
    all_true = {name: [] for name in ISSUE_HEADS}
    all_pred = {name: [] for name in ISSUE_HEADS}

    with torch.set_grad_enabled(is_train):
        for image_tensor, classical_tensor, labels in loader:
            image_tensor = image_tensor.to(device, non_blocking=True)
            classical_tensor = classical_tensor.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(image_tensor, classical_tensor)

            # BCELoss (fed by our model's internal sigmoid) is not safe to
            # run under autocast in fp16 -- PyTorch raises on this
            # combination. The loss itself is cheap (operates on (N,)
            # tensors), so it's computed in fp32 outside the autocast
            # block; the expensive CNN forward pass above still benefits
            # from mixed precision.
            outputs_fp32 = {k: v.float() for k, v in outputs.items()}
            loss, issue_loss, quality_loss = compute_loss(outputs_fp32, labels, bce, huber)

            if is_train:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

            total_loss += loss.item()
            total_issue_loss += issue_loss.item()
            total_quality_loss += quality_loss.item()
            n_batches += 1

            for name in ISSUE_HEADS:
                all_true[name].extend(labels[name].detach().cpu().numpy().astype(int).tolist())
                all_pred[name].extend((outputs[name].detach().float().cpu().numpy() >= 0.5).astype(int).tolist())

    f1_per_head = {name: f1_score(all_true[name], all_pred[name], zero_division=0) for name in ISSUE_HEADS}
    macro_f1 = float(np.mean(list(f1_per_head.values())))

    return {
        "loss": total_loss / n_batches,
        "issue_loss": total_issue_loss / n_batches,
        "quality_loss": total_quality_loss / n_batches,
        "f1_per_head": f1_per_head,
        "macro_f1": macro_f1,
    }


def fit_and_save_anomaly_detector(train_df, scaler: StandardScaler) -> None:
    clean_mask = train_df[ISSUE_HEADS].sum(axis=1) == 0
    clean_df = train_df[clean_mask]
    print(f"\n{len(clean_df)} / {len(train_df)} training images are clean-labeled (all 5 issue flags = 0)")

    vectors = np.zeros((len(clean_df), len(FEATURE_NAMES)), dtype=np.float64)
    for i, filename in enumerate(clean_df["filename"]):
        pil_image = Image.open(IMAGES_DIR / filename).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        raw = extract_feature_vector(pil_image)
        vectors[i] = scaler.transform(raw.reshape(1, -1))[0]

    detector = AnomalyDetector().fit(vectors)
    detector.save(ANOMALY_PATH)
    print(f"Fitted Isolation Forest on {len(vectors)} clean samples, saved to {ANOMALY_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume from TRAIN_CHECKPOINT_PATH if present.")
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--force-full-on-cpu",
        action="store_true",
        help="Skip the automatic epoch/dataset shrinking that normally kicks in when no CUDA GPU is found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    phase_epochs = {1: args.phase1_epochs, 2: args.phase2_epochs}
    max_train_samples, max_val_samples = args.max_train_samples, args.max_val_samples
    if device.type == "cpu" and not args.force_full_on_cpu:
        print(
            "No CUDA GPU available in this environment -- automatically reducing to a "
            "smaller, still-real training run (1 epoch/phase, 200 train / 50 val images) "
            "so it completes here. Pass --force-full-on-cpu to override, or run this same "
            "script on a CUDA machine for the full configuration.\n"
        )
        phase_epochs = {1: 1, 2: 1}
        max_train_samples = max_train_samples or 200
        max_val_samples = max_val_samples or 50
    total_epochs = phase_epochs[1] + phase_epochs[2]

    # --- Feature scaler: fit on TRAIN split only, reused across resumes ---
    if SCALER_PATH.exists():
        print(f"Loading existing feature scaler from {SCALER_PATH}")
        feature_scaler: StandardScaler = joblib.load(SCALER_PATH)
    else:
        print("Fitting StandardScaler on the training split's classical features...")
        t0 = time.time()
        feature_scaler = fit_feature_scaler(max_samples=max_train_samples)
        joblib.dump(feature_scaler, SCALER_PATH)
        print(f"Fitted and saved feature scaler to {SCALER_PATH} ({time.time() - t0:.1f}s)")
        print(f"  mean: {np.round(feature_scaler.mean_, 4)}")
        print(f"  std:  {np.round(np.sqrt(feature_scaler.var_), 4)}\n")

    train_dataset = KadidQualityDataset("train", scaler=feature_scaler, max_samples=max_train_samples)
    val_dataset = KadidQualityDataset("val", scaler=feature_scaler, max_samples=max_val_samples)
    print(f"train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = HybridQualityModel(num_classical_features=len(FEATURE_NAMES), pretrained=True).to(device)
    bce = nn.BCELoss()
    huber = nn.SmoothL1Loss()

    start_global_epoch = 0
    best_macro_f1 = -1.0
    if args.resume and TRAIN_CHECKPOINT_PATH.exists():
        ckpt = torch.load(TRAIN_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_global_epoch = ckpt["global_epoch"] + 1
        best_macro_f1 = ckpt["best_macro_f1"]
        print(f"Resumed from {TRAIN_CHECKPOINT_PATH}: global epoch {ckpt['global_epoch']}, best macro-F1 {best_macro_f1:.4f}")

    def phase_for_global_epoch(global_epoch: int) -> int:
        return 1 if global_epoch < phase_epochs[1] else 2

    current_phase = None
    optimizer = None
    scheduler = None
    amp_scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    for global_epoch in range(start_global_epoch, total_epochs):
        phase = phase_for_global_epoch(global_epoch)
        if phase != current_phase:
            current_phase = phase
            if phase == 1:
                set_phase1_trainable(model)
            else:
                set_phase2_trainable(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = build_optimizer(model, phase)
            scheduler = CosineAnnealingLR(optimizer, T_max=phase_epochs[phase])
            print(f"\n=== Phase {phase} start (global epoch {global_epoch}) -- {n_trainable:,} trainable params ===")

        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, bce, huber, optimizer=optimizer, amp_scaler=amp_scaler)
        scheduler.step()
        val_metrics = run_epoch(model, val_loader, device, bce, huber)
        elapsed = time.time() - t0

        print(
            f"[phase {phase}] epoch {global_epoch + 1}/{total_epochs} ({elapsed:.1f}s) "
            f"train_loss={train_metrics['loss']:.4f} (issue={train_metrics['issue_loss']:.4f} "
            f"quality={train_metrics['quality_loss']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} (issue={val_metrics['issue_loss']:.4f} "
            f"quality={val_metrics['quality_loss']:.4f}) | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        print(f"  val F1 per head: " + ", ".join(f"{k}={v:.3f}" for k, v in val_metrics["f1_per_head"].items()))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"  -> new best val macro-F1 {best_macro_f1:.4f}, saved to {MODEL_WEIGHTS_PATH}")

        torch.save(
            {"model_state_dict": model.state_dict(), "global_epoch": global_epoch, "best_macro_f1": best_macro_f1},
            TRAIN_CHECKPOINT_PATH,
        )

    print(f"\nTraining complete. Best val macro-F1: {best_macro_f1:.4f}")

    train_df = load_split_dataframe("train")
    if max_train_samples is not None:
        train_df = train_df.iloc[:max_train_samples]
    fit_and_save_anomaly_detector(train_df, feature_scaler)


if __name__ == "__main__":
    main()
