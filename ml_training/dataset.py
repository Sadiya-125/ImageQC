"""
PyTorch Dataset for the KADID-10k-derived quality-assessment training data
(ml_training/data_gen/labels.csv + ml_training/data_gen/kadid10k/split.csv,
both built by data_gen/build_labels_from_kadid.py -- see that file for the
label derivation and the reference-image-level train/val/test split).

Each sample returns:
  - image_tensor: (3, 224, 224) float tensor, ImageNet-normalized. The
    train split additionally gets a random horizontal flip -- the only
    augmentation used, deliberately, since anything that alters sharpness,
    exposure, noise, or color would corrupt this dataset's own labels
    (those ARE the things being predicted).
  - classical_features_tensor: (len(FEATURE_NAMES),) float tensor, computed
    from the SAME (possibly flipped) image via
    backend/app/ml/classical_features.extract_feature_vector, then
    standardized with a StandardScaler fit on the training split only (see
    fit_feature_scaler() below) -- never fit on val/test or the full
    dataset, per BUILD_SPEC.md's Prompt 3 instructions.
  - labels: dict of 5 binary issue tensors (blur, underexposure,
    overexposure, noise, corruption) + "quality_score".

IMPORT NOTE: see backend/app/ml/__init__.py for why this imports
backend/app/ml/ via sys.path insertion rather than an editable install.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

_BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.classical_features import FEATURE_NAMES, extract_feature_vector  # noqa: E402
from app.ml.cnn_model import ISSUE_HEADS  # noqa: E402

DATA_GEN_DIR = Path(__file__).resolve().parent / "data_gen"
LABELS_CSV_PATH = DATA_GEN_DIR / "labels.csv"
SPLIT_CSV_PATH = DATA_GEN_DIR / "kadid10k" / "split.csv"
IMAGES_DIR = DATA_GEN_DIR / "kadid10k" / "images"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

QUALITY_SCORE_KEY = "quality_score"
SPLITS = ("train", "val", "test")


def load_split_dataframe(split: str) -> pd.DataFrame:
    assert split in SPLITS, f"split must be one of {SPLITS}, got {split!r}"
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    split_df = pd.read_csv(SPLIT_CSV_PATH)
    merged = labels_df.merge(split_df, on="reference_id", how="inner")
    return merged[merged["split"] == split].reset_index(drop=True)


class KadidQualityDataset(Dataset):
    def __init__(
        self,
        split: str,
        scaler: Optional[StandardScaler] = None,
        augment: Optional[bool] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split = split
        self.df = load_split_dataframe(split)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)
        self.scaler = scaler
        # Default: augment the train split only, but allow an explicit
        # override (e.g. augment=False for a val-style pass over train data).
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        pil_image = Image.open(IMAGES_DIR / row["filename"]).convert("RGB")
        pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.augment and torch.rand(1).item() < 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Classical features are computed from this exact (possibly
        # flipped) image, so the CNN branch and the classical branch always
        # describe the same visual content.
        raw_features = extract_feature_vector(pil_image)
        if self.scaler is not None:
            features = self.scaler.transform(raw_features.reshape(1, -1))[0]
        else:
            features = raw_features
        classical_tensor = torch.tensor(features, dtype=torch.float32)

        image_tensor = TF.to_tensor(pil_image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        labels: Dict[str, torch.Tensor] = {
            name: torch.tensor(float(row[name]), dtype=torch.float32) for name in ISSUE_HEADS
        }
        labels[QUALITY_SCORE_KEY] = torch.tensor(float(row[QUALITY_SCORE_KEY]), dtype=torch.float32)

        return image_tensor, classical_tensor, labels


def fit_feature_scaler(max_samples: Optional[int] = None) -> StandardScaler:
    """
    Fits a StandardScaler on classical feature vectors from the TRAIN split
    only. Computed on canonical (resized, non-flipped) images for a
    deterministic, reproducible fit: our only augmentation is a horizontal
    flip, which leaves every one of our 8 features materially unchanged on
    average (all 8 are computed from luma/gradient statistics that are
    symmetric under a left-right mirror), so fitting on the canonical image
    is a safe, reproducible proxy for "whatever the model will see."

    max_samples: optional cap, for quick iteration/debugging only -- the
    real training run should leave this as None (fits on the full ~8.2k
    train-split images).
    """
    df = load_split_dataframe("train")
    if max_samples is not None:
        df = df.iloc[:max_samples]

    vectors = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float64)
    for i, filename in enumerate(df["filename"]):
        pil_image = Image.open(IMAGES_DIR / filename).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        vectors[i] = extract_feature_vector(pil_image)

    scaler = StandardScaler()
    scaler.fit(vectors)
    return scaler
