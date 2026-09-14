"""
Builds ml_training/data_gen/labels.csv from the real KADID-10k metadata
extracted by download_kadid10k.py (ml_training/data_gen/kadid10k/dmos.csv +
ml_training/data_gen/kadid10k/images/), and a reference-image-level
train/val/test split written to ml_training/data_gen/kadid10k/split.csv.

--------------------------------------------------------------------------
VERIFIED AGAINST THE ACTUAL EXTRACTED ARCHIVE (not assumed) -- see the
project conversation log for the full derivation. Two things in the
original spec turned out to need correction once real data was inspected:

1. Metadata schema. dmos.csv ships only 4 columns: dist_img, ref_img, dmos,
   var (NOT separate distortion-type / distortion-level columns). The
   distortion type (01-25) and severity level (1-5) must be parsed out of
   the dist_img filename itself, which follows the pattern
   "I<ref>_<type>_<level>.png" (e.g. "I01_01_01.png" = reference image 1,
   distortion type 1, level 1). Reference images themselves are not present
   in dmos.csv at all (they were never subjectively rated) -- they appear
   only as bare "I<ref>.png" files in images/.

2. DMOS direction. The spec assumed "DMOS 1-5, higher = worse". The real
   data shows the opposite for this benchmark: DMOS is MOS-like here
   (higher = better). Confirmed two ways against the actual downloaded
   dmos.csv:
     - Mean DMOS by severity level, across all 10,125 distorted images:
         level 1: 4.08   level 2: 3.52   level 3: 3.06
         level 4: 2.50   level 5: 2.01
       i.e. DMOS falls monotonically as distortion severity rises.
     - The official KADID-10k database page states this explicitly: "high
       value of DMOS corresponds to higher visual quality of image."
   QUALITY_SCORE_FORMULA below is therefore the direction-corrected linear
   rescale of the real DMOS scale (still 0-100, still higher = better,
   same shape as originally intended -- only the sign of dmos in the
   formula changes to match the verified real direction).
--------------------------------------------------------------------------

Usage: python ml_training/data_gen/build_labels_from_kadid.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

DATA_GEN_DIR = Path(__file__).resolve().parent
KADID_DIR = DATA_GEN_DIR / "kadid10k"
DMOS_CSV_PATH = KADID_DIR / "dmos.csv"
IMAGES_DIR = KADID_DIR / "images"

LABELS_CSV_PATH = DATA_GEN_DIR / "labels.csv"
SPLIT_CSV_PATH = KADID_DIR / "split.csv"

DISTORTED_FILENAME_RE = re.compile(r"^I(\d+)_(\d+)_(\d+)\.png$")
REFERENCE_FILENAME_RE = re.compile(r"^I(\d+)\.png$")

# All 25 KADID-10k distortion types, numbered exactly as used in filenames.
# Source: official KADID-10k database description
# (https://database.mmsp-kn.de/kadid-10k-database.html), cross-checked
# against the QoMEX 2019 paper (Lin, Hosu, Saupe).
DISTORTION_TYPE_NAMES = {
    1: "Gaussian blur",
    2: "Lens blur",
    3: "Motion blur",
    4: "Color diffusion",
    5: "Color shift",
    6: "Color quantization",
    7: "Color saturation 1",
    8: "Color saturation 2",
    9: "JPEG2000",
    10: "JPEG",
    11: "White noise",
    12: "White noise in color component",
    13: "Impulse noise",
    14: "Multiplicative noise",
    15: "Denoise",
    16: "Brighten",
    17: "Darken",
    18: "Mean shift",
    19: "Jitter",
    20: "Non-eccentricity patch",
    21: "Pixelate",
    22: "Quantization",
    23: "Color block",
    24: "High sharpen",
    25: "Contrast change",
}

# Our 6 required categories, from BUILD_SPEC.md's starting mapping, VERIFIED
# against the real archive (see docstring above and the per-type notes
# below). Deviations from the original starting mapping are called out
# explicitly.
BLUR = "blur"
UNDEREXPOSURE = "underexposure"
OVEREXPOSURE = "overexposure"
NOISE = "noise"
CORRUPTION = "corruption"

DISTORTION_CATEGORY_MAP = {
    # --- Blur: matches the starting mapping exactly. Verified via
    # Laplacian variance (sharpness) monotonically DECREASING with level
    # for all three types on real images. ---
    1: BLUR,  # Gaussian blur
    2: BLUR,  # Lens blur
    3: BLUR,  # Motion blur
    # --- Exposure: Brighten/Darken verified via mean luma monotonically
    # increasing/decreasing with level on real images.
    #
    # DEVIATION FROM STARTING MAPPING: "Mean shift" (type 18) is
    # deliberately EXCLUDED from both exposure categories. The spec's
    # starting mapping proposed folding its "low end" into underexposure
    # and its "high end" into overexposure. Measuring real mean luma across
    # its 5 levels on a reference image showed the opposite of a clean
    # split, and a shape incompatible with our severity-threshold rule:
    #   level 1: +19.2 luma (brighter)   level 2: +9.5 (brighter)
    #   level 3:  +0.0 luma (UNCHANGED)  level 4: -12.0 (darker)
    #   level 5: -22.9 luma (darker)
    # i.e. level 1-2 are the OVER-exposure-directed end (not "low end ->
    # underexposure" as the spec guessed) and level 4-5 are the
    # UNDER-exposure-directed end (not "high end -> overexposure"). Level 3
    # is essentially a no-op (no shift at all). Since our SEVERITY_THRESHOLD
    # rule below flags levels 3-5 as positive for whatever category a type
    # maps to, mapping Mean shift to either single category would
    # mislabel its unshifted level-3 images as a detected issue, and mixing
    # both categories under one severity-based binary rule isn't
    # expressible without a type-specific carve-out that would break the
    # otherwise-uniform, citable threshold rule. Excluded rather than
    # guessed.
    16: OVEREXPOSURE,  # Brighten
    17: UNDEREXPOSURE,  # Darken
    # --- Noise: matches the starting mapping's intent. Verified via
    # high-frequency residual std monotonically increasing with level for
    # White/Impulse/Multiplicative noise on real images. "Denoise" (an
    # over-smoothing/denoising-artifact type per KADID-10k's own
    # description) is included per the spec's explicit instruction to
    # include "any denoising-artifact type present". ---
    11: NOISE,  # White noise (spec's "Gaussian noise")
    12: NOISE,  # White noise in color component (spec's "color-component noise")
    13: NOISE,  # Impulse noise
    14: NOISE,  # Multiplicative noise
    15: NOISE,  # Denoise (denoising-artifact type)
    # --- Corruption: JPEG/JPEG2000 compression artifacts, plus color
    # quantization / color-block distortion per the spec's explicit
    # mention. (The unconditional "severity level 5 = corruption" rule is
    # applied separately below, regardless of type.) ---
    9: CORRUPTION,  # JPEG2000
    10: CORRUPTION,  # JPEG
    6: CORRUPTION,  # Color quantization
    23: CORRUPTION,  # Color block
}

ISSUE_LABELS = [BLUR, UNDEREXPOSURE, OVEREXPOSURE, NOISE, CORRUPTION]

# Only levels >= this threshold set a mapped category's binary label to 1;
# levels below it leave the label at 0. This is a real modeling decision
# (affects class balance) -- named here so it's citable in the writeup,
# not a magic number buried in logic.
SEVERITY_THRESHOLD = 3

# Regardless of distortion type, any image at this severity level is
# flagged as "corruption" (severe degradation), in addition to whatever
# category its own type maps to.
ALWAYS_CORRUPTION_LEVEL = 5

# --- quality_score formula -----------------------------------------------
# DMOS in this KADID-10k release is on a 1-5 scale where higher = BETTER
# (verified empirically -- see module docstring). To get a 0-100 scale
# where higher = better, matching the spec's original intent, dmos=1 (worst
# rated) maps to 0 and dmos=5 (best possible on the scale) maps to 100:
#   quality_score = 100 * (dmos - 1) / 4
# Reference (pristine) images have no DMOS rating at all (they were never
# distorted, so they were never shown to raters) -- they're assigned
# quality_score = 100.0 directly, consistent with dmos=5 mapping to 100.
DMOS_SCALE_MIN = 1.0
DMOS_SCALE_MAX = 5.0


def quality_score_from_dmos(dmos: float) -> float:
    raw = 100.0 * (dmos - DMOS_SCALE_MIN) / (DMOS_SCALE_MAX - DMOS_SCALE_MIN)
    return round(raw, 4)


def parse_distorted_filename(filename: str) -> tuple[str, int, int]:
    match = DISTORTED_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected distorted-image pattern: {filename}")
    ref_num, type_id, level = match.groups()
    return f"I{ref_num}", int(type_id), int(level)


def parse_reference_filename(filename: str) -> str:
    match = REFERENCE_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected reference-image pattern: {filename}")
    return f"I{match.group(1)}"


def build_issue_labels(type_id: int, level: int) -> dict[str, int]:
    labels = {issue: 0 for issue in ISSUE_LABELS}
    category = DISTORTION_CATEGORY_MAP.get(type_id)
    if category is not None and level >= SEVERITY_THRESHOLD:
        labels[category] = 1
    if level >= ALWAYS_CORRUPTION_LEVEL:
        labels[CORRUPTION] = 1
    return labels


def build_distorted_rows(dmos_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in dmos_df.iterrows():
        filename = row["dist_img"]
        reference_id, type_id, level = parse_distorted_filename(filename)
        assert reference_id == row["ref_img"].removesuffix(".png"), (
            f"reference id parsed from {filename} ({reference_id}) does not match "
            f"dmos.csv's ref_img column ({row['ref_img']})"
        )
        dmos = float(row["dmos"])
        record = {
            "filename": filename,
            "reference_id": reference_id,
            "kadid_distortion_type": DISTORTION_TYPE_NAMES[type_id],
            "kadid_distortion_level": level,
            "dmos": dmos,
            "quality_score": quality_score_from_dmos(dmos),
        }
        record.update(build_issue_labels(type_id, level))
        rows.append(record)
    return pd.DataFrame(rows)


def build_reference_rows(images_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(images_dir.glob("I*.png")):
        if not REFERENCE_FILENAME_RE.match(path.name):
            continue
        reference_id = parse_reference_filename(path.name)
        record = {
            "filename": path.name,
            "reference_id": reference_id,
            "kadid_distortion_type": "none",
            "kadid_distortion_level": 0,
            "dmos": float("nan"),
            "quality_score": 100.0,
        }
        record.update({issue: 0 for issue in ISSUE_LABELS})
        rows.append(record)
    return pd.DataFrame(rows)


# --- train/val/test split, at the reference-image level -------------------
SPLIT_RANDOM_SEED = 42
SPLIT_COUNTS = {"train": 65, "val": 8, "test": 8}  # sums to 81, ~80/10/10


def build_split(reference_ids: list[str]) -> pd.DataFrame:
    assert sum(SPLIT_COUNTS.values()) == len(reference_ids), (
        f"SPLIT_COUNTS sums to {sum(SPLIT_COUNTS.values())}, "
        f"but there are {len(reference_ids)} reference ids"
    )
    shuffled = sorted(reference_ids)  # sort first for determinism pre-shuffle
    rng = pd.Series(shuffled).sample(frac=1.0, random_state=SPLIT_RANDOM_SEED).tolist()

    split_rows = []
    cursor = 0
    for split_name, count in SPLIT_COUNTS.items():
        for reference_id in rng[cursor : cursor + count]:
            split_rows.append({"reference_id": reference_id, "split": split_name})
        cursor += count

    split_df = pd.DataFrame(split_rows)

    counts_per_ref = split_df.groupby("reference_id").size()
    assert (counts_per_ref == 1).all(), (
        "one or more reference_id values appear in more than one split: "
        f"{counts_per_ref[counts_per_ref > 1].index.tolist()}"
    )
    return split_df


def print_summary(labels_df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    print(f"\nTotal images processed: {len(labels_df)}")

    print("\nClass balance per issue label (positives / total, %):")
    for issue in ISSUE_LABELS:
        positives = int(labels_df[issue].sum())
        total = len(labels_df)
        print(f"  {issue:15s} {positives:5d} / {total} ({100 * positives / total:5.1f}%)")

    print("\nquality_score distribution:")
    print(labels_df["quality_score"].describe().to_string())

    print("\nSplit sizes (by reference image, then by resulting row count):")
    split_ref_counts = split_df["split"].value_counts()
    merged = labels_df.merge(split_df, on="reference_id", how="left")
    for split_name in ["train", "val", "test"]:
        n_refs = int(split_ref_counts.get(split_name, 0))
        n_rows = int((merged["split"] == split_name).sum())
        print(f"  {split_name:5s}: {n_refs} reference images -> {n_rows} labeled rows")


def main() -> None:
    if not DMOS_CSV_PATH.exists():
        print(f"ERROR: {DMOS_CSV_PATH} not found. Run download_kadid10k.py first.", file=sys.stderr)
        sys.exit(1)

    dmos_df = pd.read_csv(DMOS_CSV_PATH)
    distorted_rows = build_distorted_rows(dmos_df)
    reference_rows = build_reference_rows(IMAGES_DIR)

    labels_df = pd.concat([reference_rows, distorted_rows], ignore_index=True)
    labels_df = labels_df.sort_values(["reference_id", "kadid_distortion_type", "kadid_distortion_level"]).reset_index(drop=True)

    reference_ids = sorted(labels_df["reference_id"].unique().tolist())
    split_df = build_split(reference_ids)

    labels_df.to_csv(LABELS_CSV_PATH, index=False)
    split_df.to_csv(SPLIT_CSV_PATH, index=False)

    print(f"Wrote {LABELS_CSV_PATH} ({len(labels_df)} rows)")
    print(f"Wrote {SPLIT_CSV_PATH} ({len(split_df)} rows)")

    print_summary(labels_df, split_df)


if __name__ == "__main__":
    main()
