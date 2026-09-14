"""
Downloads the real KADID-10k benchmark archive (kadid10k.tgz) from the
Hugging Face dataset mirror chaofengc/IQA-PyTorch-Datasets, extracts it into
ml_training/data_gen/kadid10k/, and verifies the extraction:

  - counts the extracted image files and compares against the expected
    10,125 distorted images (81 reference images x 25 distortion types x 5
    severity levels) plus 81 pristine reference images,
  - locates the shipped metadata file(s) (.csv/.txt found anywhere in the
    extracted tree) and prints their exact column names plus a few sample
    rows, so the schema is confirmed by inspection rather than assumed.

This archive is a real, publicly hosted mirror of the KADID-10k benchmark
(sourced originally from Pixabay-licensed photos) -- no synthetic
degradation is generated here.

Usage: python ml_training/data_gen/download_kadid10k.py
"""

import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "chaofengc/IQA-PyTorch-Datasets"
REPO_TYPE = "dataset"
ARCHIVE_FILENAME = "kadid10k.tgz"

DATA_GEN_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = DATA_GEN_DIR / "kadid10k"

EXPECTED_REFERENCE_IMAGES = 81
EXPECTED_DISTORTION_TYPES = 25
EXPECTED_SEVERITY_LEVELS = 5
EXPECTED_DISTORTED_IMAGES = (
    EXPECTED_REFERENCE_IMAGES * EXPECTED_DISTORTION_TYPES * EXPECTED_SEVERITY_LEVELS
)  # 10,125

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
METADATA_SUFFIXES = {".csv", ".txt"}


def download_archive() -> Path:
    print(f"Downloading {ARCHIVE_FILENAME} from {REPO_ID} (~3GB, this will take a while)...")
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=ARCHIVE_FILENAME,
        local_dir=DATA_GEN_DIR,
    )
    print(f"Downloaded to {archive_path}")
    return Path(archive_path)


def extract_archive(archive_path: Path) -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"{EXTRACT_DIR} already exists and is non-empty -- skipping extraction.")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path.name} into {EXTRACT_DIR} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)
    print("Extraction complete.")

    flatten_redundant_wrapper_dir(EXTRACT_DIR)
    remove_macos_junk(EXTRACT_DIR)


def flatten_redundant_wrapper_dir(root: Path) -> None:
    """
    The archive's own top-level folder is itself named "kadid10k", so
    extracting it into a directory also named "kadid10k" produces a
    redundant kadid10k/kadid10k/ nesting. If that's what happened, move the
    inner folder's contents up one level and remove the now-empty wrapper.
    """
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        print(f"Flattening redundant wrapper directory {wrapper.name}/ ...")
        for item in wrapper.iterdir():
            shutil.move(str(item), str(root / item.name))
        wrapper.rmdir()


def remove_macos_junk(root: Path) -> None:
    """
    Mac-packaged tarballs commonly include AppleDouble resource-fork files
    (._*) and .DS_Store files alongside real content. These are never
    referenced by dmos.csv and aren't real images, so they're removed here
    rather than silently miscounted as data.
    """
    junk = [p for p in root.rglob("*") if p.is_file() and (p.name.startswith("._") or p.name == ".DS_Store")]
    if junk:
        print(f"Removing {len(junk)} macOS junk file(s) (._* / .DS_Store)...")
        for p in junk:
            p.unlink()


def find_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")]


def find_metadata_candidates(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in METADATA_SUFFIXES and not p.name.startswith("._")]


def main() -> None:
    archive_path = download_archive()
    extract_archive(archive_path)

    images = find_images(EXTRACT_DIR)
    expected_total = EXPECTED_DISTORTED_IMAGES + EXPECTED_REFERENCE_IMAGES
    print(f"\nFound {len(images)} image files under {EXTRACT_DIR}")
    print(
        f"Expected: {EXPECTED_DISTORTED_IMAGES} distorted + "
        f"{EXPECTED_REFERENCE_IMAGES} reference = {expected_total} total "
        "(exact packaging may vary slightly by mirror)"
    )

    metadata_candidates = find_metadata_candidates(EXTRACT_DIR)
    print(f"\nFound {len(metadata_candidates)} candidate metadata file(s):")
    for path in metadata_candidates:
        print(f"  {path.relative_to(EXTRACT_DIR)}  ({path.stat().st_size} bytes)")

    if not metadata_candidates:
        print(
            "WARNING: no .csv/.txt metadata file found under the extracted archive.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n--- Metadata file contents (first 5 lines of each candidate) ---")
    for path in metadata_candidates:
        print(f"\n### {path.relative_to(EXTRACT_DIR)}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip())


if __name__ == "__main__":
    main()
