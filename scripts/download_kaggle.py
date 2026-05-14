"""Download the Kaggle 'Best Artworks of All Time' dataset to data/raw/.

Auth: place your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
Get it from https://www.kaggle.com → Account → Create New API Token.

Run:
    python scripts/download_kaggle.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "ikarus777/best-artworks-of-all-time"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _check_credentials() -> None:
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.is_file():
        sys.exit(
            "ERROR: ~/.kaggle/kaggle.json not found.\n"
            "  1. Go to https://www.kaggle.com → Account → Create New API Token\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )


def main() -> None:
    _check_credentials()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET} to {RAW_DIR} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)],
        check=True,
    )

    zip_path = RAW_DIR / "best-artworks-of-all-time.zip"
    if not zip_path.is_file():
        sys.exit(f"ERROR: expected {zip_path} after download, not found.")

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"Done. Raw dataset at {RAW_DIR}")
    print("Next: python scripts/build_samples.py")


if __name__ == "__main__":
    main()
