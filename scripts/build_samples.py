"""Pick 5 paintings each from Van Gogh / Monet / Da Vinci and copy them
into data/samples/ for commit-sized distribution.

Run after scripts/download_kaggle.py:
    python scripts/build_samples.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import SAMPLES_DIR, sample_artworks  # noqa: E402


def main() -> None:
    records = sample_artworks()  # uses default seed=42 for reproducibility
    images_out = SAMPLES_DIR / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        dest = images_out / r.image_path.name
        shutil.copy2(r.image_path, dest)
        rows.append(
            {
                "artwork_id": r.artwork_id,
                "artist": r.artist,
                "filename": r.image_path.name,
            }
        )

    csv_path = SAMPLES_DIR / "sampled.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} and copied {len(rows)} images to {images_out}")


if __name__ == "__main__":
    main()
