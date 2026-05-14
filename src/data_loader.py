"""Load and sample paintings from the Kaggle 'Best Artworks of All Time' dataset.

Expected raw layout (after scripts/download_kaggle.py):

    data/raw/
        artists.csv
        images/<Artist_Name>/<Artist_Name>_<N>.jpg
        resized/<Artist_Name>_<N>.jpg   # optional, smaller copies

Sampled output (after scripts/build_samples.py):

    data/samples/
        sampled.csv
        images/<Artist_Name>_<N>.jpg
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

TARGET_ARTISTS = ("Vincent van Gogh", "Claude Monet", "Leonardo da Vinci")
SAMPLES_PER_ARTIST = 5


@dataclass(frozen=True)
class ArtworkRecord:
    artwork_id: str
    artist: str
    image_path: Path

    def exists(self) -> bool:
        return self.image_path.is_file()


def _artist_folder_name(artist: str) -> str:
    # Kaggle dataset uses underscored folder names, e.g. "Vincent_van_Gogh".
    return artist.replace(" ", "_")


def _list_artist_images(raw_images_dir: Path, artist: str) -> list[Path]:
    folder = raw_images_dir / _artist_folder_name(artist)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_artists_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the artists.csv that ships with the Kaggle dataset."""
    csv_path = raw_dir / "artists.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"artists.csv not found at {csv_path}. "
            "Run `python scripts/download_kaggle.py` first."
        )
    return pd.read_csv(csv_path)


def sample_artworks(
    raw_dir: Path = RAW_DIR,
    artists: tuple[str, ...] = TARGET_ARTISTS,
    per_artist: int = SAMPLES_PER_ARTIST,
    seed: int = 42,
) -> list[ArtworkRecord]:
    """Pick `per_artist` paintings for each artist from the raw dataset."""
    rng = random.Random(seed)
    raw_images_dir = raw_dir / "images" / "images"  # Kaggle layout: images/images/<Artist>
    if not raw_images_dir.is_dir():
        # Fallback: some unpacks put it directly at images/<Artist>
        raw_images_dir = raw_dir / "images"

    records: list[ArtworkRecord] = []
    for artist in artists:
        candidates = _list_artist_images(raw_images_dir, artist)
        if len(candidates) < per_artist:
            raise RuntimeError(
                f"Only found {len(candidates)} images for {artist} in {raw_images_dir}. "
                f"Expected at least {per_artist}."
            )
        picked = rng.sample(candidates, per_artist)
        for image_path in picked:
            records.append(
                ArtworkRecord(
                    artwork_id=image_path.stem,
                    artist=artist,
                    image_path=image_path,
                )
            )
    return records


def load_samples(samples_dir: Path = SAMPLES_DIR) -> list[ArtworkRecord]:
    """Load the 15 pre-sampled artworks committed to the repo."""
    csv_path = samples_dir / "sampled.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python scripts/build_samples.py` first."
        )
    df = pd.read_csv(csv_path)
    images_dir = samples_dir / "images"
    return [
        ArtworkRecord(
            artwork_id=row["artwork_id"],
            artist=row["artist"],
            image_path=images_dir / row["filename"],
        )
        for _, row in df.iterrows()
    ]


if __name__ == "__main__":
    # Quick sanity check
    try:
        records = load_samples()
        print(f"Loaded {len(records)} sampled artworks:")
        for r in records:
            mark = "✓" if r.exists() else "✗"
            print(f"  {mark} [{r.artist}] {r.artwork_id} -> {r.image_path}")
    except FileNotFoundError as e:
        print(e)
