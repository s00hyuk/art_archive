"""Stratified sampling of artworks from the Kaggle 'Best Artworks of All Time' set.

The Kaggle dataset ships an ``artists.csv`` with one row per artist; the
``genre`` column is a comma-separated list of art movements. We resolve each
artist to a *primary* genre, group artists by genre, then balance the sample
across genres so every condition is comparable across movements.

Expected on-disk layout (the spec calls for ``./data/best-artworks/`` but we
also accept the older ``./data/raw/`` location used by the existing
``scripts/download_kaggle.py``)::

    data/best-artworks/
        artists.csv
        images/images/<Artist_Name>/<Artist_Name>_<N>.jpg
        resized/<Artist_Name>_<N>.jpg   # optional smaller copies
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    BEST_ARTWORKS_DIR,
    LEGACY_RAW_DIR,
    RANDOM_SEED,
    SAMPLE_INDEX_CSV,
    TARGET_GENRES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtworkSample:
    """One sampled artwork plus enough metadata to drive prompts and analysis.

    Attributes:
        artwork_id: Filesystem-safe identifier (file stem from Kaggle).
        artist: Full artist name from ``artists.csv``.
        genre: Resolved primary genre (matches one of ``TARGET_GENRES``).
        nationality: Artist nationality from ``artists.csv``.
        image_path: Absolute path to the JPG on disk.
        title_guess: Best-effort title derived from the filename. The Kaggle
            dataset does *not* include true titles; this is a placeholder
            useful for logging and ground-truth pairing.
    """

    artwork_id: str
    artist: str
    genre: str
    nationality: str
    image_path: Path
    title_guess: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    """Find the on-disk Kaggle dataset, preferring the spec'd location.

    Returns:
        The directory that contains ``artists.csv`` and ``images/``.

    Raises:
        FileNotFoundError: If no candidate location contains ``artists.csv``.
    """
    candidates = [BEST_ARTWORKS_DIR, LEGACY_RAW_DIR]
    for c in candidates:
        if (c / "artists.csv").is_file():
            return c
    raise FileNotFoundError(
        "Could not locate the Kaggle dataset. Looked for artists.csv in: "
        + ", ".join(str(c) for c in candidates)
        + ". Download via scripts/download_kaggle.py or place it manually."
    )


def _resolve_images_root(data_dir: Path) -> Path:
    """Return the directory under which per-artist image folders live.

    Kaggle's archive nests images at ``images/images/<Artist>/...`` but some
    unpackers flatten that to ``images/<Artist>/...``. We try both.
    """
    nested = data_dir / "images" / "images"
    if nested.is_dir():
        return nested
    flat = data_dir / "images"
    if flat.is_dir():
        return flat
    raise FileNotFoundError(f"No images/ directory under {data_dir}")


def _artist_folder_name(artist: str) -> str:
    """Convert ``"Vincent van Gogh"`` to the Kaggle folder ``"Vincent_van_Gogh"``."""
    return artist.replace(" ", "_")


def _list_artist_images(images_root: Path, artist: str) -> list[Path]:
    folder = images_root / _artist_folder_name(artist)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _title_guess_from_filename(stem: str, artist: str) -> str:
    """Heuristic title from the file stem.

    The Kaggle filenames look like ``Vincent_van_Gogh_42.jpg``. Stripping the
    artist prefix and trailing index leaves nothing useful, so we just return
    the stem with underscores swapped for spaces. Downstream code treats this
    as opaque metadata.
    """
    artist_prefix = _artist_folder_name(artist)
    if stem.startswith(artist_prefix):
        suffix = stem[len(artist_prefix):].lstrip("_")
        return suffix or stem.replace("_", " ")
    return stem.replace("_", " ")


def _primary_genre(raw_genre: str | float, targets: Iterable[str]) -> str | None:
    """Match the longest ``targets`` entry that appears in ``raw_genre``.

    The Kaggle ``genre`` column is a comma-separated list (e.g.
    ``"Post-Impressionism,Symbolism"``). We need longest-first matching
    so an artist labeled ``"Post-Impressionism"`` isn't claimed by the
    shorter target ``"Impressionism"``.
    """
    if not isinstance(raw_genre, str):
        return None
    haystack = raw_genre.lower()
    for target in sorted(targets, key=len, reverse=True):
        if target.lower() in haystack:
            return target
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_artists_metadata(data_dir: Path | None = None) -> pd.DataFrame:
    """Load ``artists.csv``, adding a ``primary_genre`` column.

    Args:
        data_dir: Optional override for the dataset location. If ``None``, the
            standard locations (``data/best-artworks/``, ``data/raw/``) are
            searched.

    Returns:
        The artists table with an extra ``primary_genre`` column (``None`` for
        artists whose declared genres don't match any ``TARGET_GENRES`` entry).
    """
    data_dir = data_dir or _resolve_data_dir()
    df = pd.read_csv(data_dir / "artists.csv")
    df["primary_genre"] = df["genre"].apply(
        lambda g: _primary_genre(g, TARGET_GENRES)
    )
    return df


def sample_artworks(
    n_samples: int = 100,
    target_genres: tuple[str, ...] = TARGET_GENRES,
    seed: int = RANDOM_SEED,
    data_dir: Path | None = None,
) -> list[ArtworkSample]:
    """Draw a stratified sample of artworks balanced across genres.

    Per-genre quota is ``ceil(n_samples / len(target_genres))``; if a genre
    has fewer images available than its quota, we take them all and the
    overall sample size shrinks accordingly. Within a genre, we first
    distribute the quota evenly across that genre's artists (a single
    superstar artist cannot dominate their movement).

    Args:
        n_samples: Target total sample size. The actual size may be slightly
            smaller (under-supply) or rounded up by per-genre ceiling.
        target_genres: Genres to stratify over.
        seed: RNG seed for reproducibility.
        data_dir: Optional dataset-location override.

    Returns:
        List of ``ArtworkSample`` records, sorted by ``(genre, artist,
        artwork_id)`` for deterministic downstream iteration.
    """
    data_dir = data_dir or _resolve_data_dir()
    images_root = _resolve_images_root(data_dir)
    rng = random.Random(seed)

    artists_df = load_artists_metadata(data_dir)
    per_genre_quota = -(-n_samples // len(target_genres))  # ceil div

    samples: list[ArtworkSample] = []
    for genre in target_genres:
        genre_artists = artists_df[artists_df["primary_genre"] == genre]
        if genre_artists.empty:
            logger.warning("No artists matched genre %r — skipping.", genre)
            continue

        # Map artist -> available image paths.
        artist_to_images: dict[str, list[Path]] = {}
        for _, row in genre_artists.iterrows():
            imgs = _list_artist_images(images_root, row["name"])
            if imgs:
                artist_to_images[row["name"]] = imgs

        if not artist_to_images:
            logger.warning("No images on disk for any %s artist — skipping.", genre)
            continue

        # Distribute the genre quota across that genre's artists round-robin.
        per_artist = max(1, per_genre_quota // len(artist_to_images))
        picked_in_genre: list[tuple[str, Path]] = []
        for artist, imgs in artist_to_images.items():
            take = min(per_artist, len(imgs))
            for img in rng.sample(imgs, take):
                picked_in_genre.append((artist, img))
            if len(picked_in_genre) >= per_genre_quota:
                break

        # If round-robin under-filled the quota, top up by sampling more from
        # whichever artist still has images left.
        remaining = per_genre_quota - len(picked_in_genre)
        if remaining > 0:
            already_taken = {p for _, p in picked_in_genre}
            leftover = [
                (a, p)
                for a, imgs in artist_to_images.items()
                for p in imgs
                if p not in already_taken
            ]
            rng.shuffle(leftover)
            picked_in_genre.extend(leftover[:remaining])

        for artist, img_path in picked_in_genre[:per_genre_quota]:
            row = genre_artists[genre_artists["name"] == artist].iloc[0]
            samples.append(
                ArtworkSample(
                    artwork_id=img_path.stem,
                    artist=artist,
                    genre=genre,
                    nationality=str(row.get("nationality", "")),
                    image_path=img_path,
                    title_guess=_title_guess_from_filename(img_path.stem, artist),
                )
            )

    samples.sort(key=lambda s: (s.genre, s.artist, s.artwork_id))
    logger.info(
        "Sampled %d artworks across %d genres (target N=%d).",
        len(samples), len({s.genre for s in samples}), n_samples,
    )
    return samples


def write_sample_index(samples: list[ArtworkSample], path: Path = SAMPLE_INDEX_CSV) -> Path:
    """Persist the sample list as CSV so the index is reproducible across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(s), "image_path": str(s.image_path)} for s in samples]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    samples = sample_artworks()
    write_sample_index(samples)
    by_genre = pd.Series([s.genre for s in samples]).value_counts()
    print(f"Total samples: {len(samples)}")
    print("Per-genre counts:")
    print(by_genre.to_string())
