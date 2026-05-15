"""End-to-end pipeline: sample -> generate (x2 conditions) -> TTS -> evaluate.

Usage::

    python -m src.pipeline --n_samples 100 --tts_samples 20

The pipeline is *idempotent*: every paid step is cached on disk, so a partial
run that crashes mid-way can be resumed by re-running the same command.
Individual artwork failures are logged and skipped — a single bad image
should not halt a 200-call run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import (
    AUDIO_DIR,
    CONDITIONS,
    DEFAULT_N_SAMPLES,
    DEFAULT_TTS_SAMPLES,
    DESCRIPTIONS_DIR,
    EVAL_RESULTS_CSV,
    SAMPLE_INDEX_CSV,
    ensure_dirs,
)
from src.curator import ArtCurator, CurationRecord
from src.dataset import ArtworkSample, sample_artworks, write_sample_index
from src.evaluator import evaluate_all
from src.tts import Narrator

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the BLV art-curation Baseline vs ABS pipeline.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Total stratified sample size (default: %(default)s).",
    )
    p.add_argument(
        "--tts_samples",
        type=int,
        default=DEFAULT_TTS_SAMPLES,
        help=(
            "How many artworks (per condition) to also synthesize as MP3. "
            "Set to 0 to skip TTS entirely."
        ),
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip the final evaluation pass (just generate descriptions).",
    )
    p.add_argument(
        "--no_bertscore",
        action="store_true",
        help="Skip the slow BERTScore step during evaluation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by stratified sampling (default: %(default)s).",
    )
    return p.parse_args(argv)


def _generate_descriptions(
    curator: ArtCurator,
    samples: list[ArtworkSample],
) -> list[CurationRecord]:
    """Loop over samples x conditions and produce/load every description."""
    records: list[CurationRecord] = []
    pbar = tqdm(
        total=len(samples) * len(CONDITIONS),
        desc="curate",
        unit="call",
    )
    for sample in samples:
        for condition in CONDITIONS:
            try:
                rec = curator.curate(sample, condition)
                records.append(rec)
            except Exception as e:  # noqa: BLE001 — never abort the whole batch
                logger.exception(
                    "curate failed for %s [%s]: %s", sample.artwork_id, condition, e,
                )
            pbar.update(1)
    pbar.close()
    return records


def _generate_tts(
    narrator: Narrator,
    samples: list[ArtworkSample],
    descriptions_by_key: dict[tuple[str, str], CurationRecord],
    n_tts: int,
) -> None:
    """Synthesize audio for the first ``n_tts`` samples in *both* conditions."""
    if n_tts <= 0:
        return
    chosen = samples[:n_tts]
    pbar = tqdm(total=len(chosen) * len(CONDITIONS), desc="tts", unit="call")
    for sample in chosen:
        for condition in CONDITIONS:
            rec = descriptions_by_key.get((sample.artwork_id, condition))
            if rec is None or not rec.description.strip():
                logger.warning(
                    "skipping TTS for %s [%s] — no description.",
                    sample.artwork_id, condition,
                )
                pbar.update(1)
                continue
            try:
                narrator.synthesize(
                    rec.description,
                    artwork_id=sample.artwork_id,
                    condition=condition,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "tts failed for %s [%s]: %s", sample.artwork_id, condition, e,
                )
            pbar.update(1)
    pbar.close()


def _print_summary(records: list[CurationRecord], eval_df: pd.DataFrame | None) -> None:
    """Compact stdout summary at the end of a run."""
    if not records:
        print("No curation records were produced.")
        return

    df = pd.DataFrame([r.to_dict() for r in records])
    print("\n=== Generation summary ===")
    print(
        df.groupby("condition")
        .agg(
            n=("artwork_id", "count"),
            mean_chars=("description", lambda s: s.str.len().mean().round(1)),
            total_cost_usd=("cost_usd", lambda s: round(s.sum(), 3)),
            mean_latency_sec=("latency_sec", lambda s: round(s.mean(), 2)),
        )
        .to_string()
    )

    if eval_df is None or eval_df.empty:
        return

    print("\n=== Color-word ratio (lower = better for ABS) ===")
    by_cond = eval_df.groupby("condition")["color_word_ratio"].agg(["mean", "std"])
    print(by_cond.round(4).to_string())

    if {"baseline", "abs"}.issubset(set(by_cond.index)):
        base = by_cond.loc["baseline", "mean"]
        abs_ = by_cond.loc["abs", "mean"]
        if base > 0:
            delta = (base - abs_) / base * 100
            print(f"\nABS reduces color-word ratio by {delta:+.1f}% vs Baseline.")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    ensure_dirs()

    logger.info("Sampling %d artworks (seed=%d)…", args.n_samples, args.seed)
    samples = sample_artworks(n_samples=args.n_samples, seed=args.seed)
    if not samples:
        logger.error("No samples produced — check the dataset path.")
        return 1
    write_sample_index(samples, SAMPLE_INDEX_CSV)

    curator = ArtCurator()
    records = _generate_descriptions(curator, samples)

    descriptions_by_key = {(r.artwork_id, r.condition): r for r in records}
    if args.tts_samples > 0:
        narrator = Narrator()
        _generate_tts(narrator, samples, descriptions_by_key, args.tts_samples)
    else:
        logger.info("TTS skipped (tts_samples=0).")

    eval_df: pd.DataFrame | None = None
    if not args.skip_eval:
        eval_df = evaluate_all(
            desc_dir=DESCRIPTIONS_DIR,
            out_csv=EVAL_RESULTS_CSV,
            compute_bertscore=not args.no_bertscore,
        )
    else:
        logger.info("Evaluation skipped (--skip_eval).")

    _print_summary(records, eval_df)
    print(f"\nDescriptions:   {DESCRIPTIONS_DIR}")
    print(f"Audio:          {AUDIO_DIR}")
    print(f"Eval results:   {EVAL_RESULTS_CSV}")
    print(f"Sample index:   {SAMPLE_INDEX_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
