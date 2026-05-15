"""Evaluation metrics for the Baseline vs ABS comparison.

The headline number for the paper is **color-word ratio**: the fraction of
tokens in a description that are visual color words. The ABS treatment
should drive this close to zero; the baseline is expected to stay much
higher. Substring matching is intentional — Korean color terms have many
morphological variants and a real-coverage stem list (see
``config.COLOR_WORDS_KO``) gives more stable counts than a brittle
morphological analyser would.

Reference-based scores (BLEU / ROUGE-L / BERTScore) are computed only
when a ground-truth file exists at
``data/ground_truth/{artwork_id}.txt``. Otherwise the columns are NaN.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    COLOR_WORDS_KO,
    DESCRIPTIONS_DIR,
    EVAL_RESULTS_CSV,
    GROUND_TRUTH_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-metric primitives
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization, stripping punctuation.

    Returns:
        List of non-empty whitespace-separated tokens with surrounding
        punctuation removed. Not morphologically aware — good enough for
        denominator counts when the goal is a *ratio*.
    """
    cleaned = re.sub(r"[　-〿＀-￯.,;:!?\"'()\[\]{}<>·…—–-]", " ", text)
    return [t for t in _WHITESPACE_RE.split(cleaned) if t]


def color_word_ratio(text: str, lexicon: Iterable[str] = COLOR_WORDS_KO) -> float:
    """Fraction of tokens that contain any color-stem from ``lexicon``.

    Args:
        text: Description to score.
        lexicon: Iterable of color stems / modifiers. Defaults to the Korean
            list in ``config.COLOR_WORDS_KO``.

    Returns:
        ``hits / len(tokens)`` in [0, 1]. Returns 0.0 for empty input.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if any(c in t for c in lexicon))
    return hits / len(tokens)


def color_word_count(text: str, lexicon: Iterable[str] = COLOR_WORDS_KO) -> int:
    """Absolute count of color-stem token hits (numerator of the ratio)."""
    return sum(1 for t in tokenize(text) if any(c in t for c in lexicon))


def char_count(text: str) -> int:
    """Length in characters (whitespace counts as one char)."""
    return len(text)


def sentence_count(text: str) -> int:
    """Heuristic Korean/English sentence count.

    Splits on ``.!?。！？`` and newlines. Korean writing often omits
    sentence-final punctuation; in that case this falls back to a single
    sentence per non-empty input.
    """
    sents = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if sents:
        return len(sents)
    return 1 if text.strip() else 0


# ---------------------------------------------------------------------------
# Reference-based scores (lazy imports — these libs are slow to import)
# ---------------------------------------------------------------------------

def bleu_score(reference: str, hypothesis: str) -> float:
    """Corpus-style BLEU-4 with NLTK smoothing.

    NLTK is imported lazily because it pulls a sizable startup cost; we only
    pay it when a ground-truth file is actually available.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score from the ``rouge-score`` package."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def bert_score_batch(references: list[str], hypotheses: list[str]) -> list[float]:
    """BERTScore F1 for parallel reference/hypothesis lists.

    Uses ``bert-base-multilingual-cased`` for Korean. The model is loaded on
    first call; subsequent calls re-use the underlying cache.
    """
    from bert_score import score as _bertscore

    if not references:
        return []
    _, _, f1 = _bertscore(
        hypotheses,
        references,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANG,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(x) for x in f1.tolist()]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _ground_truth_path(artwork_id: str, gt_dir: Path = GROUND_TRUTH_DIR) -> Path:
    return gt_dir / f"{artwork_id}.txt"


def _load_ground_truth(artwork_id: str, gt_dir: Path = GROUND_TRUTH_DIR) -> str | None:
    p = _ground_truth_path(artwork_id, gt_dir)
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text or None


@dataclass
class EvalRow:
    """One row of the evaluation table — one (artwork_id, condition) pair."""

    artwork_id: str
    condition: str
    artist: str
    genre: str
    nationality: str
    char_count: int
    sentence_count: int
    color_word_count: int
    color_word_ratio: float
    latency_sec: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    has_ground_truth: bool
    bleu: float | None = None
    rouge_l: float | None = None
    bertscore_f1: float | None = None


def _load_curation_records(desc_dir: Path = DESCRIPTIONS_DIR) -> list[dict]:
    """Read every ``output/descriptions/*.json`` file the curator has written."""
    records: list[dict] = []
    for p in sorted(desc_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            logger.warning("skipping malformed %s: %s", p, e)
    return records


def evaluate_all(
    desc_dir: Path = DESCRIPTIONS_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
    out_csv: Path = EVAL_RESULTS_CSV,
    compute_bertscore: bool = True,
) -> pd.DataFrame:
    """Evaluate every cached curation and write a CSV.

    Args:
        desc_dir: Directory of per-artwork JSON files (curator output).
        gt_dir: Directory of optional ground-truth ``.txt`` files keyed by
            ``artwork_id``.
        out_csv: Destination CSV. Parents are created if missing.
        compute_bertscore: Set to ``False`` to skip the (slow, GPU-friendly)
            BERTScore pass.

    Returns:
        A pandas DataFrame mirroring the CSV layout.
    """
    raw = _load_curation_records(desc_dir)
    if not raw:
        logger.warning("No curation records found under %s", desc_dir)
        return pd.DataFrame()

    rows: list[EvalRow] = []
    gt_pairs: list[tuple[int, str, str]] = []  # (row_index, reference, hypothesis)

    for rec in raw:
        text = rec["description"]
        gt = _load_ground_truth(rec["artwork_id"], gt_dir)
        row = EvalRow(
            artwork_id=rec["artwork_id"],
            condition=rec["condition"],
            artist=rec["artist"],
            genre=rec["genre"],
            nationality=rec.get("nationality", ""),
            char_count=char_count(text),
            sentence_count=sentence_count(text),
            color_word_count=color_word_count(text),
            color_word_ratio=color_word_ratio(text),
            latency_sec=float(rec.get("latency_sec", 0.0)),
            cost_usd=float(rec.get("cost_usd", 0.0)),
            prompt_tokens=int(rec.get("prompt_tokens", 0)),
            completion_tokens=int(rec.get("completion_tokens", 0)),
            has_ground_truth=gt is not None,
        )
        if gt is not None:
            row.bleu = bleu_score(gt, text)
            row.rouge_l = rouge_l_score(gt, text)
            gt_pairs.append((len(rows), gt, text))
        rows.append(row)

    if compute_bertscore and gt_pairs:
        logger.info("Computing BERTScore on %d reference pairs…", len(gt_pairs))
        refs = [r for _, r, _ in gt_pairs]
        hyps = [h for _, _, h in gt_pairs]
        f1s = bert_score_batch(refs, hyps)
        for (idx, _, _), f1 in zip(gt_pairs, f1s):
            rows[idx].bertscore_f1 = f1

    df = pd.DataFrame([row.__dict__ for row in rows])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d eval rows to %s", len(df), out_csv)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = evaluate_all()
    if df.empty:
        print("No data to evaluate.")
    else:
        print(df.groupby("condition")[["color_word_ratio", "char_count", "cost_usd"]].mean())
