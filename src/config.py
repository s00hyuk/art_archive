"""Central configuration for the BLV art-curation PoC.

Paths, model names, pricing, and a handful of experiment-wide constants live
here so the rest of the codebase can import a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Kaggle "Best Artworks of All Time" dataset.
# Per the spec, raw data lives under ./data/best-artworks/. The legacy layout
# (data/raw/) is kept as a fallback so anyone who already downloaded via
# scripts/download_kaggle.py does not have to re-download.
DATA_DIR: Path = REPO_ROOT / "data"
BEST_ARTWORKS_DIR: Path = DATA_DIR / "best-artworks"
LEGACY_RAW_DIR: Path = DATA_DIR / "raw"
GROUND_TRUTH_DIR: Path = DATA_DIR / "ground_truth"

OUTPUT_DIR: Path = REPO_ROOT / "output"
DESCRIPTIONS_DIR: Path = OUTPUT_DIR / "descriptions"
AUDIO_DIR: Path = OUTPUT_DIR / "audio"
EVAL_RESULTS_CSV: Path = OUTPUT_DIR / "eval_results.csv"
SAMPLE_INDEX_CSV: Path = OUTPUT_DIR / "sample_index.csv"

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# Genres targeted for stratified sampling. These names must match the values
# in artists.csv (case-insensitive substring match — see dataset.py).
TARGET_GENRES: tuple[str, ...] = (
    "Renaissance",
    "Baroque",
    "Impressionism",
    "Post-Impressionism",
    "Expressionism",
    "Cubism",
    "Surrealism",
    "Symbolism",
)

DEFAULT_N_SAMPLES: int = 100
DEFAULT_TTS_SAMPLES: int = 20

CONDITIONS: tuple[str, ...] = ("baseline", "abs")

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

# Vision model used for the curation step.
VLM_MODEL: str = "gpt-4o"

# OpenAI text-to-speech model + voice.
TTS_MODEL: str = "tts-1-hd"
TTS_VOICE: str = "nova"  # "shimmer" is the alternative we A/B-tested for Korean

# Pricing (USD per 1M tokens / chars). Update if OpenAI changes its prices.
# Source: https://openai.com/api/pricing/ (snapshot used at PoC time).
VLM_INPUT_PRICE_PER_M: float = 2.50
VLM_OUTPUT_PRICE_PER_M: float = 10.00
TTS_PRICE_PER_M_CHARS: float = 30.00

# Vision-call generation knobs.
VLM_TEMPERATURE: float = 0.7
VLM_MAX_TOKENS: int = 700
VLM_IMAGE_MAX_EDGE: int = 1024  # px — downscaled before base64 encoding

# Retry / backoff for transient API errors.
API_MAX_RETRIES: int = 5
API_BACKOFF_BASE_SEC: float = 2.0

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Korean color-word dictionary used by evaluator.color_word_ratio.
# Substring matching (not morphological), so we include both modifier and
# stem forms (e.g. "파란" + "파랗").
COLOR_WORDS_KO: tuple[str, ...] = (
    # 파랑 계열
    "파란", "파랗", "파르", "푸른", "푸릇", "푸르스", "새파",
    # 빨강 계열
    "빨간", "빨갛", "붉은", "붉게", "새빨", "발그",
    # 노랑 계열
    "노란", "노랗", "누런", "샛노",
    # 초록 계열
    "초록", "푸르른", "녹색", "연두",
    # 검정/흰색/회색
    "검은", "검정", "까만", "까맣", "어두운",
    "하얀", "하얗", "흰", "새하",
    "회색", "잿빛",
    # 기타
    "주황", "주홍", "갈색", "보라", "자주", "분홍", "핑크",
    "금색", "은색", "남색", "청록",
)

# Multilingual BERT model used by bert-score for Korean evaluation.
BERTSCORE_MODEL: str = "bert-base-multilingual-cased"
BERTSCORE_LANG: str = "ko"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (
        OUTPUT_DIR,
        DESCRIPTIONS_DIR,
        AUDIO_DIR,
        GROUND_TRUTH_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
