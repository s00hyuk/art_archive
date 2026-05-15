"""OpenAI TTS-1-HD: Korean description -> MP3 file.

TTS is the expensive last step, so we only run it on a small evaluation
subset (default N=20 per condition). The output filename encodes the
artwork id and condition so the audio file is unambiguous.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.config import (
    API_BACKOFF_BASE_SEC,
    API_MAX_RETRIES,
    AUDIO_DIR,
    TTS_MODEL,
    TTS_PRICE_PER_M_CHARS,
    TTS_VOICE,
)

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Path to the generated MP3 plus cost telemetry."""

    artwork_id: str
    condition: str
    audio_path: Path
    char_count: int
    cost_usd: float
    latency_sec: float


def _audio_path(artwork_id: str, condition: str, out_dir: Path = AUDIO_DIR) -> Path:
    return out_dir / f"{artwork_id}_{condition}.mp3"


def _compute_tts_cost(char_count: int) -> float:
    """USD cost for a TTS call at OpenAI's per-character pricing."""
    return char_count / 1_000_000 * TTS_PRICE_PER_M_CHARS


class Narrator:
    """Stateless wrapper around the OpenAI TTS endpoint."""

    def __init__(
        self,
        model: str = TTS_MODEL,
        voice: str = TTS_VOICE,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def synthesize(
        self,
        text: str,
        artwork_id: str,
        condition: str,
        *,
        out_dir: Path = AUDIO_DIR,
        use_cache: bool = True,
    ) -> TTSResult:
        """Convert ``text`` to an MP3 saved at ``out_dir``.

        Args:
            text: Description to synthesize. Must be non-empty.
            artwork_id: Identifier used in the output filename.
            condition: ``"baseline"`` or ``"abs"`` — included in the filename.
            out_dir: Where the MP3 is written. Created if missing.
            use_cache: If ``True`` and the destination file already exists,
                skip the API call. Cost is reported as 0 in that case.

        Returns:
            A ``TTSResult`` with the absolute MP3 path and cost telemetry.

        Raises:
            ValueError: If ``text`` is empty.
            openai.OpenAIError: If retries are exhausted.
        """
        if not text or not text.strip():
            raise ValueError("Refusing to synthesize empty text.")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _audio_path(artwork_id, condition, out_dir)
        char_count = len(text)

        if use_cache and out_path.is_file():
            logger.debug("tts cache hit: %s", out_path.name)
            return TTSResult(
                artwork_id=artwork_id,
                condition=condition,
                audio_path=out_path,
                char_count=char_count,
                cost_usd=0.0,
                latency_sec=0.0,
            )

        t0 = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                )
                response.stream_to_file(str(out_path))
                latency = time.perf_counter() - t0
                return TTSResult(
                    artwork_id=artwork_id,
                    condition=condition,
                    audio_path=out_path,
                    char_count=char_count,
                    cost_usd=_compute_tts_cost(char_count),
                    latency_sec=latency,
                )
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_exc = e
                wait = API_BACKOFF_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "TTS error (%s) attempt %d/%d — sleeping %.1fs",
                    type(e).__name__, attempt + 1, API_MAX_RETRIES, wait,
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc
