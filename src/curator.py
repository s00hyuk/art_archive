"""GPT-4o Vision call: image + prompt -> Korean description.

This wraps a single OpenAI chat-completion call and adds:

* On-disk caching at ``output/descriptions/{artwork_id}.json`` — if a
  condition has already been generated, we skip the (paid) API call.
* Retry-with-exponential-backoff on rate limits and transient errors.
* Cost accounting based on the prompt/completion token usage returned by
  OpenAI.

The returned record contains every field needed by the evaluator and the
analysis notebook, so we never have to re-call the API to re-evaluate.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError
from PIL import Image

from src.config import (
    API_BACKOFF_BASE_SEC,
    API_MAX_RETRIES,
    DESCRIPTIONS_DIR,
    VLM_IMAGE_MAX_EDGE,
    VLM_INPUT_PRICE_PER_M,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_OUTPUT_PRICE_PER_M,
    VLM_TEMPERATURE,
)
from src.dataset import ArtworkSample
from src.prompts import abs_prompt, baseline_prompt, build_user_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CurationRecord:
    """One image x one condition output, including telemetry."""

    artwork_id: str
    condition: str  # "baseline" | "abs"
    artist: str
    genre: str
    nationality: str
    description: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    system_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artwork_id": self.artwork_id,
            "condition": self.condition,
            "artist": self.artist,
            "genre": self.genre,
            "nationality": self.nationality,
            "description": self.description,
            "latency_sec": round(self.latency_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "system_prompt": self.system_prompt,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image_data_url(image_path: Path, max_edge: int = VLM_IMAGE_MAX_EDGE) -> str:
    """JPEG-downscale and base64-encode an image as a ``data:`` URL.

    Down-scaling keeps base64 payloads small enough that even slow uplinks
    finish the request quickly, and OpenAI's "high detail" tier still has
    plenty of pixels to work with at 1024 px on the long edge.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_edge / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for one call given OpenAI's per-token pricing."""
    return (
        prompt_tokens / 1_000_000 * VLM_INPUT_PRICE_PER_M
        + completion_tokens / 1_000_000 * VLM_OUTPUT_PRICE_PER_M
    )


def _cached_path(artwork_id: str, condition: str, out_dir: Path = DESCRIPTIONS_DIR) -> Path:
    return out_dir / f"{artwork_id}__{condition}.json"


def _load_cached(artwork_id: str, condition: str, out_dir: Path = DESCRIPTIONS_DIR) -> CurationRecord | None:
    p = _cached_path(artwork_id, condition, out_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return CurationRecord(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Stale cache at %s (%s) — ignoring.", p, e)
        return None


def _save(record: CurationRecord, out_dir: Path = DESCRIPTIONS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _cached_path(record.artwork_id, record.condition, out_dir)
    p.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------

class ArtCurator:
    """Stateless wrapper around the OpenAI vision call.

    A single ``ArtCurator`` is meant to be reused across all samples in a run:
    its only state is the OpenAI client and the configured model name.
    """

    def __init__(self, model: str = VLM_MODEL, api_key: str | None = None) -> None:
        load_dotenv()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # ---------------- public API ----------------

    def curate(
        self,
        sample: ArtworkSample,
        condition: str,
        *,
        use_cache: bool = True,
        out_dir: Path = DESCRIPTIONS_DIR,
    ) -> CurationRecord:
        """Generate (or load from cache) one description.

        Args:
            sample: The artwork to describe.
            condition: ``"baseline"`` or ``"abs"``.
            use_cache: If ``True``, skip the API call when a cached record
                exists. Cached records are returned untouched.
            out_dir: Where to read/write the JSON cache file.

        Returns:
            A populated ``CurationRecord``.

        Raises:
            ValueError: If ``condition`` is unknown.
            openai.OpenAIError: If retries are exhausted.
        """
        if condition not in {"baseline", "abs"}:
            raise ValueError(f"unknown condition: {condition!r}")

        if use_cache:
            cached = _load_cached(sample.artwork_id, condition, out_dir)
            if cached is not None:
                logger.debug("cache hit: %s / %s", sample.artwork_id, condition)
                return cached

        system_prompt = (
            baseline_prompt()
            if condition == "baseline"
            else abs_prompt(sample.artist, sample.genre, sample.nationality)
        )
        user_text = build_user_message(condition, sample.title_guess)

        data_url = _encode_image_data_url(sample.image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            },
        ]

        t0 = time.perf_counter()
        response = self._call_with_retries(messages)
        latency = time.perf_counter() - t0

        description = (response.choices[0].message.content or "").strip()
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        record = CurationRecord(
            artwork_id=sample.artwork_id,
            condition=condition,
            artist=sample.artist,
            genre=sample.genre,
            nationality=sample.nationality,
            description=description,
            latency_sec=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_compute_cost(prompt_tokens, completion_tokens),
            model=self.model,
            system_prompt=system_prompt,
        )
        _save(record, out_dir)
        return record

    # ---------------- internals ----------------

    def _call_with_retries(self, messages: list[dict[str, Any]]):
        """Call OpenAI with exponential backoff on rate-limit/timeout errors."""
        last_exc: Exception | None = None
        for attempt in range(API_MAX_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=VLM_TEMPERATURE,
                    max_tokens=VLM_MAX_TOKENS,
                )
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_exc = e
                wait = API_BACKOFF_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "OpenAI error (%s) on attempt %d/%d — sleeping %.1fs",
                    type(e).__name__, attempt + 1, API_MAX_RETRIES, wait,
                )
                time.sleep(wait)
        # Out of retries.
        assert last_exc is not None
        raise last_exc
