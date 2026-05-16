"""VLM curation pipeline (v4 — naturalized).

Given an artwork image and the gold dataset's artist→dominant_sense map,
produce a 7-block Korean docent script (intro / composition prefix-with-천천히 /
step1 spatial overview / natural scan guide / sensory prefix / step2 sensory
zoom-in / natural merged closing) packaged as `tts_script`.

The system prompt at src/prompts/system_prompt.md carries the structure and
3 hand-curated few-shot examples (Van Gogh / Monet / Da Vinci). The gold
dataset at data/gold/sensedocent_100.csv provides the artist→dominant_sense
lookup.

Usage:
    from src.vlm_pipeline import SensoryCurator
    curator = SensoryCurator()
    result = curator.curate(image_path, artwork_id="...", artist="Vincent van Gogh")
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = REPO_ROOT / "src" / "prompts" / "system_prompt.md"
GOLD_CSV_PATH = REPO_ROOT / "data" / "gold" / "sensedocent_100.csv"

# Schema mirrors the gold dataset columns. OpenAI strict mode requires every
# `properties` field to also be listed in `required`, and additionalProperties
# must be false at every level.
CURATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "sensory_curation_v4",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artwork_id",
            "artist",
            "title_ko",
            "artwork_year",
            "period_style",
            "genre",
            "visual_feature",
            "target_sense",
            "dominant_sense_type",
            "sensory_mapping_rule",
            "step1_spatial_overview",
            "step2_sensory_zoom_in",
            "tts_script",
        ],
        "properties": {
            "artwork_id": {"type": "string"},
            "artist": {"type": "string"},
            "title_ko": {"type": "string"},
            "artwork_year": {"type": "string"},
            "period_style": {"type": "string"},
            "genre": {"type": "string"},
            "visual_feature": {"type": "string"},
            "target_sense": {"type": "string"},
            "dominant_sense_type": {"type": "string"},
            "sensory_mapping_rule": {"type": "string"},
            "step1_spatial_overview": {"type": "string"},
            "step2_sensory_zoom_in": {"type": "string"},
            "tts_script": {"type": "string"},
        },
    },
}

# Fallback if an artist is not in the gold dataset (e.g., a future expansion).
DEFAULT_DOMINANT_SENSE = "spatial"


@dataclass
class CurationResult:
    artwork_id: str
    artist: str
    title_ko: str
    artwork_year: str
    period_style: str
    genre: str
    visual_feature: str
    target_sense: str
    dominant_sense_type: str
    sensory_mapping_rule: str
    step1_spatial_overview: str
    step2_sensory_zoom_in: str
    tts_script: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artwork_id": self.artwork_id,
            "artist": self.artist,
            "title_ko": self.title_ko,
            "artwork_year": self.artwork_year,
            "period_style": self.period_style,
            "genre": self.genre,
            "visual_feature": self.visual_feature,
            "target_sense": self.target_sense,
            "dominant_sense_type": self.dominant_sense_type,
            "sensory_mapping_rule": self.sensory_mapping_rule,
            "step1_spatial_overview": self.step1_spatial_overview,
            "step2_sensory_zoom_in": self.step2_sensory_zoom_in,
            "tts_script": self.tts_script,
        }


def _encode_image_data_url(image_path: Path, max_edge: int = 1024) -> str:
    """Downscale + JPEG-encode the image and return a data URL for the API."""
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


def _load_artist_sense_map(csv_path: Path = GOLD_CSV_PATH) -> dict[str, str]:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    return df.groupby("artist")["dominant_sense_type"].first().to_dict()


class SensoryCurator:
    """Wraps an OpenAI vision call with the v2 system prompt + gold artist map.

    The system prompt (including 3 hand-curated few-shot examples) is loaded
    once at construction time. The artist→dominant_sense map is loaded from
    the gold CSV so we don't have to hardcode it.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt_path: Path = SYSTEM_PROMPT_PATH,
        gold_csv_path: Path = GOLD_CSV_PATH,
    ):
        load_dotenv()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(api_key=api_key)
        self._system_prompt = system_prompt_path.read_text(encoding="utf-8")
        self.artist_sense_map = _load_artist_sense_map(gold_csv_path)

    def dominant_sense_for(self, artist: str) -> str:
        return self.artist_sense_map.get(artist, DEFAULT_DOMINANT_SENSE)

    def curate(
        self,
        image_path: str | Path,
        artwork_id: str,
        artist: str,
        dominant_sense: str | None = None,
        extra_user_hint: str | None = None,
    ) -> CurationResult:
        image_path = Path(image_path)
        data_url = _encode_image_data_url(image_path)
        dominant_sense = dominant_sense or self.dominant_sense_for(artist)

        user_text = (
            f"artwork_id: {artwork_id}\n"
            f"artist: {artist}\n"
            f"dominant_sense_type: {dominant_sense}\n\n"
            "이 그림을 위의 dominant_sense에 따라 2-step 도슨트 스크립트로 옮겨라. "
            "지정된 JSON 스키마 한 객체만 출력하라. "
            "artwork_id, artist, dominant_sense_type 값은 입력 그대로 사용하라."
        )
        if extra_user_hint:
            user_text += f"\n\n추가 지시: {extra_user_hint}"

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_schema", "json_schema": CURATION_JSON_SCHEMA},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
            temperature=0.4,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        # Force the canonical input values in case the model drifted.
        data["artwork_id"] = artwork_id
        data["artist"] = artist
        data["dominant_sense_type"] = dominant_sense

        return CurationResult(raw=data, **{k: data[k] for k in CURATION_JSON_SCHEMA["schema"]["required"]})

    def curate_to_file(
        self,
        image_path: str | Path,
        artwork_id: str,
        artist: str,
        out_dir: Path,
        **kwargs: Any,
    ) -> Path:
        result = self.curate(image_path, artwork_id=artwork_id, artist=artist, **kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{artwork_id}.json"
        out_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path


if __name__ == "__main__":
    # CLI: python -m src.vlm_pipeline <image> <artwork_id> <artist>
    import sys

    if len(sys.argv) != 4:
        print("usage: python -m src.vlm_pipeline <image_path> <artwork_id> <artist>")
        sys.exit(1)
    img, aid, who = sys.argv[1], sys.argv[2], sys.argv[3]
    res = SensoryCurator().curate(img, artwork_id=aid, artist=who)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
