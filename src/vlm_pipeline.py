"""VLM curation pipeline (v5 — long-form few-shot).

Given an artwork image and the gold dataset's artist→dominant_sense map,
produce a structured curation. step1_spatial_overview and step2_sensory_zoom_in
are 350-500 chars each; tts_script is 800-1100 chars and follows the format
"[intro] 먼저 화면의 큰 배치를 떠올려보겠습니다. [step1] 이제 이 장면을
몸의 감각으로 가까이 느껴보겠습니다. [step2]".

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
import re
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
    "name": "sensory_curation_v5",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artwork_id",
            "artwork_name",
            "artist",
            "artist_ko",
            "title_en",
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
            "tts_voice_style",
            "difficulty_level",
            "evaluation_group",
            "quality_check_status",
        ],
        "properties": {
            "artwork_id": {"type": "string"},
            "artwork_name": {"type": "string"},
            "artist": {"type": "string"},
            "artist_ko": {"type": "string"},
            "title_en": {"type": "string"},
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
            "tts_voice_style": {"type": "string"},
            "difficulty_level": {"type": "string"},
            "evaluation_group": {"type": "string"},
            "quality_check_status": {"type": "string"},
        },
    },
}

# Fallback if an artist is not in the gold dataset (e.g., a future expansion).
DEFAULT_DOMINANT_SENSE = "spatial"


@dataclass
class CurationResult:
    artwork_id: str
    artwork_name: str
    artist: str
    artist_ko: str
    title_en: str
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
    tts_voice_style: str
    difficulty_level: str
    evaluation_group: str
    quality_check_status: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artwork_id": self.artwork_id,
            "artwork_name": self.artwork_name,
            "artist": self.artist,
            "artist_ko": self.artist_ko,
            "title_en": self.title_en,
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
            "tts_voice_style": self.tts_voice_style,
            "difficulty_level": self.difficulty_level,
            "evaluation_group": self.evaluation_group,
            "quality_check_status": self.quality_check_status,
        }


# Post-processing safety net. The prompt forbids these patterns but Korean
# idioms ("멀리서 보이는", "두드러져 보입니다", "X처럼 보이며") slip through
# occasionally. Replace them with blind-audience-appropriate equivalents.
_VISUAL_REPLACEMENTS: list[tuple[str, str]] = [
    (r"멀리에서 보이는", "멀리 자리한"),
    (r"멀리서 보이는", "멀리 자리한"),
    (r"멀리 보이는", "멀리 자리한"),
    (r"멀리에서 보이며", "멀리 떨어져 있으며"),
    (r"멀리서 보이며", "멀리 떨어져 있으며"),
    (r"멀리 보이며", "멀리 떨어져 있으며"),
    (r"멀리에서 보입니다", "멀리 떨어져 있습니다"),
    (r"멀리서 보입니다", "멀리 떨어져 있습니다"),
    (r"멀리 보입니다", "멀리 떨어져 있습니다"),
    # "반쯤 보이는 X" → "반쯤 드러난 X"
    (r"반쯤 보이는", "반쯤 드러난"),
    (r"어렴풋이 보이는", "어렴풋이 자리한"),
    (r"흐릿하게 보이는", "흐릿하게 자리한"),
    # Object-marker "을/를 보이며" → "을/를 띠며"
    (r"을 보이며", "을 띠며"),
    (r"를 보이며", "를 띠며"),
    (r"을 보이는", "을 띤"),
    (r"를 보이는", "를 띤"),
    (r"두드러져 보입니다", "두드러집니다"),
    (r"두드러져 보이며", "두드러지며"),
    (r"도드라져 보입니다", "도드라집니다"),
    (r"도드라져 보이며", "도드라지며"),
    (r"돋보입니다", "두드러집니다"),
    (r"돋보이며", "두드러지며"),
    (r"눈에 띄는", "두드러지는"),
    (r"눈에 띕니다", "두드러집니다"),
    (r"한눈에", "한 번에"),
    (r"시야에 들어옵니다", "가까이 다가옵니다"),
    (r"시야에 들어오며", "가까이 다가오며"),
    # Bare end-of-sentence "보입니다" → "있습니다"
    (r"보입니다(\s|$|\.)", r"있습니다\1"),
    # "X처럼 보이며" → "X처럼 느껴지며"
    (r"처럼 보이며", "처럼 느껴지며"),
    (r"처럼 보이는", "처럼 느껴지는"),
    (r"처럼 보입니다", "처럼 느껴집니다"),
    # General "보이며" / "보이는" mid-sentence — slightly grammar-aware
    (r"이 보이며", "이 자리하며"),
    (r"가 보이며", "가 자리하며"),
    (r"이 보이는", "이 자리한"),
    (r"가 보이는", "가 자리한"),
    # "그림에서/화면에서 ... 보입니다/보이는"
    (r"화면에 보입니다", "이 작품에 있습니다"),
    (r"그림에서 보입니다", "이 작품에 있습니다"),
    (r"그림을 보는", "이 작품 앞에 서면"),
    (r"보는 이에게", "청자에게"),
    # Bare color noun phrases — anchor with sensation
    (r"푸른 하늘이 드러나", "맑고 차가운 하늘이 트여"),
    (r"푸른 하늘이 펼쳐", "맑고 차가운 하늘이 펼쳐"),
    (r"푸른 하늘이 ", "맑고 차가운 하늘이 "),
    (r"푸른 하늘은", "맑고 차가운 하늘은"),
    (r"회색빛 하늘", "흐리고 가라앉은 하늘"),
    (r"회색빛 구름", "차갑게 내려앉은 구름"),
    (r"녹색 벽이", "서늘한 결의 벽이"),
    (r"녹색 벽은", "서늘한 결의 벽은"),
    (r"붉은 벽이", "따뜻한 결의 벽이"),
    (r"노란 벽이", "따뜻한 결의 벽이"),
]


def _sanitize_visual(text: str) -> str:
    """Replace residual visual-centric phrases that slip past the prompt."""
    for pattern, replacement in _VISUAL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


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
            temperature=0.7,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        # Force the canonical input values in case the model drifted.
        data["artwork_id"] = artwork_id
        data["artist"] = artist
        data["dominant_sense_type"] = dominant_sense

        # Sanitize visual-centric phrasing from the three narrative fields.
        for narrative_field in ("step1_spatial_overview", "step2_sensory_zoom_in", "tts_script"):
            if narrative_field in data and isinstance(data[narrative_field], str):
                data[narrative_field] = _sanitize_visual(data[narrative_field])

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
