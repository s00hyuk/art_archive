"""VLM curation pipeline.

Given an artwork image + the sensory ontology, ask an OpenAI vision model to
produce a JSON curation tailored for blind / low-vision audiences.

The system prompt lives at src/prompts/system_prompt.md so it can be edited
without touching code. The ontology lives at configs/sensory_ontology.json.

Usage:
    from src.vlm_pipeline import SensoryCurator
    curator = SensoryCurator()
    result = curator.curate(image_path, artwork_id="...", artist="...")
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = REPO_ROOT / "src" / "prompts" / "system_prompt.md"
ONTOLOGY_PATH = REPO_ROOT / "configs" / "sensory_ontology.json"

# JSON Schema enforced on the model's output. Mirrors the contract in the prompt.
CURATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "sensory_curation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artwork_id",
            "artist",
            "visual_features_extracted",
            "sensory_mapping",
            "final_curation_korean",
        ],
        "properties": {
            "artwork_id": {"type": "string"},
            "artist": {"type": "string"},
            "visual_features_extracted": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
            },
            "sensory_mapping": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tactile", "temperature", "spatial", "auditory", "kinesthetic"],
                "properties": {
                    "tactile": {"type": "string"},
                    "temperature": {"type": "string"},
                    "spatial": {"type": "string"},
                    "auditory": {"type": "string"},
                    "kinesthetic": {"type": "string"},
                },
            },
            "final_curation_korean": {"type": "string"},
        },
    },
}


@dataclass
class CurationResult:
    artwork_id: str
    artist: str
    visual_features_extracted: list[str]
    sensory_mapping: dict[str, str]
    final_curation_korean: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artwork_id": self.artwork_id,
            "artist": self.artist,
            "visual_features_extracted": self.visual_features_extracted,
            "sensory_mapping": self.sensory_mapping,
            "final_curation_korean": self.final_curation_korean,
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


class SensoryCurator:
    """Wraps an OpenAI vision call with the system prompt + ontology baked in.

    The system prompt and ontology are loaded once at construction time. They
    are reused on every call, which means OpenAI's automatic prompt caching
    kicks in for the static prefix.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt_path: Path = SYSTEM_PROMPT_PATH,
        ontology_path: Path = ONTOLOGY_PATH,
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
        self._ontology_text = ontology_path.read_text(encoding="utf-8")

    def _build_system_message(self) -> str:
        return (
            f"{self._system_prompt}\n\n"
            "---\n\n"
            "다음은 번역에 참고할 `sensory_ontology` 사전이다. "
            "그대로 베끼지 말고, 작품의 구체적 디테일에 맞게 변주해 활용하라.\n\n"
            "```json\n"
            f"{self._ontology_text}\n"
            "```\n"
        )

    def curate(
        self,
        image_path: str | Path,
        artwork_id: str,
        artist: str,
        extra_user_hint: str | None = None,
    ) -> CurationResult:
        image_path = Path(image_path)
        data_url = _encode_image_data_url(image_path)

        user_text = (
            f"artwork_id: {artwork_id}\n"
            f"artist: {artist}\n\n"
            "이 그림을 시각장애인 관람객에게 들려줄 공감각 큐레이션으로 옮겨라. "
            "출력은 지정된 JSON 스키마 한 객체만."
        )
        if extra_user_hint:
            user_text += f"\n\n추가 지시: {extra_user_hint}"

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_schema", "json_schema": CURATION_JSON_SCHEMA},
            messages=[
                {"role": "system", "content": self._build_system_message()},
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

        # The model may echo artwork_id/artist; force the canonical values.
        data["artwork_id"] = artwork_id
        data["artist"] = artist

        return CurationResult(
            artwork_id=data["artwork_id"],
            artist=data["artist"],
            visual_features_extracted=data["visual_features_extracted"],
            sensory_mapping=data["sensory_mapping"],
            final_curation_korean=data["final_curation_korean"],
            raw=data,
        )

    def curate_to_file(
        self,
        image_path: str | Path,
        artwork_id: str,
        artist: str,
        out_dir: Path,
    ) -> Path:
        result = self.curate(image_path, artwork_id=artwork_id, artist=artist)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{artwork_id}.json"
        out_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path


if __name__ == "__main__":
    # Tiny CLI for ad-hoc test:
    #   python -m src.vlm_pipeline path/to/image.jpg ARTWORK_ID "Artist Name"
    import sys

    if len(sys.argv) != 4:
        print("usage: python -m src.vlm_pipeline <image_path> <artwork_id> <artist>")
        sys.exit(1)
    img, aid, who = sys.argv[1], sys.argv[2], sys.argv[3]
    res = SensoryCurator().curate(img, artwork_id=aid, artist=who)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
