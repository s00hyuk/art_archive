"""환경설정·DB ID·온톨로지 로딩."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# knowledge_pipeline/ 루트 (src/knowledge/config.py 기준 2단계 상위)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def _load_json(name: str) -> dict[str, Any]:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Settings:
    notion_token: str
    openai_api_key: str
    openai_model: str
    confidence_auto_done: float
    worker_id: str
    processing_version: str
    process_sensitive: bool
    db: dict[str, str]  # {"inbox": id, "concepts": id, "entities": id, "claims": id}
    ontology: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(ROOT / ".env")

        databases = _load_json("notion_databases.json")["databases"]
        db = {k: v["id"] for k, v in databases.items()}
        ontology = _load_json("knowledge_ontology.json")

        missing = [
            k for k in ("NOTION_TOKEN", "OPENAI_API_KEY")
            if not (os.getenv(k) or "").strip()
        ]
        if missing:
            raise RuntimeError(
                f"환경변수 누락: {', '.join(missing)}. .env를 확인하세요 (.env.example 참고)."
            )

        return cls(
            notion_token=os.environ["NOTION_TOKEN"].strip(),
            openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip(),
            confidence_auto_done=float(os.getenv("CONFIDENCE_AUTO_DONE", "0.8")),
            worker_id=os.getenv("WORKER_ID", "main-pc").strip(),
            processing_version=os.getenv("PROCESSING_VERSION", "v1").strip(),
            process_sensitive=os.getenv("PROCESS_SENSITIVE", "false").lower() == "true",
            db=db,
            ontology=ontology,
        )

    # 온톨로지에서 자주 쓰는 값들
    @property
    def domains(self) -> list[str]:
        return [t["name"] for t in self.ontology.get("topics", [])]

    @property
    def entity_kinds(self) -> list[str]:
        # Notion Entities.유형 select 값과 일치해야 함
        return self.ontology.get("entity_kinds_notion",
                                 ["인물", "조직", "기술", "제품", "장소", "기타"])
