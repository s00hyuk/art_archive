"""OpenAI 구조화 추출: 요약·분류·개념·엔터티·주장을 한 번에.

OpenAI Structured Outputs(response_format=json_schema)로 스키마를 강제해
파싱 실패 없이 dict를 돌려받는다.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import Settings

MAX_INPUT_CHARS = 24000  # 토큰 폭주 방지용 컷


def _schema(domains: list[str], entity_kinds: list[str], claim_relations: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "description": "자료의 간결한 한국어 제목"},
            "summary": {"type": "string", "description": "3~5문장 한국어 요약(TL;DR)"},
            "key_points": {
                "type": "array", "items": {"type": "string"},
                "description": "핵심 논지 3~7개",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string", "enum": domains},
                "description": "해당 상위 도메인(복수 가능)",
            },
            "subjects": {
                "type": "array",
                "items": {"type": "string", "enum": ["업무", "개인", "연구"]},
                "description": "용도 축 태그",
            },
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "definition": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "definition", "aliases"],
                },
                "description": "핵심 개념/용어(추상적). 인물·기관 등 고유명사는 entities로.",
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": entity_kinds},
                        "description": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "kind", "description", "aliases"],
                },
                "description": "고유명사(인물/조직/기술/제품/장소 등)",
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "statement": {"type": "string", "description": "자료가 내세우는 핵심 주장 한 문장"},
                        "evidence": {"type": "string", "description": "그 주장의 근거 요약"},
                        "evidence_location": {"type": "string", "description": "근거 위치(섹션/문단/페이지 등, 없으면 빈 문자열)"},
                        "relation": {"type": "string", "enum": claim_relations},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["statement", "evidence", "evidence_location", "relation", "confidence"],
                },
                "description": "검증 가능한 주장 0~5개",
            },
            "confidence": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "전체 분류 신뢰도",
            },
        },
        "required": [
            "title", "summary", "key_points", "domains", "subjects",
            "concepts", "entities", "claims", "confidence",
        ],
    }


def _system_prompt(s: Settings) -> str:
    topics_desc = "\n".join(
        f"- {t['name']}: {t.get('definition', '')}"
        + (f" (하위: {', '.join(t['children'])})" if t.get("children") else "")
        for t in s.ontology.get("topics", [])
    )
    return (
        "당신은 개인 지식DB의 자료 분석가입니다. 주어진 자료를 읽고 구조화된 메타데이터를 추출합니다.\n"
        "규칙:\n"
        "1) 모든 텍스트 값은 한국어로 작성합니다(고유명사는 원어 유지 가능).\n"
        "2) domains는 아래 상위 도메인 중에서만 고릅니다. 딱 맞는 게 없으면 가장 가까운 것을 고르되 confidence를 낮춥니다.\n"
        "3) concepts는 추상 개념/용어, entities는 고유명사(인물·조직·기술·제품·장소)로 분리합니다.\n"
        "4) claims는 자료가 실제로 '주장'하는, 검증 가능한 문장만. 사실 나열은 제외합니다.\n"
        "5) 확실하지 않으면 지어내지 말고 비웁니다.\n\n"
        f"[상위 도메인]\n{topics_desc}\n"
    )


def classify(text: str, title_hint: str | None, url: str | None, s: Settings) -> dict[str, Any]:
    client = OpenAI(api_key=s.openai_api_key)
    schema = _schema(s.domains, s.entity_kinds,
                     s.ontology.get("claim_relations_notion", ["주장", "찬성", "반박", "보완"]))

    user_content = (
        f"[URL] {url or '(없음)'}\n"
        f"[제목 힌트] {title_hint or '(없음)'}\n\n"
        f"[본문]\n{(text or '')[:MAX_INPUT_CHARS]}"
    )

    resp = client.chat.completions.create(
        model=s.openai_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _system_prompt(s)},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "knowledge_extraction", "strict": True, "schema": schema},
        },
    )
    return json.loads(resp.choices[0].message.content)
