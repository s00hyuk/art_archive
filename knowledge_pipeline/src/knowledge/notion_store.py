"""Notion 접근 계층 — Personal Knowledge OS 스키마 전용 래퍼.

속성 이름은 실제 Notion DB와 정확히 일치해야 한다(모두 한글).
relation 은 Inbox 쪽에서 한 번에 설정한다(dual relation 이 자식 쪽 '관련 자료/원본 자료'로 자동 동기화).
"""
from __future__ import annotations

from typing import Any

from notion_client import Client

from .config import Settings

# ---- Inbox(자료) 속성명 ----
P_TITLE = "제목"
P_TYPE = "자료 유형"
P_URL = "원문 URL"
P_FILE = "원본 파일"
P_SUMMARY = "요약"
P_STATUS = "상태"
P_ACCESS = "접근 등급"
P_AI_OK = "AI 처리 허용"
P_HASH = "콘텐츠 해시"
P_PVER = "처리 버전"
P_WORKER = "Worker ID"
P_RETRY = "재시도"
P_ERROR = "오류"
P_SUBJECT = "주제"
P_REL_CONCEPTS = "개념"
P_REL_ENTITIES = "엔터티"
P_REL_CLAIMS = "주장"

# 상태 select 값
ST_PENDING, ST_PROCESSING, ST_REVIEW, ST_DONE, ST_FAILED = (
    "대기", "처리중", "검토", "완료", "실패",
)


def _rich(text: str) -> list[dict]:
    """rich_text 값. 2000자 단위로 분할."""
    text = text or ""
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)] or [""]
    return [{"type": "text", "text": {"content": c}} for c in chunks]


def _plain(prop: dict | None) -> str:
    if not prop:
        return ""
    for key in ("rich_text", "title"):
        if prop.get(key):
            return "".join(seg.get("plain_text", "") for seg in prop[key])
    return ""


class NotionStore:
    def __init__(self, s: Settings):
        self.s = s
        self.client = Client(auth=s.notion_token)
        self.db = s.db

    # ---------- 조회 ----------
    def query_pending(self, limit: int = 25) -> list[dict]:
        res = self.client.databases.query(
            database_id=self.db["inbox"],
            filter={"property": P_STATUS, "select": {"equals": ST_PENDING}},
            page_size=limit,
        )
        return res.get("results", [])

    def get_page(self, page_id: str) -> dict:
        return self.client.pages.retrieve(page_id=page_id)

    def read_body_text(self, page_id: str) -> str:
        """페이지 본문 블록에서 텍스트 수집(사용자가 본문에 붙여넣은 경우)."""
        out: list[str] = []
        cursor: str | None = None
        while True:
            resp = self.client.blocks.children.list(
                block_id=page_id, start_cursor=cursor, page_size=100
            )
            for block in resp.get("results", []):
                btype = block.get("type")
                data = block.get(btype, {})
                rich = data.get("rich_text")
                if rich:
                    out.append("".join(seg.get("plain_text", "") for seg in rich))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
        return "\n".join(out).strip()

    def find_by_hash(self, content_hash: str, exclude_id: str | None = None) -> str | None:
        res = self.client.databases.query(
            database_id=self.db["inbox"],
            filter={"property": P_HASH, "rich_text": {"equals": content_hash}},
            page_size=5,
        )
        for page in res.get("results", []):
            if page["id"] != exclude_id:
                return page["id"]
        return None

    # ---------- Inbox 필드 헬퍼 ----------
    @staticmethod
    def field_status(page: dict) -> str:
        sel = page["properties"].get(P_STATUS, {}).get("select")
        return sel["name"] if sel else ""

    @staticmethod
    def field_access(page: dict) -> str:
        sel = page["properties"].get(P_ACCESS, {}).get("select")
        return sel["name"] if sel else ""

    @staticmethod
    def field_ai_ok(page: dict) -> bool:
        return bool(page["properties"].get(P_AI_OK, {}).get("checkbox"))

    @staticmethod
    def field_url(page: dict) -> str | None:
        return page["properties"].get(P_URL, {}).get("url")

    @staticmethod
    def field_files(page: dict) -> list[dict]:
        return page["properties"].get(P_FILE, {}).get("files", [])

    @staticmethod
    def field_retry(page: dict) -> int:
        return int(page["properties"].get(P_RETRY, {}).get("number") or 0)

    @staticmethod
    def field_title(page: dict) -> str:
        return _plain(page["properties"].get(P_TITLE))

    # ---------- 상태 전이 ----------
    def mark_processing(self, page_id: str) -> None:
        self.client.pages.update(page_id=page_id, properties={
            P_STATUS: {"select": {"name": ST_PROCESSING}},
            P_WORKER: {"rich_text": _rich(self.s.worker_id)},
            P_PVER: {"rich_text": _rich(self.s.processing_version)},
        })

    def mark_failed(self, page_id: str, error: str, retry: int) -> None:
        self.client.pages.update(page_id=page_id, properties={
            P_STATUS: {"select": {"name": ST_FAILED}},
            P_ERROR: {"rich_text": _rich(error[:1900])},
            P_RETRY: {"number": retry},
        })

    def mark_duplicate(self, page_id: str, original_id: str) -> None:
        self.client.pages.update(page_id=page_id, properties={
            P_STATUS: {"select": {"name": ST_DONE}},
            P_ERROR: {"rich_text": _rich(f"중복: 원본 {original_id}")},
        })

    # ---------- 개념/엔터티/주장 upsert ----------
    def find_or_create_concept(self, name: str, definition: str,
                               aliases: list[str], domains: list[str]) -> str:
        found = self.client.databases.query(
            database_id=self.db["concepts"],
            filter={"property": "개념명", "title": {"equals": name}},
            page_size=1,
        ).get("results", [])
        if found:
            return found[0]["id"]
        page = self.client.pages.create(
            parent={"database_id": self.db["concepts"]},
            properties={
                "개념명": {"title": _rich(name)},
                "정의": {"rich_text": _rich(definition)},
                "별칭": {"multi_select": [{"name": a[:100]} for a in aliases[:10]]},
                "분야": {"multi_select": [{"name": d} for d in domains]},
                "검토 상태": {"select": {"name": "제안"}},
            },
        )
        return page["id"]

    def find_or_create_entity(self, name: str, kind: str,
                              description: str, aliases: list[str]) -> str:
        found = self.client.databases.query(
            database_id=self.db["entities"],
            filter={"property": "이름", "title": {"equals": name}},
            page_size=1,
        ).get("results", [])
        if found:
            return found[0]["id"]
        page = self.client.pages.create(
            parent={"database_id": self.db["entities"]},
            properties={
                "이름": {"title": _rich(name)},
                "유형": {"select": {"name": kind}},
                "설명": {"rich_text": _rich(description)},
                "별칭": {"multi_select": [{"name": a[:100]} for a in aliases[:10]]},
                "검토 상태": {"select": {"name": "제안"}},
            },
        )
        return page["id"]

    def create_claim(self, statement: str, evidence: str, evidence_location: str,
                     relation: str, confidence: float, source_id: str) -> str:
        page = self.client.pages.create(
            parent={"database_id": self.db["claims"]},
            properties={
                "주장": {"title": _rich(statement)},
                "근거": {"rich_text": _rich(evidence)},
                "근거 위치": {"rich_text": _rich(evidence_location)},
                "관계": {"select": {"name": relation}},
                "검증 상태": {"select": {"name": "미검토"}},
                "AI 신뢰도": {"number": round(float(confidence), 3)},
                "원본 자료": {"relation": [{"id": source_id}]},
            },
        )
        return page["id"]

    # ---------- Inbox 최종 기록 ----------
    def finalize_source(self, page_id: str, *, title: str, content_type: str,
                        summary: str, content_hash: str, subjects: list[str],
                        concept_ids: list[str], entity_ids: list[str],
                        claim_ids: list[str], status: str) -> None:
        props: dict[str, Any] = {
            P_TITLE: {"title": _rich(title)},
            P_TYPE: {"select": {"name": content_type}},
            P_SUMMARY: {"rich_text": _rich(summary)},
            P_HASH: {"rich_text": _rich(content_hash)},
            P_STATUS: {"select": {"name": status}},
            P_REL_CONCEPTS: {"relation": [{"id": i} for i in concept_ids]},
            P_REL_ENTITIES: {"relation": [{"id": i} for i in entity_ids]},
            P_REL_CLAIMS: {"relation": [{"id": i} for i in claim_ids]},
        }
        if subjects:
            props[P_SUBJECT] = {"multi_select": [{"name": x} for x in subjects]}
        self.client.pages.update(page_id=page_id, properties=props)

    def create_inbox_item(self, *, title: str, url: str | None,
                          content_type: str, access: str = "일반",
                          ai_ok: bool = True) -> str:
        """Notion 밖(CLI)에서 자료를 넣을 때 Inbox 행을 새로 생성."""
        props: dict[str, Any] = {
            P_TITLE: {"title": _rich(title)},
            P_TYPE: {"select": {"name": content_type}},
            P_STATUS: {"select": {"name": ST_PENDING}},
            P_ACCESS: {"select": {"name": access}},
            P_AI_OK: {"checkbox": ai_ok},
        }
        if url:
            props[P_URL] = {"url": url}
        page = self.client.pages.create(
            parent={"database_id": self.db["inbox"]}, properties=props
        )
        return page["id"]
