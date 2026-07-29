"""한 건의 Inbox 자료를 처리하는 오케스트레이션."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import extract
from .classify import classify
from .config import Settings
from .notion_store import NotionStore, ST_DONE, ST_REVIEW


@dataclass
class Result:
    page_id: str
    status: str          # done / review / duplicate / skipped / failed
    title: str = ""
    detail: str = ""


def _gather_content(store: NotionStore, page: dict) -> extract.Extracted:
    """URL → 파일 → 본문 텍스트 순으로 콘텐츠를 확보."""
    url = store.field_url(page)
    if url:
        return extract.from_url(url)

    files = store.field_files(page)
    if files:
        f = files[0]
        file_url = (f.get("file") or f.get("external") or {}).get("url")
        name = f.get("name", "attachment")
        if not file_url:
            raise ValueError("첨부 파일 URL을 읽을 수 없습니다.")
        import requests

        resp = requests.get(file_url, timeout=60)
        resp.raise_for_status()
        tmp = Path("/tmp") / (extract.content_hash(file_url)[:16] + "_" + name)
        tmp.write_bytes(resp.content)
        return extract.from_file(tmp)

    body = store.read_body_text(page["id"])
    if body:
        return extract.from_text(body, title=store.field_title(page) or None)

    raise ValueError("처리할 콘텐츠가 없습니다(URL/파일/본문 모두 비어 있음).")


def process_source(store: NotionStore, s: Settings, page: dict) -> Result:
    page_id = page["id"]
    title0 = store.field_title(page)

    # 정책 게이트
    access = store.field_access(page) or "일반"
    if not store.field_ai_ok(page):
        return Result(page_id, "skipped", title0, "AI 처리 허용 꺼짐")
    if access == "민감" and not s.process_sensitive:
        return Result(page_id, "skipped", title0, "민감 등급 — LLM 전송 보류")

    store.mark_processing(page_id)
    try:
        ex = _gather_content(store, page)
        chash = extract.content_hash(ex.text)

        dup = store.find_by_hash(chash, exclude_id=page_id)
        if dup:
            store.mark_duplicate(page_id, dup)
            return Result(page_id, "duplicate", ex.title or title0, f"원본 {dup}")

        data = classify(ex.text, ex.title or title0, store.field_url(page), s)

        concept_ids = [
            store.find_or_create_concept(
                c["name"], c.get("definition", ""), c.get("aliases", []),
                data.get("domains", []),
            )
            for c in data.get("concepts", []) if c.get("name")
        ]
        entity_ids = [
            store.find_or_create_entity(
                e["name"], e.get("kind", "기타"), e.get("description", ""),
                e.get("aliases", []),
            )
            for e in data.get("entities", []) if e.get("name")
        ]
        claim_ids = [
            store.create_claim(
                cl["statement"], cl.get("evidence", ""), cl.get("evidence_location", ""),
                cl.get("relation", "주장"), cl.get("confidence", 0.5), page_id,
            )
            for cl in data.get("claims", []) if cl.get("statement")
        ]

        confidence = float(data.get("confidence", 0.0))
        auto = confidence >= s.confidence_auto_done and access != "민감"
        status = ST_DONE if auto else ST_REVIEW

        store.finalize_source(
            page_id,
            title=data.get("title") or ex.title or title0 or "(제목 없음)",
            content_type=ex.content_type,
            summary=data.get("summary", ""),
            content_hash=chash,
            subjects=data.get("subjects", []),
            concept_ids=concept_ids,
            entity_ids=entity_ids,
            claim_ids=claim_ids,
            status=status,
        )
        return Result(
            page_id, "done" if auto else "review",
            data.get("title") or title0,
            f"conf={confidence:.2f} 개념 {len(concept_ids)} 엔터티 {len(entity_ids)} 주장 {len(claim_ids)}",
        )
    except Exception as exc:  # noqa: BLE001 — 워커가 다음 자료로 계속 진행해야 함
        retry = store.field_retry(page) + 1
        store.mark_failed(page_id, f"{type(exc).__name__}: {exc}", retry)
        return Result(page_id, "failed", title0, str(exc))
