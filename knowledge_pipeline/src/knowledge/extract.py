"""자료 유형별 본문 추출: 웹페이지 / PDF / 텍스트.

반환값: {"text": str, "title": str|None, "content_type": str}
content_type 은 Notion Inbox.자료 유형 select 값 중 하나로 맞춘다.
"""
from __future__ import annotations

import hashlib
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

# Notion Inbox.자료 유형 select 값
TYPE_TEXT = "텍스트"
TYPE_WEB = "웹페이지"
TYPE_PAPER = "논문"
TYPE_PDF = "PDF"
TYPE_DOC = "문서"
TYPE_IMAGE = "이미지"


@dataclass
class Extracted:
    text: str
    title: str | None
    content_type: str


def content_hash(text: str) -> str:
    """정규화 후 SHA256 — 공백/개행 차이로 중복을 놓치지 않게."""
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _looks_like_paper(text: str) -> bool:
    head = (text or "")[:3000].lower()
    hits = sum(k in head for k in ("abstract", "초록", "doi", "arxiv", "references", "참고문헌"))
    return hits >= 2


def from_text(text: str, title: str | None = None) -> Extracted:
    return Extracted(text=text.strip(), title=title, content_type=TYPE_TEXT)


def from_pdf(path: str | Path) -> Extracted:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages).strip()
    meta_title = None
    try:
        meta_title = (reader.metadata or {}).get("/Title") or None
    except Exception:
        meta_title = None
    title = meta_title or Path(path).stem
    ctype = TYPE_PAPER if _looks_like_paper(text) else TYPE_PDF
    return Extracted(text=text, title=title, content_type=ctype)


def from_url(url: str) -> Extracted:
    """웹 본문 추출. PDF 링크면 내려받아 PDF 경로로 처리."""
    import requests

    resp = requests.get(url, timeout=30, headers={"User-Agent": "knowledge-pipeline/0.1"})
    resp.raise_for_status()
    ctype_header = resp.headers.get("Content-Type", "")

    if "application/pdf" in ctype_header or url.lower().endswith(".pdf"):
        tmp = Path("/tmp") / (content_hash(url)[:16] + ".pdf")
        tmp.write_bytes(resp.content)
        ex = from_pdf(tmp)
        return ex

    import trafilatura

    downloaded = resp.text
    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True, favor_precision=True
    ) or ""
    title = None
    meta = trafilatura.extract_metadata(downloaded)
    if meta and meta.title:
        title = meta.title
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", downloaded, re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
    ctype = TYPE_PAPER if _looks_like_paper(text) else TYPE_WEB
    return Extracted(text=text.strip(), title=title, content_type=ctype)


def from_file(path: str | Path) -> Extracted:
    """확장자로 분기. PDF/텍스트만 v1 지원; docx/pptx 등은 후속 과제."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return from_pdf(path)
    if ext in {".txt", ".md", ".markdown"}:
        return from_text(path.read_text(encoding="utf-8", errors="replace"), title=path.stem)
    guessed, _ = mimetypes.guess_type(str(path))
    raise NotImplementedError(
        f"아직 지원하지 않는 파일 형식: {ext or guessed}. "
        "v1은 URL/PDF/텍스트만 처리합니다(docx/pptx는 후속)."
    )
