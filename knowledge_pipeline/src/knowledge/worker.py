"""CLI 진입점.

  # Inbox의 '대기' 자료를 모두 처리 (메인 PC에서 주기 실행)
  python -m knowledge.worker run [--limit 25]

  # Notion 밖에서 단건 투입 후 즉시 처리 (Phase 1)
  python -m knowledge.worker ingest --url https://...
  python -m knowledge.worker ingest --file paper.pdf --access 개인
  python -m knowledge.worker ingest --text "메모 내용" --title "아이디어"
"""
from __future__ import annotations

import argparse
import sys

from . import extract
from .config import Settings
from .notion_store import NotionStore
from .pipeline import process_source

_ICON = {"done": "✅", "review": "🟡", "duplicate": "♻️", "skipped": "⏭️", "failed": "❌"}


def _log(r) -> None:
    print(f"{_ICON.get(r.status, '•')} [{r.status}] {r.title or r.page_id} — {r.detail}")


def cmd_run(s: Settings, args) -> int:
    store = NotionStore(s)
    pending = store.query_pending(limit=args.limit)
    if not pending:
        print("대기 중인 자료가 없습니다.")
        return 0
    print(f"대기 자료 {len(pending)}건 처리 시작 (worker={s.worker_id})")
    counts: dict[str, int] = {}
    for page in pending:
        r = process_source(store, s, page)
        counts[r.status] = counts.get(r.status, 0) + 1
        _log(r)
    print("완료:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def cmd_ingest(s: Settings, args) -> int:
    store = NotionStore(s)

    # 유형·제목 추정을 위해 먼저 가볍게 추출
    if args.url:
        ex = extract.from_url(args.url)
        url = args.url
    elif args.file:
        ex = extract.from_file(args.file)
        url = None
    elif args.text:
        ex = extract.from_text(args.text, title=args.title)
        url = None
    else:
        print("--url / --file / --text 중 하나가 필요합니다.", file=sys.stderr)
        return 2

    title = args.title or ex.title or "(제목 없음)"
    page_id = store.create_inbox_item(
        title=title, url=url, content_type=ex.content_type,
        access=args.access, ai_ok=True,
    )
    print(f"Inbox 생성: {title} ({page_id})")
    page = store.get_page(page_id)
    r = process_source(store, s, page)
    _log(r)
    return 0 if r.status not in {"failed"} else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="knowledge.worker")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Inbox 대기 자료 일괄 처리")
    pr.add_argument("--limit", type=int, default=25)

    pi = sub.add_parser("ingest", help="단건 투입 후 즉시 처리")
    pi.add_argument("--url")
    pi.add_argument("--file")
    pi.add_argument("--text")
    pi.add_argument("--title")
    pi.add_argument("--access", default="일반", choices=["일반", "개인", "민감"])

    args = p.parse_args(argv)
    s = Settings.load()

    if args.cmd == "run":
        return cmd_run(s, args)
    if args.cmd == "ingest":
        return cmd_ingest(s, args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
