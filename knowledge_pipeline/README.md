# Knowledge Pipeline — Notion 지식DB 자동 구축

붙여넣은 자료(웹페이지 · 논문 · PDF · 파일 · 텍스트)를 자동으로 추출·요약·분류하고,
Notion의 **🧠 Personal Knowledge OS**(Inbox · Concepts · Entities · Claims)에 기록해
개인 지식 그래프를 키우는 파이프라인.

> ℹ️ 이 폴더는 최종적으로 `s00hyuk/Knowledge_DB_Repository`로 옮겨질 예정입니다.
> 현재는 쓰기 권한이 있는 `sense-docent` 브랜치에 임시 보관 중입니다(작업 보존용).
> 옮길 때는 `knowledge_pipeline/`의 내용물을 새 리포 루트로 복사하면 됩니다.

## 동작 개요

```
[모든 PC/모바일]  Notion Inbox에 URL·텍스트·파일 붙여넣기 (상태=대기)
        │
        ▼
[메인 PC 워커]   대기 자료 폴링 → 본문 추출 → 중복 체크(콘텐츠 해시)
        │        → LLM 구조화 추출(요약·도메인·개념·엔터티·주장)
        ▼        → Concepts/Entities/Claims upsert + Inbox에 relation 연결
[Notion 지식DB]  신뢰도 높으면 자동 '완료', 낮으면 '검토'로 남김
```

## 타깃 Notion 스키마 (이미 구축됨)

| DB | 핵심 속성 |
|---|---|
| **Knowledge Inbox** | 제목·자료 유형·원문 URL·원본 파일·요약·상태·접근 등급·AI 처리 허용·콘텐츠 해시·처리 버전·Worker ID·재시도·오류·주제 + relation(개념/엔터티/주장) |
| **Concepts** | 개념명·정의·별칭·분야·검토 상태 |
| **Entities** | 이름·유형(인물/조직/기술/제품/장소/기타)·설명·별칭·검토 상태 |
| **Claims** | 주장·근거·근거 위치·관계(주장/찬성/반박/보완)·검증 상태·AI 신뢰도 |

DB ID는 `configs/notion_databases.json`에 있습니다.

## 설치 (메인 PC)

```bash
pip install -r requirements.txt

# Notion 통합(integration) 생성: https://www.notion.so/my-integrations
#   → 생성한 integration을 위 4개 DB에 각각 'Connections'로 공유
cp .env.example .env      # NOTION_TOKEN, OPENAI_API_KEY 채우기
```

## 사용

```bash
export PYTHONPATH=src

# 1) Notion Inbox의 '대기' 자료를 일괄 처리 (메인 PC에서 주기적으로 실행)
python -m knowledge.worker run --limit 25

# 2) Notion 밖에서 단건 투입 후 즉시 처리
python -m knowledge.worker ingest --url https://example.com/article
python -m knowledge.worker ingest --file paper.pdf --access 개인
python -m knowledge.worker ingest --text "떠오른 아이디어" --title "메모"
```

주기 실행 예 (cron, 10분마다):

```
*/10 * * * * cd /path/to/repo && PYTHONPATH=src python -m knowledge.worker run >> worker.log 2>&1
```

## 처리 규칙 (설정 가능)

- **자동 완료 임계값**: `CONFIDENCE_AUTO_DONE`(기본 0.8) 이상이면 사람 검토 없이 `완료`, 미만이면 `검토`.
- **중복 방지**: 정규화 본문의 SHA256(`콘텐츠 해시`). 이미 있으면 새로 분석하지 않고 `완료`(중복 표시).
- **신규 개념/엔터티**: 항상 `검토 상태=제안`으로 생성 → 사람이 승격/병합.
- **보안**: `AI 처리 허용`이 꺼졌거나 `접근 등급=민감`(+`PROCESS_SENSITIVE=false`)이면 LLM 전송 보류.

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `config.py` | .env·DB ID·온톨로지 로딩 |
| `extract.py` | 웹/PDF/텍스트 본문 추출 + 콘텐츠 해시 |
| `classify.py` | OpenAI Structured Outputs로 요약·분류·개념·엔터티·주장 추출 |
| `notion_store.py` | Notion 스키마 전용 읽기/쓰기 래퍼 |
| `pipeline.py` | 단건 처리 오케스트레이션(게이트→추출→분류→기록) |
| `worker.py` | CLI (`run` / `ingest`) |

## 현재 상태 / 후속 과제

- v1 지원 입력: **URL · PDF · 텍스트(본문/속성)**. `docx/pptx/xlsx`는 후속(기존 sense-docent의 문서 처리 재사용 가능).
- 원본 파일 첨부: 사용자가 Inbox에 직접 첨부하면 워커가 그 URL로 내려받아 처리. 프로그램이 새로 업로드하는 건 Notion File Upload API 연동으로 후속.
- 엔터티/개념 해소: v1은 제목 완전일치. 별칭(`별칭`) 기반 매칭·병합은 후속.
- 활용 도구(주제별 dossier 생성, 개념 공동출현 인사이트 잡)는 Phase 3.
