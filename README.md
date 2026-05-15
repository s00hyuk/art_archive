# art_archive — 시각장애인을 위한 공감각 미술 큐레이션 PoC

회화 작품의 시각 요소(색·붓터치·구도·빛·움직임)를 **촉각·온도·공간감·청각·운동감**으로
옮겨, 시각장애인 관람객이 작품을 몸으로 이해하도록 돕는 데이터 파이프라인과 데모.

본 리포는 **두 갈래의 PoC**를 함께 담고 있습니다.

| 트랙 | 목적 | 진입점 |
|---|---|---|
| **A. Gradio 데모** (기존) | 3 작가 × 5점 = 15점에 대한 공감각 큐레이션 시연 | `app.py` (`src/data_loader.py`, `src/vlm_pipeline.py`) |
| **B. Baseline vs ABS 실험** (신규, 학술논문용) | 일반 프롬프트와 Art Beyond Sight 가이드라인 프롬프트의 색채-어휘 비율 등 정량 비교 | `python -m src.pipeline` (`src/config.py`, `src/dataset.py`, `src/prompts.py`, `src/curator.py`, `src/tts.py`, `src/evaluator.py`, `src/pipeline.py`) |

데이터셋은 두 트랙 모두 Kaggle의
[Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)을 사용합니다.

---

## 디렉토리 구조

```
art_archive/
├── configs/sensory_ontology.json    # (트랙 A) 시각→비시각 감각 매핑 사전
├── src/
│   ├── data_loader.py               # (트랙 A) 15개 샘플 로드
│   ├── vlm_pipeline.py              # (트랙 A) Gradio용 VLM 호출
│   ├── prompts/system_prompt.md     # (트랙 A) 감각 치환 시스템 프롬프트
│   ├── config.py                    # (트랙 B) 경로·모델·가격·상수
│   ├── dataset.py                   # (트랙 B) 사조 층화 추출 (N=100)
│   ├── prompts.py                   # (트랙 B) Baseline / ABS 프롬프트
│   ├── curator.py                   # (트랙 B) GPT-4o Vision 호출, 캐시·재시도·비용 집계
│   ├── tts.py                       # (트랙 B) OpenAI TTS-1-HD
│   ├── evaluator.py                 # (트랙 B) 색채-어휘 비율 · BLEU · ROUGE-L · BERTScore
│   └── pipeline.py                  # (트랙 B) End-to-end CLI
├── notebooks/analysis.ipynb         # (트랙 B) 시각화 + paired test
├── scripts/
│   ├── download_kaggle.py           # Kaggle 원본 다운로드 (로컬 전용)
│   └── build_samples.py             # (트랙 A) 15개 샘플 추출
├── data/
│   ├── best-artworks/               # (트랙 B) 사용자가 압축 해제
│   ├── raw/                         # (트랙 A 호환) Kaggle 원본 — 동일 위치 자동 인식
│   ├── samples/                     # (트랙 A) 커밋되는 15개 샘플
│   └── ground_truth/                # (트랙 B) `{artwork_id}.txt` 형식의 GT (선택)
├── output/
│   ├── descriptions/{id}__{cond}.json   # (트랙 B) 작품 × 조건별 설명 + 메타
│   ├── audio/{id}_{cond}.mp3            # (트랙 B) TTS 결과
│   ├── eval_results.csv                 # (트랙 B) 평가 지표 표
│   └── sample_index.csv                 # (트랙 B) 이번 런에 사용된 N개 샘플 목록
├── app.py                           # (트랙 A) Gradio PoC
└── requirements.txt
```

---

## 설치

```bash
git clone https://github.com/s00hyuk/art_archive.git
cd art_archive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
```

### Kaggle 인증

API 키는 **절대 채팅이나 코드에 붙여넣지 마세요.** 표준 위치에 두면 됩니다.

1. https://www.kaggle.com → Account → **Create New API Token**
2. ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 트랙 B — Baseline vs ABS 실험 실행

### 1. 데이터셋 배치

`./data/best-artworks/` 경로에 Kaggle 데이터셋을 압축 해제하거나, 기존
`./data/raw/`를 그대로 사용해도 됩니다 (두 경로 모두 자동 인식).

```
data/best-artworks/
├── artists.csv
├── images/images/<Artist_Name>/<Artist_Name>_<N>.jpg
└── resized/  (선택)
```

### 2. 파이프라인 실행

```bash
python -m src.pipeline --n_samples 100 --tts_samples 20
```

옵션:

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--n_samples` | `100` | 사조별 층화 추출 총 표본 크기 |
| `--tts_samples` | `20` | 음성 생성 표본 (0이면 TTS 건너뜀) |
| `--skip_eval` | `False` | 마지막 평가 단계 건너뛰기 |
| `--no_bertscore` | `False` | 느린 BERTScore 단계 건너뛰기 |
| `--seed` | `42` | 샘플링 시드 |

파이프라인은 **재개 가능**합니다. `output/descriptions/{id}__{cond}.json`,
`output/audio/{id}_{cond}.mp3`가 이미 있으면 해당 API 호출은 건너뜁니다.

### 3. 분석 노트북

```bash
jupyter notebook notebooks/analysis.ipynb
```

색채-어휘 비율 boxplot, Wilcoxon signed-rank / paired t 결과, 사조별
breakdown, 정성 예시 5개가 자동 출력됩니다.

### 4. (선택) Ground Truth 추가

BLEU / ROUGE-L / BERTScore를 계산하려면 `data/ground_truth/{artwork_id}.txt`
형태로 정답 텍스트를 두면 됩니다. 파일이 없는 작품은 해당 컬럼이 NaN입니다.

### 예상 비용

GPT-4o pricing 기준 (입력 $2.50 / 출력 $10.00 per 1M tokens),
TTS-1-HD pricing 기준 ($30 / 1M chars). 2025 기준 스냅샷이며 OpenAI 가격
변경 시 `src/config.py`의 상수를 업데이트해야 합니다.

| 항목 | 계산 | 추정 비용 |
|---|---|---|
| 1회 Vision 호출 (이미지 1024px + ABS 프롬프트 ~700 토큰 → 출력 ~350 토큰) | 입력 ~2,000 tok × $2.50/M + 출력 350 tok × $10/M | **≈ $0.0085** |
| 100점 × 2조건 = 200 호출 | 200 × $0.0085 | **≈ $1.7** |
| TTS 1점 (~350자 × 1조건) | 350 × $30/M | **≈ $0.01** |
| 20점 × 2조건 = 40 호출 | 40 × $0.01 | **≈ $0.4** |
| **합계 (100점 + TTS 20점)** | | **≈ $2.1** |

실제 비용은 ABS 프롬프트가 baseline보다 길어 입력 토큰이 ~3 배 더 들기 때문에
출력 조건에 따라 ±30% 정도 변동할 수 있습니다.

---

## 트랙 A — 기존 Gradio 데모 (15점)

```bash
# 1) Kaggle 원본 다운로드 (~2GB, data/raw/만 로컬 보관)
python scripts/download_kaggle.py

# 2) 15개 샘플 추출 (data/samples/로 복사)
python scripts/build_samples.py

# 3) 단일 큐레이션 확인 (선택)
python -m src.vlm_pipeline data/samples/images/<filename>.jpg "<artwork_id>" "Vincent van Gogh"

# 4) Gradio PoC 실행
python app.py
```

`outputs/curations/*.json`은 *AI가 만든 초안*입니다. 데이터셋으로서 가치를
가지려면 (a) 연구자 라벨링, (b) 시각장애인 당사자 검증 두 단계가 필요합니다.

---

## Design Decisions (트랙 B)

구현 중 내린 합리적 디폴트들. 논문 재현 시 참고용.

- **사조 매칭**: Kaggle ``artists.csv``의 ``genre``는 콤마 구분 다중 라벨
  (예: `"Post-Impressionism,Symbolism"`). 목표 사조와 부분 문자열 매칭하되,
  **긴 라벨을 먼저** 검사해 `"Post-Impressionism"`이 `"Impressionism"`으로
  잘못 잡히지 않도록 했습니다.
- **사조당 쿼터**: `ceil(n_samples / |TARGET_GENRES|)`. 사조 내에서는 작가별
  라운드-로빈으로 분배해 한 작가가 사조 표본을 독점하지 못하게 합니다.
- **`title_guess`**: Kaggle 파일명은 `{Artist}_{N}.jpg` 형식뿐이고 실제 제목
  정보가 없습니다. 임시 식별자로만 사용하고 모델 프롬프트에는 핵심 정보로
  전달하지 않습니다.
- **색채 어휘 매칭**: KoNLPy 형태소 분석 없이 substring 매칭. 색 단어 stem
  18종을 ``COLOR_WORDS_KO``에 정리. 활용형(파란/파랗/파르) 변이를 모두 포함해
  recall을 확보하는 것이 절대 정밀도보다 중요하다고 판단했습니다.
- **이미지 인코딩**: 긴 변 1024 px로 리사이즈 후 JPEG 88 quality로 base64.
  OpenAI Vision "high detail" 입력에 충분한 해상도이며 페이로드는 작게 유지.
- **재시도**: rate limit / timeout / transient API 오류에 대해 최대 5회,
  지수 백오프 (2, 4, 8, 16, 32 초). 단일 작품 실패는 로그만 남기고 전체
  파이프라인은 계속 진행합니다.
- **캐싱**: `output/descriptions/{id}__{cond}.json`과 MP3 파일 존재 여부를
  보고 스킵. 중단 후 재실행해도 같은 비용을 두 번 내지 않습니다.
- **TTS 음성**: `nova` 선택 (한국어 자연스러움 기준 내부 비교 결과). 더
  부드러운 톤을 원하면 `src/config.py`의 `TTS_VOICE`를 `shimmer`로 변경.

---

## 보안 체크리스트

- [ ] `.env`가 `.gitignore`에 포함되어 있다 (이 리포는 기본 포함)
- [ ] `kaggle.json`을 코드/리포에 넣지 않았다 (`~/.kaggle/`만 사용)
- [ ] 출력물(`output/`, `outputs/`)을 커밋하기 전, 개인정보나 민감 정보가 모델
  출력에 섞이지 않았는지 확인

## 라이선스

코드: MIT. 데이터셋(Kaggle 원본 이미지)은 원본 라이선스를 따릅니다.
