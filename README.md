# art_archive — 시각장애인을 위한 공감각 미술 큐레이션 PoC

회화 작품의 시각 요소(색·붓터치·구도·빛·움직임)를 **촉각·온도·공간감·청각·운동감**으로
옮겨, 시각장애인 관람객이 작품을 몸으로 이해하도록 돕는 데이터 파이프라인과 데모.

데이터셋은 Kaggle의 [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)을
사용하며, 화풍이 뚜렷한 세 작가(Van Gogh, Monet, Da Vinci)의 작품 각 5점, 총 15점을 샘플링합니다.

## 디렉토리 구조

```
art_archive/
├── configs/sensory_ontology.json    # 시각→비시각 감각 매핑 사전
├── src/
│   ├── data_loader.py               # 모듈 1: 데이터 로드/샘플링
│   ├── vlm_pipeline.py              # 모듈 3: OpenAI VLM 큐레이션
│   └── prompts/system_prompt.md     # VLM 시스템 프롬프트 (별도 버전관리)
├── scripts/
│   ├── download_kaggle.py           # Kaggle 원본 다운로드 (로컬 전용)
│   └── build_samples.py             # 15개 샘플 추출
├── data/
│   ├── raw/                         # Kaggle 원본 (gitignore)
│   └── samples/                     # 커밋되는 15개 샘플
├── outputs/curations/               # 생성된 큐레이션 JSON (정답셋 후보)
├── app.py                           # 모듈 4: Gradio PoC
└── requirements.txt
```

## 설치

```bash
git clone https://github.com/s00hyuk/art_archive.git
cd art_archive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
```

## Kaggle 인증

API 키는 **절대 채팅이나 코드에 붙여넣지 마세요.** 표준 위치에 두면 됩니다:

1. https://www.kaggle.com → Account → **Create New API Token**
2. ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 실행 순서

```bash
# 1) Kaggle 원본 다운로드 (약 2GB, 로컬 data/raw/에만 저장됨 — git에 올라가지 않음)
python scripts/download_kaggle.py

# 2) 15개 샘플 추출 (data/samples/로 복사, 이게 리포에 커밋됨)
python scripts/build_samples.py

# 3) 큐레이션 1개만 빠르게 확인 (선택)
python -m src.vlm_pipeline data/samples/images/<filename>.jpg "<artwork_id>" "Vincent van Gogh"

# 4) Gradio PoC 실행
python app.py
```

## 데이터셋 정답셋 만들기 (다음 단계)

PoC가 생성한 `outputs/curations/*.json`은 *AI가 만든 초안*입니다. 데이터셋으로서
가치를 가지려면 두 단계 검증이 필요합니다:

1. **연구자 라벨링**: 각 큐레이션에 대해 *선천 전맹* / *중도 실명* 두 그룹을 고려한
   수정안을 작성. 색 단어가 비시각 앵커와 함께 묶여 있는지, 작품에 없는 사물을
   만들어내지 않았는지 점검.
2. **당사자 검증**: 시각장애인 당사자 소수에게 두 버전(원안 vs 수정안)을 들려주고
   어떤 쪽이 작품 이해에 더 도움이 되는지 선호도 수집. 이 선호 라벨이 향후
   파인튜닝/RLHF의 시드.

## 보안 체크리스트

- [ ] `.env`가 `.gitignore`에 포함되어 있다 (이 리포는 기본 포함)
- [ ] `kaggle.json`을 코드/리포에 넣지 않았다 (`~/.kaggle/`만 사용)
- [ ] `outputs/curations/`를 커밋하기 전, 개인정보나 민감 정보가 모델 출력에 섞이지 않았는지 확인

## 라이선스

코드: MIT. 데이터셋(Kaggle 원본 이미지)은 원본 라이선스를 따릅니다.
