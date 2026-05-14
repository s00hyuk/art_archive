"""Gradio PoC for the sensory curation pipeline.

Run:
    python app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
from gtts import gTTS

from src.data_loader import load_samples
from src.vlm_pipeline import SensoryCurator

REPO_ROOT = Path(__file__).resolve().parent
CURATIONS_DIR = REPO_ROOT / "outputs" / "curations"

CSS = """
.big-curation textarea {
    font-size: 22px !important;
    line-height: 1.7 !important;
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
}
.sensory-box textarea {
    font-size: 16px !important;
    line-height: 1.6 !important;
}
"""


def _cached_curation_path(artwork_id: str) -> Path:
    return CURATIONS_DIR / f"{artwork_id}.json"


def _load_cached(artwork_id: str) -> dict | None:
    p = _cached_curation_path(artwork_id)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(artwork_id: str, data: dict) -> None:
    CURATIONS_DIR.mkdir(parents=True, exist_ok=True)
    _cached_curation_path(artwork_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_app() -> gr.Blocks:
    records = load_samples()
    id_to_record = {f"[{r.artist}] {r.artwork_id}": r for r in records}
    choices = list(id_to_record.keys())

    # Curator is constructed lazily so the UI can launch even without an API
    # key — the user only hits the error if they actually click "큐레이션 생성".
    curator: dict[str, SensoryCurator | None] = {"instance": None}

    def get_curator() -> SensoryCurator:
        if curator["instance"] is None:
            curator["instance"] = SensoryCurator()
        return curator["instance"]

    def on_select(label: str):
        rec = id_to_record[label]
        cached = _load_cached(rec.artwork_id)
        if cached:
            return (
                str(rec.image_path),
                ", ".join(cached["visual_features_extracted"]),
                cached["sensory_mapping"]["tactile"],
                cached["sensory_mapping"]["temperature"],
                cached["sensory_mapping"]["spatial"],
                cached["sensory_mapping"]["auditory"],
                cached["sensory_mapping"]["kinesthetic"],
                cached["final_curation_korean"],
                gr.update(value="(저장된 큐레이션을 불러왔습니다)"),
            )
        return (
            str(rec.image_path),
            "", "", "", "", "", "", "",
            gr.update(value="아직 큐레이션이 없습니다. '큐레이션 생성' 버튼을 눌러주세요."),
        )

    def on_generate(label: str):
        rec = id_to_record[label]
        try:
            result = get_curator().curate(
                rec.image_path, artwork_id=rec.artwork_id, artist=rec.artist
            )
        except Exception as e:  # surface API errors to the UI rather than crashing
            err = f"큐레이션 생성 실패: {type(e).__name__}: {e}"
            return ("", "", "", "", "", "", "", gr.update(value=err))

        _save_cache(rec.artwork_id, result.to_dict())
        sm = result.sensory_mapping
        return (
            ", ".join(result.visual_features_extracted),
            sm["tactile"],
            sm["temperature"],
            sm["spatial"],
            sm["auditory"],
            sm["kinesthetic"],
            result.final_curation_korean,
            gr.update(value="새 큐레이션을 생성하고 저장했습니다."),
        )

    def on_tts(text: str):
        if not text or not text.strip():
            return None
        tts = gTTS(text=text, lang="ko")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name

    with gr.Blocks(css=CSS, title="공감각 미술 큐레이션 PoC") as demo:
        gr.Markdown(
            "# 시각장애인을 위한 공감각 미술 큐레이션\n"
            "Van Gogh · Monet · Da Vinci 작품 15점을 비시각 감각언어로 옮겨봅니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                selector = gr.Dropdown(
                    label="작품 선택", choices=choices, value=choices[0]
                )
                image = gr.Image(label="원본 이미지", type="filepath", height=420)
                generate_btn = gr.Button("큐레이션 생성", variant="primary")
                status = gr.Markdown("")

            with gr.Column(scale=1):
                features = gr.Textbox(
                    label="추출된 시각적 특징",
                    lines=2,
                    elem_classes=["sensory-box"],
                )
                with gr.Accordion("감각 치환 (촉각 · 온도 · 공간 · 청각 · 운동)", open=True):
                    tactile = gr.Textbox(label="촉각", lines=2, elem_classes=["sensory-box"])
                    temperature = gr.Textbox(label="온도", lines=2, elem_classes=["sensory-box"])
                    spatial = gr.Textbox(label="공간감", lines=2, elem_classes=["sensory-box"])
                    auditory = gr.Textbox(label="청각", lines=2, elem_classes=["sensory-box"])
                    kinesthetic = gr.Textbox(label="운동감", lines=2, elem_classes=["sensory-box"])

                final = gr.Textbox(
                    label="최종 큐레이션 (도슨트 스크립트)",
                    lines=6,
                    elem_classes=["big-curation"],
                )
                with gr.Row():
                    tts_btn = gr.Button("음성으로 듣기")
                    audio = gr.Audio(label="음성", type="filepath")

        outputs_on_select = [
            image, features, tactile, temperature, spatial,
            auditory, kinesthetic, final, status,
        ]
        outputs_on_generate = [
            features, tactile, temperature, spatial,
            auditory, kinesthetic, final, status,
        ]

        selector.change(on_select, inputs=selector, outputs=outputs_on_select)
        generate_btn.click(on_generate, inputs=selector, outputs=outputs_on_generate)
        tts_btn.click(on_tts, inputs=final, outputs=audio)
        demo.load(on_select, inputs=selector, outputs=outputs_on_select)

    return demo


if __name__ == "__main__":
    build_app().launch()
