"""Gradio PoC for the sensory curation pipeline (v4 — 7-block naturalized tts).

Run:
    python app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
from gtts import gTTS

from src.data_loader import load_samples
from src.vlm_pipeline import SensoryCurator

# READONLY=1 hides the "큐레이션 생성" button so a public demo (e.g. HF Spaces)
# can only serve the 15 pre-generated curations — no OpenAI API calls from
# anonymous visitors, no credit drain.
READONLY = os.environ.get("READONLY", "0") == "1"

import subprocess

# Use the ffmpeg binary bundled with imageio_ffmpeg so users don't need a
# system install. pydub is avoided because it also requires ffprobe.
try:
    import imageio_ffmpeg

    _FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    _FFMPEG_OK = True
except Exception:  # pragma: no cover
    _FFMPEG_BIN = "ffmpeg"
    _FFMPEG_OK = False

INTRO_PAUSE_MS = 1500  # 1.5 s breathing pause after the docent intro
_SILENCE_CACHE: dict[str, str] = {}


def _silent_mp3(duration_ms: int) -> str:
    """Return the path to a cached silent MP3 of the requested duration."""
    key = str(duration_ms)
    if key in _SILENCE_CACHE:
        return _SILENCE_CACHE[key]
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    subprocess.run(
        [
            _FFMPEG_BIN, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{duration_ms / 1000:.3f}",
            "-q:a", "9", out.name,
        ],
        check=True,
    )
    _SILENCE_CACHE[key] = out.name
    return out.name


def _concat_mp3s(parts: list[str]) -> str:
    """Concat MP3 files with ffmpeg's concat demuxer (no re-encode)."""
    listing = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    for p in parts:
        listing.write(f"file '{p}'\n")
    listing.close()
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    subprocess.run(
        [
            _FFMPEG_BIN, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", listing.name, "-c", "copy", out.name,
        ],
        check=True,
    )
    return out.name

REPO_ROOT = Path(__file__).resolve().parent
CURATIONS_DIR = REPO_ROOT / "outputs" / "curations"

CSS = """
.big-tts textarea {
    font-size: 22px !important;
    line-height: 1.75 !important;
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
}
.step-box textarea {
    font-size: 17px !important;
    line-height: 1.6 !important;
}
.meta-box textarea {
    font-size: 14px !important;
}
"""

EMPTY_OUTPUTS = ("", "", "", "", "", "", "", "")


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


def _pack(d: dict) -> tuple:
    """Pack a curation dict into the UI output tuple order."""
    title = d.get("title_ko", "")
    year = d.get("artwork_year", "")
    if year and year != "연도 미상":
        title_display = f"{title} ({year})" if title else year
    else:
        title_display = title
    return (
        title_display,
        d.get("period_style", ""),
        d.get("genre", ""),
        d.get("dominant_sense_type", ""),
        d.get("visual_feature", ""),
        d.get("step1_spatial_overview", ""),
        d.get("step2_sensory_zoom_in", ""),
        d.get("tts_script", ""),
    )


def build_app() -> gr.Blocks:
    records = load_samples()
    id_to_record = {f"[{r.artist}] {r.artwork_id}": r for r in records}
    choices = list(id_to_record.keys())

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
                *_pack(cached),
                gr.update(value="(저장된 큐레이션을 불러왔습니다)"),
            )
        return (
            str(rec.image_path),
            *EMPTY_OUTPUTS,
            gr.update(value="아직 큐레이션이 없습니다. '큐레이션 생성' 버튼을 눌러주세요."),
        )

    def on_generate(label: str):
        if READONLY:
            return (
                *EMPTY_OUTPUTS,
                gr.update(value="🔒 데모 모드에서는 새 큐레이션 생성이 차단되어 있습니다."),
            )
        rec = id_to_record[label]
        try:
            result = get_curator().curate(
                rec.image_path, artwork_id=rec.artwork_id, artist=rec.artist
            )
        except Exception as e:
            err = f"큐레이션 생성 실패: {type(e).__name__}: {e}"
            return (*EMPTY_OUTPUTS, gr.update(value=err))

        d = result.to_dict()
        _save_cache(rec.artwork_id, d)
        return (*_pack(d), gr.update(value="새 큐레이션을 생성하고 저장했습니다."))

    def _gtts_to_file(text: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        gTTS(text=text, lang="ko").save(tmp.name)
        return tmp.name

    def on_tts(text: str):
        """Synthesize Korean TTS, inserting a 1.5s breathing pause between the
        artwork-intro sentence and the spatial overview so listeners can absorb
        the title before details begin.
        """
        if not text or not text.strip():
            return None

        # Find the boundary: intro ends right before "먼저 작품의" (the fixed
        # composition prefix). Fall back to single-pass TTS if not found.
        marker = "먼저 작품의"
        idx = text.find(marker)
        if idx <= 0 or not _FFMPEG_OK:
            return _gtts_to_file(text)

        intro_text = text[:idx].strip()
        body_text = text[idx:].strip()
        if not intro_text or not body_text:
            return _gtts_to_file(text)

        try:
            intro_path = _gtts_to_file(intro_text)
            body_path = _gtts_to_file(body_text)
            silence_path = _silent_mp3(INTRO_PAUSE_MS)
            return _concat_mp3s([intro_path, silence_path, body_path])
        except subprocess.CalledProcessError:
            # ffmpeg unhappy; fall back to single-pass TTS rather than erroring out.
            return _gtts_to_file(text)

    with gr.Blocks(css=CSS, title="공감각 미술 큐레이션 PoC v4") as demo:
        gr.Markdown(
            "# 시각장애인을 위한 공감각 미술 큐레이션 (v4)\n"
            "**도입 → 구도 → Step 1 공간 개요 → 자연 스캔 → Step 2 감각 줌인 → 자연 마무리** 의 7블록 자연화 도슨트. "
            "작가별로 지배 감각(dominant sense)이 정해져 있고, TTS에는 도입부 뒤 1.5초 호흡 포즈가 들어갑니다."
        )
        if READONLY:
            gr.Markdown(
                "🔒 **데모 모드** — 사전 생성된 15개 작품 큐레이션을 둘러볼 수 있습니다. "
                "(새로운 큐레이션 생성은 비활성화되어 있습니다.)"
            )

        with gr.Row():
            with gr.Column(scale=1):
                selector = gr.Dropdown(
                    label="작품 선택", choices=choices, value=choices[0]
                )
                image = gr.Image(label="원본 이미지", type="filepath", height=420)
                generate_btn = gr.Button(
                    "큐레이션 생성", variant="primary", visible=not READONLY
                )
                status = gr.Markdown("")

            with gr.Column(scale=1):
                with gr.Row():
                    title_ko = gr.Textbox(label="제목 / 연도", elem_classes=["meta-box"])
                    period_style = gr.Textbox(label="사조", elem_classes=["meta-box"])
                with gr.Row():
                    genre = gr.Textbox(label="장르", elem_classes=["meta-box"])
                    dominant_sense = gr.Textbox(
                        label="Dominant Sense", elem_classes=["meta-box"]
                    )
                visual_feature = gr.Textbox(
                    label="시각적 특징", lines=2, elem_classes=["meta-box"]
                )

                step1 = gr.Textbox(
                    label="Step 1 — 공간 개요",
                    lines=3,
                    elem_classes=["step-box"],
                )
                step2 = gr.Textbox(
                    label="Step 2 — 감각 줌인",
                    lines=4,
                    elem_classes=["step-box"],
                )
                tts_script = gr.Textbox(
                    label="TTS 도슨트 스크립트 (음성용 통합 본문)",
                    lines=7,
                    elem_classes=["big-tts"],
                )
                with gr.Row():
                    tts_btn = gr.Button("🔊 음성으로 듣기")
                    audio = gr.Audio(label="음성", type="filepath")

        outputs_on_select = [
            image,
            title_ko,
            period_style,
            genre,
            dominant_sense,
            visual_feature,
            step1,
            step2,
            tts_script,
            status,
        ]
        outputs_on_generate = [
            title_ko,
            period_style,
            genre,
            dominant_sense,
            visual_feature,
            step1,
            step2,
            tts_script,
            status,
        ]

        selector.change(on_select, inputs=selector, outputs=outputs_on_select)
        generate_btn.click(on_generate, inputs=selector, outputs=outputs_on_generate)
        tts_btn.click(on_tts, inputs=tts_script, outputs=audio)
        demo.load(on_select, inputs=selector, outputs=outputs_on_select)

    return demo


if __name__ == "__main__":
    import os

    # show_api=False avoids gradio_client 1.3's JSON-schema introspection bug
    # with pydantic >=2.10 (additionalProperties is emitted as a bool which
    # the client's `if "const" in schema` chokes on).
    # SHARE=1 opens a 72-hour public gradio.live tunnel.
    share = os.environ.get("SHARE", "0") == "1"
    build_app().launch(show_api=False, share=share)
