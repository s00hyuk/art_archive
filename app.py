"""Gradio PoC for the sensory curation pipeline (v5 — long-form few-shot).

Two tabs:
  - 샘플 작품: browse the 15 pre-curated artworks (free, cache-only).
  - 직접 업로드: upload your own image and curate it. Subject to a daily
    quota (UPLOAD_DAILY_QUOTA, default 10) that resets at UTC midnight.

Run:
    python app.py                # local, full feature
    READONLY=1 python app.py     # sample-tab generate hidden (e.g. HF Spaces)
    SHARE=1 python app.py        # public gradio.live tunnel
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from gtts import gTTS

from src.data_loader import load_samples
from src.vlm_pipeline import SensoryCurator

# READONLY=1 hides the sample-tab "큐레이션 생성" button so a public demo
# can't accidentally regenerate the curated samples. It does NOT disable
# the upload tab — upload uses its own daily quota.
READONLY = os.environ.get("READONLY", "0") == "1"

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

REPO_ROOT = Path(__file__).resolve().parent
CURATIONS_DIR = REPO_ROOT / "outputs" / "curations"
QUOTA_FILE = REPO_ROOT / "outputs" / ".upload_quota.json"
DAILY_QUOTA = int(os.environ.get("UPLOAD_DAILY_QUOTA", "10"))

SENSE_TYPES = [
    "vibration", "humidity", "spatial", "materiality", "softness",
    "weight", "pressure", "rhythm", "psychological_pressure", "movement",
]

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


# ---------------------------------------------------------------------------
# Audio splicing helpers (intro → 1.5 s silence → body)
# ---------------------------------------------------------------------------


def _silent_mp3(duration_ms: int) -> str:
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


# ---------------------------------------------------------------------------
# Daily upload quota (file-based, UTC midnight reset)
# ---------------------------------------------------------------------------


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_quota_state() -> dict:
    if not QUOTA_FILE.is_file():
        return {"date": _today_utc(), "count": 0}
    try:
        data = json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
        if data.get("date") != _today_utc():
            return {"date": _today_utc(), "count": 0}
        return data
    except Exception:
        return {"date": _today_utc(), "count": 0}


def quota_remaining() -> int:
    state = _read_quota_state()
    return max(0, DAILY_QUOTA - int(state.get("count", 0)))


def consume_quota() -> tuple[bool, int]:
    """Increment usage counter atomically-ish. Returns (allowed, remaining_after)."""
    state = _read_quota_state()
    if state["count"] >= DAILY_QUOTA:
        return False, 0
    state["count"] += 1
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_FILE.write_text(json.dumps(state), encoding="utf-8")
    return True, DAILY_QUOTA - state["count"]


def quota_display_text() -> str:
    remaining = quota_remaining()
    return f"📊 오늘의 잔여 큐레이션 횟수: **{remaining} / {DAILY_QUOTA}** (UTC 자정 기준 초기화)"


# ---------------------------------------------------------------------------
# Curation cache helpers (sample tab)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Output panel — created twice (one per tab) so each tab is self-contained
# ---------------------------------------------------------------------------


def make_output_panel() -> dict:
    """Build the right-column output widgets. Caller wires them up."""
    with gr.Row():
        title_ko = gr.Textbox(label="제목 / 연도", elem_classes=["meta-box"])
        period_style = gr.Textbox(label="사조", elem_classes=["meta-box"])
    with gr.Row():
        genre = gr.Textbox(label="장르", elem_classes=["meta-box"])
        dominant_sense = gr.Textbox(label="Dominant Sense", elem_classes=["meta-box"])
    visual_feature = gr.Textbox(label="시각적 특징", lines=2, elem_classes=["meta-box"])
    step1 = gr.Textbox(label="Step 1 — 공간 개요", lines=3, elem_classes=["step-box"])
    step2 = gr.Textbox(label="Step 2 — 감각 줌인", lines=4, elem_classes=["step-box"])
    tts_script = gr.Textbox(
        label="TTS 도슨트 스크립트 (음성용 통합 본문)",
        lines=7,
        elem_classes=["big-tts"],
    )
    with gr.Row():
        tts_btn = gr.Button("🔊 음성으로 듣기")
        audio = gr.Audio(label="음성", type="filepath")
    return {
        "title_ko": title_ko,
        "period_style": period_style,
        "genre": genre,
        "dominant_sense": dominant_sense,
        "visual_feature": visual_feature,
        "step1": step1,
        "step2": step2,
        "tts_script": tts_script,
        "tts_btn": tts_btn,
        "audio": audio,
    }


def _gtts_to_file(text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    gTTS(text=text, lang="ko").save(tmp.name)
    return tmp.name


def on_tts(text: str):
    """Synthesize Korean TTS with a 1.5 s breath between intro and body.

    Splits at the first transition marker — v5 uses "먼저 화면의", earlier
    versions used "먼저 작품의". We accept either so cached old curations
    still get the pause.
    """
    if not text or not text.strip():
        return None
    idx = -1
    for marker in ("먼저 화면의", "먼저 작품의"):
        i = text.find(marker)
        if i > 0:
            idx = i
            break
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
        return _gtts_to_file(text)


# ---------------------------------------------------------------------------
# App build
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    records = load_samples()
    id_to_record = {f"[{r.artist}] {r.artwork_id}": r for r in records}
    choices = list(id_to_record.keys())

    curator: dict[str, SensoryCurator | None] = {"instance": None}

    def get_curator() -> SensoryCurator:
        if curator["instance"] is None:
            curator["instance"] = SensoryCurator()
        return curator["instance"]

    # Snapshot the known-artist list from the gold dataset. The curator is
    # constructed lazily, so we load the map directly here.
    from src.vlm_pipeline import _load_artist_sense_map  # local import to avoid heavy chain

    artist_sense_map = _load_artist_sense_map()
    known_artists = sorted(artist_sense_map.keys())
    OTHER = "기타 (직접 입력)"
    artist_choices = known_artists + [OTHER]

    # ---- Sample tab handlers ----

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
            print(f"[sample] {err}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return (*EMPTY_OUTPUTS, gr.update(value=err))
        d = result.to_dict()
        _save_cache(rec.artwork_id, d)
        return (*_pack(d), gr.update(value="새 큐레이션을 생성하고 저장했습니다."))

    # ---- Upload tab handlers ----

    def on_artist_change(choice: str):
        is_other = choice == OTHER
        return gr.update(visible=is_other), gr.update(visible=is_other)

    def on_upload(image_path: str | None, artist_choice: str, manual_artist: str, manual_sense: str):
        if not image_path:
            return (
                *EMPTY_OUTPUTS,
                gr.update(value="❗ 이미지를 먼저 업로드해주세요."),
                gr.update(value=quota_display_text()),
            )

        if artist_choice == OTHER:
            artist = (manual_artist or "").strip()
            if not artist:
                return (
                    *EMPTY_OUTPUTS,
                    gr.update(value="❗ 작가명을 입력해주세요."),
                    gr.update(value=quota_display_text()),
                )
            sense = manual_sense or "spatial"
        else:
            artist = artist_choice
            sense = None  # let curator look up from gold map

        # Quota check + consume
        allowed, remaining = consume_quota()
        if not allowed:
            return (
                *EMPTY_OUTPUTS,
                gr.update(
                    value=(
                        f"🚫 오늘의 일일 한도({DAILY_QUOTA}건)에 도달했습니다. "
                        "UTC 자정에 초기화됩니다."
                    )
                ),
                gr.update(value=quota_display_text()),
            )

        artwork_id = "upload_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        try:
            result = get_curator().curate(
                image_path, artwork_id=artwork_id, artist=artist, dominant_sense=sense
            )
        except Exception as e:
            err = f"큐레이션 생성 실패: {type(e).__name__}: {e}"
            # Also dump full traceback to stderr so it shows up in HF Space Logs.
            print(f"[upload] {err}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            # Show key prefix in logs (NOT the UI) for debugging — masked so the
            # value can't be reconstructed.
            k = os.environ.get("OPENAI_API_KEY", "")
            print(
                f"[upload] OPENAI_API_KEY length={len(k)}, "
                f"starts_with={k[:7]!r} (env var presence diagnostic)",
                file=sys.stderr, flush=True,
            )
            return (
                *EMPTY_OUTPUTS,
                gr.update(value=err),
                gr.update(value=quota_display_text()),
            )

        return (
            *_pack(result.to_dict()),
            gr.update(value=f"✅ 큐레이션 생성 완료 (오늘 {remaining}/{DAILY_QUOTA} 남음)"),
            gr.update(value=quota_display_text()),
        )

    # ---- UI ----

    with gr.Blocks(css=CSS, title="공감각 미술 큐레이션 PoC v5") as demo:
        gr.Markdown(
            "# 시각장애인을 위한 공감각 미술 큐레이션 (v5)\n"
            "**도입 → 화면의 큰 배치 (Step 1, 350–500자) → 몸의 감각 (Step 2, 350–500자)** 의 장문 도슨트 (총 800–1,100자). "
            "작가별로 지배 감각(dominant sense)이 정해져 있고, TTS에는 도입부 뒤 1.5초 호흡 포즈가 들어갑니다."
        )

        with gr.Tabs():
            # ============== TAB 1: SAMPLE ARTWORKS ==============
            with gr.Tab("🖼️ 샘플 작품"):
                if READONLY:
                    gr.Markdown(
                        "🔒 **데모 모드** — 사전 생성된 15개 작품 큐레이션을 둘러볼 수 있습니다. "
                        "(샘플 재생성은 비활성화되어 있습니다. 직접 이미지를 큐레이션하려면 ‘직접 업로드’ 탭을 이용해주세요.)"
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
                        out1 = make_output_panel()

                outputs_on_select = [
                    image, out1["title_ko"], out1["period_style"], out1["genre"],
                    out1["dominant_sense"], out1["visual_feature"],
                    out1["step1"], out1["step2"], out1["tts_script"], status,
                ]
                outputs_on_generate = [
                    out1["title_ko"], out1["period_style"], out1["genre"],
                    out1["dominant_sense"], out1["visual_feature"],
                    out1["step1"], out1["step2"], out1["tts_script"], status,
                ]
                selector.change(on_select, inputs=selector, outputs=outputs_on_select)
                generate_btn.click(on_generate, inputs=selector, outputs=outputs_on_generate)
                out1["tts_btn"].click(on_tts, inputs=out1["tts_script"], outputs=out1["audio"])
                demo.load(on_select, inputs=selector, outputs=outputs_on_select)

            # ============== TAB 2: UPLOAD YOUR OWN ==============
            with gr.Tab("📤 직접 업로드"):
                gr.Markdown(
                    "내 작품 이미지를 업로드해서 같은 7블록 도슨트로 큐레이션해볼 수 있어요. "
                    "이 데모는 호스트의 OpenAI API 키로 동작하며, **하루 전체 사용량을 "
                    f"{DAILY_QUOTA}건으로 제한**합니다. 무분별한 사용을 막기 위함입니다."
                )
                quota_status = gr.Markdown(quota_display_text())

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="이미지 업로드", type="filepath", height=420
                        )
                        artist_dd = gr.Dropdown(
                            label="작가 (선택)",
                            choices=artist_choices,
                            value=known_artists[0] if known_artists else OTHER,
                            info="알려진 작가를 고르면 dominant_sense를 자동 적용합니다. 그 외는 ‘기타’.",
                        )
                        manual_artist_tb = gr.Textbox(
                            label="작가명 (직접 입력)",
                            placeholder="예: 김환기",
                            visible=False,
                        )
                        manual_sense_dd = gr.Dropdown(
                            label="Dominant Sense (직접 선택)",
                            choices=SENSE_TYPES,
                            value="spatial",
                            visible=False,
                        )
                        upload_btn = gr.Button("큐레이션 생성", variant="primary")
                        upload_status = gr.Markdown("")
                    with gr.Column(scale=1):
                        out2 = make_output_panel()

                outputs_on_upload = [
                    out2["title_ko"], out2["period_style"], out2["genre"],
                    out2["dominant_sense"], out2["visual_feature"],
                    out2["step1"], out2["step2"], out2["tts_script"],
                    upload_status, quota_status,
                ]
                artist_dd.change(
                    on_artist_change,
                    inputs=artist_dd,
                    outputs=[manual_artist_tb, manual_sense_dd],
                )
                upload_btn.click(
                    on_upload,
                    inputs=[upload_image, artist_dd, manual_artist_tb, manual_sense_dd],
                    outputs=outputs_on_upload,
                )
                out2["tts_btn"].click(on_tts, inputs=out2["tts_script"], outputs=out2["audio"])

    return demo


if __name__ == "__main__":
    # show_api=False avoids gradio_client 1.3's JSON-schema introspection bug
    # with pydantic >=2.10 (additionalProperties is emitted as a bool which
    # the client's `if "const" in schema` chokes on).
    # SHARE=1 opens a 72-hour public gradio.live tunnel.
    share = os.environ.get("SHARE", "0") == "1"
    build_app().launch(show_api=False, share=share)
