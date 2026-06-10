import argparse
import os
import re
import subprocess
import sys
import threading
import uuid
import warnings
from pathlib import Path

# ── Stderr filter: suppress direct print()-to-stderr spam from libraries ──────
# (warnings.filterwarnings only covers Python's warnings module,
#  not packages that call print(..., file=sys.stderr) directly.)

_STDERR_SUPPRESS = (
    "triton",
    "Triton",
    "A matching Triton",
    "xFormers can't load",
    "XFORMERS",
    "Memory-efficient attention, SwiGLU",
    "Please reinstall xformers",
    "Set XFORMERS_MORE_DETAILS",
)

class _FilteredStderr:
    """Pass stderr through, except for known noisy library lines."""
    def __init__(self, real):
        self._real = real

    def write(self, text):
        if any(pat in text for pat in _STDERR_SUPPRESS):
            return len(text)
        return self._real.write(text)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    # Proxy every other attribute so nothing downstream breaks
    def __getattr__(self, name):
        return getattr(self._real, name)

sys.stderr = _FilteredStderr(sys.stderr)

# ── Python warnings module suppressions ───────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*xFormers.*")
warnings.filterwarnings("ignore", message=".*triton.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")

# Load .env before any HuggingFace/torch imports
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)

import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from script_gen import (generate_script, generate_script_openai_compat,
                         check_ollama, OPENAI_PROVIDERS)
import tts_models

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Per-topic folder helpers ──────────────────────────────────────────────────

_file_registry: dict[str, str] = {}   # filename → absolute path


def _sanitize_topic(topic: str) -> str:
    """Turn a topic string into a safe folder name."""
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', (topic or "untitled").strip())
    safe = re.sub(r'_+', '_', safe).strip('_ ')
    return safe[:60] or "untitled"


def _topic_dir(topic: str, subdir: str) -> Path:
    d = OUTPUT_DIR / _sanitize_topic(topic) / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _register(path: str) -> str:
    """Register a generated file path; return just its filename."""
    fname = Path(path).name
    _file_registry[fname] = path
    return fname


def _resolve(filename: str, *fallback_dirs: Path) -> Path | None:
    """Locate a file: registry → fallback dirs → full rglob search."""
    fname = Path(filename).name
    reg = _file_registry.get(fname)
    if reg and Path(reg).exists():
        return Path(reg)
    for d in fallback_dirs:
        p = d / fname
        if p.exists():
            return p
    # Last resort — walk the whole output tree
    for p in OUTPUT_DIR.rglob(fname):
        return p
    return None


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/device-info")
def device_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return jsonify({"device": "cuda", "name": name, "vram_mb": vram})
    return jsonify({"device": "cpu", "name": "CPU", "vram_mb": 0})


@app.route("/list-models", methods=["POST"])
def list_models():
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key required"}), 400
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=api_key)
        all_models = list(client.models.list())
        text_models = []
        for m in all_models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                model_id = m.name.replace("models/", "") if m.name.startswith("models/") else m.name
                # Skip experimental/preview noise — keep flash/lite/pro stable ones
                if any(x in model_id for x in ["flash", "lite", "pro"]) and "exp" not in model_id:
                    text_models.append({
                        "id": model_id,
                        "name": getattr(m, "display_name", model_id)
                    })
        # Sort: flash first, then lite, then pro
        text_models.sort(key=lambda x: (
            0 if "flash" in x["id"] and "lite" not in x["id"] else
            1 if "lite" in x["id"] else
            2
        ))
        return jsonify({"models": text_models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/script-providers")
def script_providers():
    """Return available script generation providers including Ollama status."""
    providers = [{"id": "gemini", "name": "Gemini (Google)", "type": "cloud",
                  "key_hint": "Enter Gemini API key in the sidebar"}]
    for pid, info in OPENAI_PROVIDERS.items():
        entry = {"id": pid, "name": info["name"], "type": "openai_compat",
                 "base_url": info["base_url"], "models": info["models"],
                 "key_hint": info["key_hint"]}
        if pid == "ollama":
            entry["ollama_models"] = check_ollama()
            entry["available"] = len(entry["ollama_models"]) > 0
        providers.append(entry)
    return jsonify({"providers": providers})


@app.route("/generate-script", methods=["POST"])
def gen_script():
    data = request.get_json()
    topic = data.get("topic", "").strip()
    provider = data.get("provider", "gemini")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        # OpenAI-compatible providers (Qwen, Ollama, LM Studio)
        if provider in OPENAI_PROVIDERS:
            script = generate_script_openai_compat(
                topic=topic,
                niche=data.get("niche", "educational"),
                duration=data.get("duration", "5-7 minutes"),
                tone=data.get("tone", "engaging and conversational"),
                context=data.get("context", ""),
                api_key=data.get("api_key", ""),
                model=data.get("model", OPENAI_PROVIDERS[provider]["models"][0]),
                base_url=OPENAI_PROVIDERS[provider]["base_url"],
            )
            return jsonify({"script": script})

        # Default: Gemini
        api_key = data.get("api_key", "").strip()
        if not api_key:
            return jsonify({"error": "Gemini API key is required"}), 400

        script = generate_script(
            topic=topic,
            niche=data.get("niche", "educational"),
            duration=data.get("duration", "5-7 minutes"),
            tone=data.get("tone", "engaging and conversational"),
            context=data.get("context", ""),
            api_key=api_key,
            model=data.get("model", "gemini-2.0-flash"),
        )
        return jsonify({"script": script})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Async audio job tracker ────────────────────────────────────
_jobs: dict = {}


def _run_audio_job(job_id: str, text: str, output_path: str, filename: str,
                   model_id: str = "chatterbox", voice: str = "default",
                   exaggeration: float = 0.5, cfg_weight: float = 0.5,
                   voice_ref: str | None = None, chunk_mode: str = "paragraph",
                   speed: float = 1.0, gemini_api_key: str = ""):
    def progress_cb(msg: str, chunk: int, total: int):
        _jobs[job_id].update({"msg": msg, "chunk": chunk, "total": total})

    _jobs[job_id] = {"status": "running", "msg": "Loading model…", "chunk": 0, "total": 0}
    try:
        tts_models.generate(
            model_id=model_id,
            text=text,
            output_path=output_path,
            voice=voice,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            voice_ref=voice_ref,
            chunk_mode=chunk_mode,
            speed=speed,
            gemini_api_key=gemini_api_key,
            progress_cb=progress_cb,
        )
        _jobs[job_id] = {"status": "done", "filename": filename, "msg": "Complete", "chunk": 1, "total": 1}
    except Exception as e:
        import traceback; traceback.print_exc()
        _jobs[job_id] = {"status": "error", "error": str(e), "msg": str(e), "chunk": 0, "total": 0}


@app.route("/tts-models")
def get_tts_models():
    result = []
    for mid, info in tts_models.REGISTRY.items():
        installed = tts_models.is_installed(mid)
        result.append({
            "id": mid,
            "name": info["name"],
            "author": info["author"],
            "vram": info["vram"],
            "description": info["description"],
            "supports_clone": info["supports_clone"],
            "voices": info["voices"],
            "installed": installed,
        })
    return jsonify({"models": result})


@app.route("/generate-audio", methods=["POST"])
def gen_audio():
    data = request.get_json()
    text = data.get("text", "").strip()
    topic = data.get("topic", "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    job_id = uuid.uuid4().hex[:8]
    filename = f"voiceover_{job_id}.wav"
    out_dir = _topic_dir(topic, "voiceovers") if topic else OUTPUT_DIR
    output_path = str(out_dir / filename)
    _register(output_path)

    thread = threading.Thread(
        target=_run_audio_job,
        args=(job_id, text, output_path, filename),
        kwargs={
            "model_id":       data.get("tts_model", "chatterbox"),
            "voice":          data.get("voice", "default"),
            "exaggeration":   float(data.get("exaggeration", 0.5)),
            "cfg_weight":     float(data.get("cfg_weight", 0.5)),
            "voice_ref":      data.get("voice_ref") or None,
            "chunk_mode":     data.get("chunk_size", "paragraph"),
            "speed":          float(data.get("speed", 1.0)),
            "gemini_api_key": data.get("gemini_api_key", ""),
        },
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/audio-status/<job_id>")
def audio_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/audio/<filename>")
def serve_audio(filename):
    p = _resolve(filename, OUTPUT_DIR)
    if not p:
        return jsonify({"error": "File not found"}), 404
    # Detect mime from extension so MP3s stream correctly too
    ext = Path(p).suffix.lower()
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
            "ogg": "audio/ogg", "m4a": "audio/mp4"}.get(ext.lstrip("."), "audio/wav")
    return send_file(str(p), mimetype=mime)


@app.route("/load-audio", methods=["POST"])
def load_audio():
    """Copy an external audio file into the topic folder and register it."""
    import shutil
    data = request.get_json()
    src = data.get("file_path", "").strip()
    topic = data.get("topic", "").strip()

    if not src or not Path(src).exists():
        return jsonify({"error": "File not found"}), 400

    ext = Path(src).suffix.lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    job_id = uuid.uuid4().hex[:8]
    filename = f"voiceover_{job_id}{ext}"
    out_dir = _topic_dir(topic, "voiceovers") if topic else OUTPUT_DIR
    dest = str(out_dir / filename)
    shutil.copy2(src, dest)
    _register(dest)
    return jsonify({"filename": filename, "path": dest})


@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio_route():
    """Transcribe an audio file; return timed segments + joined transcript text."""
    data = request.get_json()
    audio_file = data.get("audio_filename", "")
    model_size  = data.get("model_size", "tiny")

    audio_p = _resolve(audio_file, OUTPUT_DIR)
    if not audio_p:
        return jsonify({"error": "Audio file not found"}), 404

    try:
        segments = sg.transcribe(str(audio_p), model_size=model_size)
        transcript = " ".join(s["text"] for s in segments)
        return jsonify({"segments": segments, "transcript": transcript})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/pick-file", methods=["POST"])
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        data = request.get_json() or {}
        mode = data.get("filetypes", "voice")  # "voice" | "audio" | "video"

        if mode == "audio":
            title = "Select audio file"
            filetypes = [("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                         ("WAV", "*.wav"), ("MP3", "*.mp3"),
                         ("All files", "*.*")]
        elif mode == "video":
            title = "Select video file"
            filetypes = [("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                         ("All files", "*.*")]
        else:
            title = "Select voice reference audio"
            filetypes = [("WAV files", "*.wav"), ("All audio", "*.mp3 *.wav *.flac")]

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return jsonify({"path": path or ""})
    except Exception as e:
        return jsonify({"path": "", "error": str(e)})


# ── Video endpoints ────────────────────────────────────────────────────────────
import video_engine as ve
import video_gen as vg
import scene_gen as sg

@app.route("/pexels-search", methods=["POST"])
def pexels_search():
    data = request.get_json()
    try:
        results = ve.search_pexels_videos(
            query=data.get("query", ""),
            api_key=data.get("pexels_key", ""),
            per_page=data.get("per_page", 6),
        )
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-clip", methods=["POST"])
def download_clip():
    data = request.get_json()
    try:
        path = ve.download_clip(data["url"], data.get("id", uuid.uuid4().hex[:8]))
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate-slide", methods=["POST"])
def generate_slide():
    data = request.get_json()
    try:
        path = ve.generate_slide(
            text=data.get("text", ""),
            duration=float(data.get("duration", 5)),
            style=data.get("style", "dark"),
            subtitle=data.get("subtitle", ""),
        )
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/slide-preview/<path:filename>")
def slide_preview(filename):
    p = ve.CLIP_CACHE / filename
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="image/png")


@app.route("/assemble-video", methods=["POST"])
def assemble_video_route():
    data = request.get_json()
    topic = data.get("topic", "").strip()
    job_id = uuid.uuid4().hex[:8]
    audio_file = data.get("audio_filename", "")

    audio_p = _resolve(audio_file, OUTPUT_DIR) if audio_file else None
    if not audio_p:
        return jsonify({"error": "No voiceover found. Generate a voiceover first."}), 400

    out_dir = _topic_dir(topic, "videos") if topic else ve.VIDEO_OUTPUT
    out_path = str(out_dir / f"video_{job_id}.mp4")
    _register(out_path)

    ve.start_assemble(
        job_id=job_id,
        sequence=data.get("sequence", []),
        audio_path=str(audio_p),
        output_path=out_path,
        burn_captions=data.get("burn_captions", False),
        resolution=tuple(data.get("resolution", [1920, 1080])),
        fps=int(data.get("fps", 30)),
    )
    return jsonify({"job_id": job_id})


@app.route("/video-status/<job_id>")
def video_status(job_id):
    job = ve._video_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/video/<filename>")
def serve_video(filename):
    p = _resolve(filename, ve.VIDEO_OUTPUT)
    if not p:
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="video/mp4")


@app.route("/generate-thumbnail", methods=["POST"])
def gen_thumbnail():
    import shutil as _shutil
    data = request.get_json()
    topic = data.get("topic", "").strip()
    try:
        if data.get("openai_key"):
            path = ve.generate_thumbnail_dalle(
                title=data.get("title", ""),
                topic=topic,
                api_key=data["openai_key"],
            )
        else:
            bg = data.get("bg_clip_path")
            path = ve.generate_thumbnail_local(
                title=data.get("title", ""),
                topic=topic,
                bg_image=bg,
            )
        # Move into topic folder
        if topic:
            dest = str(_topic_dir(topic, "thumbnails") / Path(path).name)
            _shutil.move(path, dest)
            path = dest
        _register(path)
        return jsonify({"path": path, "filename": Path(path).name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/thumbnail/<filename>")
def serve_thumbnail(filename):
    p = _resolve(filename, ve.THUMB_OUTPUT)
    if not p:
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="image/png")


# ── AI video generation endpoints ─────────────────────────────────────────────

@app.route("/video-methods")
def video_methods():
    methods = []
    for mid, info in vg.METHODS.items():
        entry = dict(info)
        entry["id"] = mid
        if mid == "higgsfield":
            entry["installed"] = vg.higgsfield_installed()
            entry["models_list"] = vg.get_higgsfield_models()
        methods.append(entry)
    return jsonify({"methods": methods})


@app.route("/hf-token")
def hf_token():
    token = os.environ.get("HF_TOKEN", "")
    return jsonify({"token": token})


@app.route("/higgsfield-status")
def higgsfield_status():
    exe = vg._find_higgsfield()
    if not exe:
        return jsonify({"installed": False, "authenticated": False,
                        "hint": "Run: npm install -g @higgsfield/cli"})
    try:
        r = subprocess.run([exe, "auth", "token"], capture_output=True, text=True, timeout=8,
                           shell=(os.name == "nt"))
        token = r.stdout.strip()
        authenticated = bool(token) and r.returncode == 0
        return jsonify({"installed": True, "authenticated": authenticated,
                        "hint": "" if authenticated else "Run: higgsfield auth login"})
    except Exception as e:
        return jsonify({"installed": True, "authenticated": False, "hint": str(e)})


@app.route("/generate-ai-clip", methods=["POST"])
def generate_ai_clip():
    data = request.get_json()
    topic = data.get("topic", "").strip()
    method = data.get("method", "ken_burns")
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    job_id = uuid.uuid4().hex[:8]
    out_dir    = _topic_dir(topic, "clips")  if topic else vg.AI_CLIP_OUTPUT
    images_dir = _topic_dir(topic, "images") if topic else None
    out_path   = str(out_dir / f"clip_{job_id}.mp4")
    _register(out_path)

    vg.start_generate(
        job_id=job_id,
        method=method,
        prompt=prompt,
        duration=float(data.get("duration", 5)),
        hf_token=data.get("hf_token", "") or os.environ.get("HF_TOKEN", ""),
        model=data.get("model", ""),
        output_path=out_path,
        images_dir=str(images_dir) if images_dir else None,
        gemini_key=data.get("gemini_key", "") or os.environ.get("GEMINI_API_KEY", ""),
    )
    return jsonify({"job_id": job_id})


@app.route("/ai-clip-status/<job_id>")
def ai_clip_status(job_id):
    job = vg._gen_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    # Register image so serve_ai_clip_thumb can find it without rglob
    if job.get("status") == "done" and job.get("img_path"):
        _register(job["img_path"])
    return jsonify(job)


@app.route("/ai-clip-file/<filename>")
def serve_ai_clip(filename):
    p = _resolve(filename, vg.AI_CLIP_OUTPUT)
    if not p:
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="video/mp4")


@app.route("/ai-clip-thumb/<filename>")
def serve_ai_clip_thumb(filename):
    p = _resolve(filename, sg.CACHE_DIR)
    if not p:
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), mimetype="image/png")


@app.route("/generate-scenes", methods=["POST"])
def generate_scenes():
    """
    Analyse a voiceover (or fall back to script text) and return a list of
    scenes with AI-enhanced image prompts ready for clip generation.
    """
    data = request.get_json()
    audio_file = data.get("audio_filename", "")
    script_text = data.get("script_text", "").strip()
    topic = data.get("topic", "video")
    style = data.get("style", "cinematic")
    interval   = float(data.get("interval", 8))
    max_scenes = int(data.get("max_scenes", 500))
    api_key    = data.get("api_key", "").strip()

    try:
        # ── Transcribe if audio present ──────────────────────────────────────
        segments = []
        if audio_file:
            audio_p = _resolve(audio_file, OUTPUT_DIR)
            if audio_p:
                try:
                    segments = sg.transcribe(str(audio_p), model_size="tiny")
                except Exception as e:
                    print(f"[scenes] Transcription skipped: {e}", flush=True)

        # ── Build scenes ──────────────────────────────────────────────────────
        if segments:
            scenes = sg.group_into_scenes(segments, mode="interval",
                                          interval_secs=interval, max_scenes=max_scenes)
        elif script_text:
            scenes = vg.scenes_from_text(script_text,
                                          interval_chars=max(1, int(interval * 25)),
                                          min_dur=interval, max_scenes=max_scenes)
        else:
            return jsonify({"error": "Provide audio_filename or script_text"}), 400

        if not scenes:
            return jsonify({"error": "Could not extract scenes"}), 400

        # ── Enhance prompts with Gemini ───────────────────────────────────────
        if api_key:
            try:
                scenes = sg.enhance_prompts_gemini(scenes, topic, api_key, style=style)
            except Exception as e:
                print(f"[scenes] Prompt enhancement failed: {e}", flush=True)
                for s in scenes:
                    s["prompt"] = sg.build_scene_prompt(s["text"], style,
                                                        s.get("start", 0), s.get("end", 5))
        else:
            for s in scenes:
                s["prompt"] = sg.build_scene_prompt(s["text"], style,
                                                    s.get("start", 0), s.get("end", 5))

        return jsonify({"scenes": scenes})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5847)
    args = parser.parse_args()

    print(f"[server] Starting on port {args.port}", flush=True)

    # Warmup TTS in background thread so first generation is faster
    import threading
    threading.Thread(target=tts_models.warmup, args=("chatterbox",), daemon=True).start()

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
