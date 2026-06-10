"""
AI video generation methods:
  - ken_burns : AI image (HF) → pan/zoom animation — fully free with HF token
  - hf_video  : HuggingFace text-to-video inference API — free with HF token
  - higgsfield: Higgsfield CLI (npm install -g @higgsfield/cli) — credit-based, 35+ models
"""
import os
import json
import time
import shutil
import subprocess
import threading
from pathlib import Path

import requests


def extract_video_thumb(video_path: str, out_dir: Path | None = None) -> str | None:
    """Extract first frame of a video as JPEG using ffmpeg. Returns thumb path or None."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        save_dir = out_dir or Path(video_path).parent
        thumb = str(save_dir / (Path(video_path).stem + "_thumb.jpg"))
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path,
             "-ss", "00:00:00.5", "-vframes", "1", "-q:v", "3", thumb],
            capture_output=True, timeout=30,
        )
        return thumb if Path(thumb).exists() else None
    except Exception as e:
        print(f"[video_gen] thumb extract failed: {e}", flush=True)
        return None

ROOT = Path(__file__).parent.parent
AI_CLIP_OUTPUT = ROOT / "output" / "clips"   # global fallback (no topic)
AI_CLIP_OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Job tracker ───────────────────────────────────────────────────────────────

_gen_jobs: dict = {}


def _upd(job_id: str, msg: str, pct: int = 0, status: str = "running"):
    _gen_jobs.setdefault(job_id, {}).update({"msg": msg, "pct": pct, "status": status})
    print(f"[video_gen] {msg}", flush=True)


# ── Gemini image model registry (needs to be defined before METHODS) ─────────

GEMINI_IMAGE_MODELS: dict[str, str] = {
    "gemini_flash_img": "Gemini Flash Image (Free — Gemini Key)",
    "gemini_imagen_3":  "Imagen 3 (Free — Gemini Key)",
}

# ── Higgsfield image model registry (needs to be defined before METHODS) ─────

HIGGSFIELD_IMAGE_MODELS: dict[str, str] = {
    "nano_banana_2":             "Nano Banana Pro (Higgsfield)",
    "nano_banana_flash":         "Nano Banana 2 (Higgsfield)",
    "nano_banana":               "Nano Banana (Higgsfield)",
    "flux_2":                    "FLUX.2 (Higgsfield)",
    "flux_kontext":              "Flux Kontext (Higgsfield)",
    "grok_image":                "Grok Image (Higgsfield)",
    "seedream_v5_lite":          "Seedream V5 Lite (Higgsfield)",
    "seedream_v4_5":             "Seedream 4.5 (Higgsfield)",
    "cinematic_studio_2_5":      "Cinematic Studio 2.5 (Higgsfield)",
    "text2image_soul_v2":        "Soul V2 (Higgsfield)",
    "imagegen_2_0":              "GPT Image 2 (Higgsfield)",
    "z_image":                   "Z Image (Higgsfield)",
    "image_auto":                "Image Auto / Best (Higgsfield)",
}


# ── Methods registry ──────────────────────────────────────────────────────────

METHODS = {
    "ken_burns": {
        "name": "AI Images + Ken Burns",
        "description": "Generates an AI image then animates with pan/zoom. HF models are free; Higgsfield models are credit-based but higher quality.",
        "free": True,
        "badge": "FREE/PAID",
        "models": {
            # ── Google Gemini (free with existing Gemini API key) ─────────────
            **{k: v for k, v in GEMINI_IMAGE_MODELS.items()},
            # ── HuggingFace (free with HF token) ─────────────────────────────
            "black-forest-labs/FLUX.1-schnell":         "FLUX Schnell (HF — Fast, Free)",
            "stabilityai/stable-diffusion-xl-base-1.0": "SDXL (HF — Best Quality, Free)",
            "stabilityai/stable-diffusion-2-1":         "SD 2.1 (HF — Lighter, Free)",
            "Lykon/dreamshaper-xl-1-0":                 "DreamShaper XL (HF — Free)",
            # ── Higgsfield (credit-based, high quality) ────────────────────
            **{k: v for k, v in HIGGSFIELD_IMAGE_MODELS.items()},
        },
    },
    "hf_video": {
        "name": "HF Text-to-Video",
        "description": "Short video clips (~2–4 s) via HF Inference API. Free with HF token.",
        "free": True,
        "badge": "FREE",
        "models": {
            "damo-vilab/text-to-video-ms-1.7b": "Text-to-Video 1.7B (DAMO)",
            "ali-vilab/text-to-video-ms-1.7b": "Text-to-Video 1.7B (Ali)",
        },
    },
    "higgsfield": {
        "name": "Higgsfield AI",
        "description": "20+ models: Kling, Hailuo, Veo 3, Grok, Wan, Seedance. Credit-based. Requires npm install -g @higgsfield/cli then higgsfield auth login",
        "free": False,
        "badge": "PAID",
        "models": {
            "kling3_0":             "Kling v3.0",
            "kling2_6":             "Kling 2.6 Video",
            "minimax_hailuo":       "Minimax Hailuo",
            "veo3":                 "Google Veo 3",
            "veo3_1":               "Google Veo 3.1",
            "veo3_1_lite":          "Google Veo 3.1 Lite",
            "grok_video":           "Grok Video",
            "wan2_7":               "Wan 2.7",
            "wan2_6":               "Wan 2.6 Video",
            "seedance_2_0":         "Seedance 2.0",
            "seedance1_5":          "Seedance 1.5 Pro",
            "cinematic_studio_3_0": "Cinematic Studio 3.0",
        },
    },
}

HF_API_BASE = "https://api-inference.huggingface.co/models"


# ── Higgsfield CLI ────────────────────────────────────────────────────────────

def _find_higgsfield() -> str | None:
    # Try several names — npm on Windows installs as .CMD (case-insensitive)
    for name in ["higgsfield", "higgsfield.cmd", "higgs", "higgs.cmd"]:
        found = shutil.which(name)
        if found:
            return found          # return the FULL resolved path, not just the name
    # Fallback: check known npm global locations
    candidates = [
        os.path.expandvars(r"%APPDATA%\npm\higgsfield.cmd"),
        os.path.expandvars(r"%APPDATA%\npm\higgsfield.CMD"),
        os.path.expandvars(r"%APPDATA%\npm\higgsfield"),
        r"C:\Program Files\nodejs\higgsfield.cmd",
        os.path.expanduser("~/.npm-global/bin/higgsfield"),
        "/usr/local/bin/higgsfield",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def higgsfield_installed() -> bool:
    return _find_higgsfield() is not None


def generate_higgsfield(
    job_id: str,
    prompt: str,
    duration: int = 5,   # kept in signature for call-site compat; not sent to CLI
    model: str = "kling3_0",
    output_path: str | None = None,
):
    del duration  # model controls its own duration; passing --duration breaks most models
    """
    Uses: higgsfield generate create <model> --prompt "..." --wait --json
    Then downloads the output video URL to disk.
    """
    exe = _find_higgsfield()
    if not exe:
        _upd(job_id, "Higgsfield CLI not found. Run: npm install -g @higgsfield/cli", 0, "error")
        _gen_jobs[job_id]["error"] = "Higgsfield CLI not installed"
        return

    if not output_path:
        output_path = str(AI_CLIP_OUTPUT / f"hig_{job_id}.mp4")

    _upd(job_id, f"Submitting to Higgsfield ({model})…", 10)

    # --duration is model-specific; omit it to avoid "unknown parameter" errors.
    # Use --wait so the call blocks until the video is ready.
    cmd = [
        exe, "generate", "create", model,
        "--prompt", prompt,
        "--wait",
        "--wait-timeout", "12m",
        "--json",
    ]

    try:
        _upd(job_id, "Waiting for generation (can take 1–3 min)…", 20)
        # shell=True ensures .CMD files resolve on Windows
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=750,
                                shell=(os.name == "nt"))
        stdout = result.stdout.strip()

        if result.returncode != 0:
            err = (result.stderr or stdout or "Higgsfield error").strip()
            raise RuntimeError(err[:500])

        # Parse JSON output to find the video URL
        video_url = None
        try:
            data = json.loads(stdout)
            # Try common output shapes
            if isinstance(data, dict):
                outputs = data.get("outputs") or data.get("result") or []
                if isinstance(outputs, list):
                    for o in outputs:
                        url = o.get("url") or o.get("uri") or o.get("download_url") or ""
                        if url.endswith(".mp4") or "video" in o.get("type", ""):
                            video_url = url; break
                if not video_url:
                    video_url = (data.get("result_url") or data.get("url") or
                                 data.get("download_url") or "")
        except Exception:
            # Fall back: scan lines for a URL ending in .mp4
            for line in stdout.splitlines():
                ln = line.strip()
                if ln.startswith("http") and ".mp4" in ln:
                    video_url = ln.split()[0]; break

        if not video_url:
            raise RuntimeError(f"No video URL in response: {stdout[:300]}")

        _upd(job_id, "Downloading video…", 85)
        r = requests.get(video_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(256 * 1024):
                f.write(chunk)

        thumb = extract_video_thumb(output_path)
        _upd(job_id, "Video ready!", 100, "done")
        _gen_jobs[job_id]["clip_path"] = output_path
        if thumb:
            _gen_jobs[job_id]["img_path"] = thumb

    except subprocess.TimeoutExpired:
        _upd(job_id, "Higgsfield timed out (>12 min)", 0, "error")
        _gen_jobs[job_id]["error"] = "Timeout"
    except Exception as e:
        _upd(job_id, str(e), 0, "error")
        _gen_jobs[job_id]["error"] = str(e)


# ── HuggingFace Text-to-Video ─────────────────────────────────────────────────

def generate_hf_video(
    job_id: str,
    prompt: str,
    hf_token: str,
    model_id: str = "damo-vilab/text-to-video-ms-1.7b",
    output_path: str | None = None,
):
    if not output_path:
        output_path = str(AI_CLIP_OUTPUT / f"hfv_{job_id}.mp4")

    _upd(job_id, "Calling HF text-to-video model…", 10)
    headers = {"Authorization": f"Bearer {hf_token}"}

    for attempt in range(4):
        r = requests.post(
            f"{HF_API_BASE}/{model_id}",
            headers=headers,
            json={"inputs": prompt},
            timeout=180,
        )
        if r.status_code == 503:
            wait = min(int(r.json().get("estimated_time", 30)), 60)
            _upd(job_id, f"Model loading, waiting {wait}s… (attempt {attempt+1}/4)", 10 + attempt * 10)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            break
        raise RuntimeError(f"HF API {r.status_code}: {r.text[:300]}")
    else:
        raise RuntimeError("HF model did not load after 4 retries")

    r.raise_for_status()
    Path(output_path).write_bytes(r.content)
    thumb = extract_video_thumb(output_path)
    _upd(job_id, "Video ready!", 100, "done")
    _gen_jobs[job_id]["clip_path"] = output_path
    if thumb:
        _gen_jobs[job_id]["img_path"] = thumb


# ── Higgsfield image generation ──────────────────────────────────────────────


def _extract_image_url(stdout: str) -> str:
    """Parse Higgsfield JSON output to find the generated image URL."""
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            outputs = data.get("outputs") or data.get("result") or []
            if isinstance(outputs, list):
                for o in outputs:
                    url = o.get("url") or o.get("uri") or o.get("download_url") or ""
                    if url:
                        return url
            # Flat fields
            for key in ("url", "result_url", "download_url", "image_url"):
                if data.get(key):
                    return data[key]
    except Exception:
        pass
    # Fallback: scan for any http URL in the output
    for line in stdout.splitlines():
        ln = line.strip()
        if ln.startswith("http"):
            return ln.split()[0]
    return ""


def generate_image_higgsfield(
    prompt: str,
    model_id: str,
    output_dir: Path | None = None,
) -> str:
    """Generate a still image via Higgsfield CLI; return saved file path."""
    exe = _find_higgsfield()
    if not exe:
        raise RuntimeError("Higgsfield CLI not found. Run: npm install -g @higgsfield/cli")

    save_dir = output_dir or AI_CLIP_OUTPUT.parent / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "generate", "create", model_id, "--prompt", prompt, "--wait", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                            shell=(os.name == "nt"))

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown error").strip()
        raise RuntimeError(f"Higgsfield image error: {err[:400]}")

    image_url = _extract_image_url(result.stdout.strip())
    if not image_url:
        raise RuntimeError(f"No image URL in response: {result.stdout.strip()[:300]}")

    print(f"[video_gen] Downloading Higgsfield image: {image_url[:60]}…", flush=True)
    r = requests.get(image_url, stream=True, timeout=60)
    r.raise_for_status()

    # Detect extension from Content-Type
    ct = r.headers.get("Content-Type", "image/jpeg")
    ext = ".jpg" if "jpeg" in ct else ".png" if "png" in ct else ".webp" if "webp" in ct else ".jpg"
    import uuid as _uuid
    img_path = str(save_dir / f"hig_{_uuid.uuid4().hex[:8]}{ext}")
    with open(img_path, "wb") as f:
        for chunk in r.iter_content(256 * 1024):
            f.write(chunk)
    return img_path


def _get_image_for_ken_burns(
    prompt: str,
    image_model: str,
    hf_token: str,
    output_dir: Path | None = None,
    gemini_key: str = "",
) -> str:
    """Route image generation to Gemini, Higgsfield CLI, or HF API."""
    if image_model in GEMINI_IMAGE_MODELS:
        from scene_gen import generate_image_gemini
        return generate_image_gemini(prompt, image_model, gemini_key, output_dir)
    if image_model in HIGGSFIELD_IMAGE_MODELS:
        return generate_image_higgsfield(prompt, image_model, output_dir)
    # Default: HuggingFace inference API
    from scene_gen import generate_image_hf
    return generate_image_hf(prompt, image_model, hf_token,
                              width=1920, height=1080, output_dir=output_dir)


# ── Ken Burns (AI image → pan/zoom animation) ─────────────────────────────────

def generate_ken_burns(
    job_id: str,
    prompt: str,
    hf_token: str,
    duration: float = 5.0,
    image_model: str = "black-forest-labs/FLUX.1-schnell",
    output_path: str | None = None,
    images_dir: str | None = None,
    gemini_key: str = "",
):
    """Generate an AI image then animate it with a slow Ken Burns pan/zoom."""
    if not output_path:
        output_path = str(AI_CLIP_OUTPUT / f"kb_{job_id}.mp4")

    # ── Step 1: generate image (Gemini / Higgsfield / HF) ──────────────────────
    if image_model in GEMINI_IMAGE_MODELS:
        src_label = "Gemini"
    elif image_model in HIGGSFIELD_IMAGE_MODELS:
        src_label = "Higgsfield"
    else:
        src_label = "HuggingFace"
    _upd(job_id, f"Generating image via {src_label}…", 10)
    try:
        img_dir = Path(images_dir) if images_dir else None
        img_path = _get_image_for_ken_burns(prompt, image_model, hf_token, img_dir, gemini_key)
    except Exception as e:
        _upd(job_id, f"Image generation failed: {e}", 0, "error")
        _gen_jobs[job_id]["error"] = str(e)
        return

    _gen_jobs[job_id]["img_path"] = img_path  # expose for thumbnail

    # ── Step 2: animate ──────────────────────────────────────────────────────
    _upd(job_id, "Animating with Ken Burns effect…", 60)
    try:
        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.open(img_path).convert("RGB").resize((1920, 1080), PILImage.LANCZOS)
        img_arr = np.array(img)
        W, H = 1920, 1080
        fps = 24

        from moviepy.editor import VideoClip
        import moviepy.config as mpconfig
        try:
            import imageio_ffmpeg
            mpconfig.change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
        except Exception:
            pass

        def make_frame(t: float):
            progress = t / duration  # 0 → 1

            # Slow zoom-out + slight rightward pan
            scale = 1.18 - 0.18 * progress          # 1.18x → 1.0x
            scale = max(1.0, scale)
            new_w = int(W * scale)
            new_h = int(H * scale)

            # Pan: start top-left, end center
            ox = int((new_w - W) * (1.0 - progress) * 0.5)
            oy = int((new_h - H) * (1.0 - progress) * 0.5)

            frame_img = PILImage.fromarray(img_arr).resize(
                (new_w, new_h), PILImage.LANCZOS
            )
            cropped = np.array(frame_img)[oy : oy + H, ox : ox + W]
            return cropped

        clip = VideoClip(make_frame, duration=duration).set_fps(fps)
        clip.write_videofile(
            output_path, fps=fps, codec="libx264",
            audio=False, logger=None, preset="fast",
            ffmpeg_params=["-crf", "23"],
        )
        clip.close()

        _upd(job_id, "Video ready!", 100, "done")
        _gen_jobs[job_id]["clip_path"] = output_path

    except Exception as e:
        import traceback; traceback.print_exc()
        _upd(job_id, str(e), 0, "error")
        _gen_jobs[job_id]["error"] = str(e)


# ── Auto-generate scenes from script text ────────────────────────────────────

def scenes_from_text(
    text: str,
    interval_chars: int = 400,
    max_scenes: int = 500,
    min_dur: float = 0.5,
) -> list[dict]:
    """Split plain text into scenes when no audio is available."""
    import re
    # Split on paragraph breaks first, then sentence breaks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # If paragraphs are too long, split further by sentence
    scenes = []
    for para in paragraphs:
        if len(para) <= interval_chars * 1.5:
            scenes.append(para)
        else:
            sents = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for s in sents:
                if len(buf) + len(s) < interval_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        scenes.append(buf)
                    buf = s
            if buf:
                scenes.append(buf)

    # Cap and build output
    scenes = scenes[:max_scenes]
    # Estimate duration: ~140 words/min speaking rate
    result = []
    for i, s in enumerate(scenes):
        words = len(s.split())
        dur = max(min_dur, round(words / 140 * 60, 2))
        result.append({"idx": i, "text": s, "duration": dur, "prompt": s[:80]})
    return result


# ── Dispatch ──────────────────────────────────────────────────────────────────

def start_generate(
    job_id: str,
    method: str,
    prompt: str,
    duration: float = 5.0,
    hf_token: str = "",
    model: str = "",
    output_path: str | None = None,
    images_dir: str | None = None,
    gemini_key: str = "",
):
    _gen_jobs[job_id] = {"status": "running", "msg": "Starting…", "pct": 0}

    def run():
        try:
            if method == "higgsfield":
                generate_higgsfield(job_id, prompt, int(duration), model or "auto", output_path)
            elif method == "hf_video":
                mid = model or "damo-vilab/text-to-video-ms-1.7b"
                generate_hf_video(job_id, prompt, hf_token, mid, output_path)
            elif method == "ken_burns":
                mid = model or "black-forest-labs/FLUX.1-schnell"
                generate_ken_burns(job_id, prompt, hf_token, duration, mid,
                                   output_path, images_dir, gemini_key)
            else:
                _gen_jobs[job_id] = {
                    "status": "error", "error": f"Unknown method: {method}",
                    "msg": f"Unknown method: {method}", "pct": 0,
                }
        except Exception as e:
            import traceback; traceback.print_exc()
            _gen_jobs[job_id] = {
                "status": "error", "error": str(e), "msg": str(e), "pct": 0,
            }

    threading.Thread(target=run, daemon=True).start()


def get_higgsfield_models() -> list[dict]:
    """Try to list models from the Higgsfield CLI; fall back to known list."""
    exe = _find_higgsfield()
    if not exe:
        return []
    try:
        r = subprocess.run(
            [exe, "model", "list", "--json"],
            capture_output=True, text=True, timeout=15,
            shell=(os.name == "nt"),
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    # Known fallback list (job_set_type values)
    return [
        {"id": "kling3_0",             "name": "Kling v3.0"},
        {"id": "kling2_6",             "name": "Kling 2.6 Video"},
        {"id": "minimax_hailuo",       "name": "Minimax Hailuo"},
        {"id": "veo3",                 "name": "Google Veo 3"},
        {"id": "veo3_1_lite",          "name": "Google Veo 3.1 Lite"},
        {"id": "grok_video",           "name": "Grok Video"},
        {"id": "wan2_7",               "name": "Wan 2.7"},
        {"id": "seedance_2_0",         "name": "Seedance 2.0"},
        {"id": "cinematic_studio_3_0", "name": "Cinematic Studio 3.0"},
    ]
