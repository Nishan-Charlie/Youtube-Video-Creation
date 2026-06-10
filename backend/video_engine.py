"""
Video assembly engine — stock footage, text slides, captions, thumbnail.
Uses moviepy 1.x API.
"""
import os
import re
import json
import time
import uuid
import shutil
import textwrap
import threading
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# ── Output dirs ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
VIDEO_OUTPUT  = ROOT / "output" / "videos"
CLIP_CACHE    = ROOT / "output" / "clip_cache"
THUMB_OUTPUT  = ROOT / "output" / "thumbnails"
for d in [VIDEO_OUTPUT, CLIP_CACHE, THUMB_OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Progress tracking ─────────────────────────────────────────────────────────

_video_jobs: dict = {}

def _upd(job_id, msg, pct=0):
    _video_jobs.setdefault(job_id, {}).update({"msg": msg, "pct": pct})
    print(f"[video] {msg}", flush=True)


# ── Pexels stock footage ──────────────────────────────────────────────────────

PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"

def search_pexels_videos(query: str, api_key: str, per_page: int = 6) -> list[dict]:
    if not api_key:
        return []
    headers = {"Authorization": api_key}
    r = requests.get(PEXELS_VIDEO_URL,
                     params={"query": query, "per_page": per_page, "orientation": "landscape"},
                     headers=headers, timeout=10)
    r.raise_for_status()
    results = []
    for v in r.json().get("videos", []):
        # Pick best quality ≤ 1080p
        files = sorted(v.get("video_files", []),
                       key=lambda f: f.get("width", 0), reverse=True)
        best = next((f for f in files if f.get("width", 0) <= 1920), files[0] if files else None)
        if best:
            results.append({
                "id": v["id"],
                "url": best["link"],
                "thumb": v.get("image", ""),
                "duration": v.get("duration", 5),
                "width": best.get("width", 1920),
                "height": best.get("height", 1080),
                "photographer": v.get("user", {}).get("name", ""),
            })
    return results


def download_clip(url: str, clip_id: str) -> str:
    dest = CLIP_CACHE / f"{clip_id}.mp4"
    if dest.exists():
        return str(dest)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024 * 256):
            f.write(chunk)
    return str(dest)


# ── Text slide generator ──────────────────────────────────────────────────────

SLIDE_FONTS = ["Arial", "Calibri", "Segoe UI", "DejaVu Sans", "FreeSans"]

def _get_font(size: int):
    for name in SLIDE_FONTS:
        try:
            return ImageFont.truetype(name + ".ttf", size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


SLIDE_PRESETS = {
    "dark":    {"bg": (15, 15, 15),    "text": (230, 230, 230), "accent": (255, 69, 0)},
    "light":   {"bg": (245, 245, 245), "text": (20, 20, 20),    "accent": (99, 56, 237)},
    "blue":    {"bg": (10, 25, 60),    "text": (220, 230, 255), "accent": (100, 160, 255)},
    "gradient":{"bg": (20, 10, 40),    "text": (255, 255, 255), "accent": (180, 100, 255)},
}

def generate_slide(text: str, duration: float = 5.0, style: str = "dark",
                   subtitle: str = "", size=(1920, 1080)) -> str:
    preset = SLIDE_PRESETS.get(style, SLIDE_PRESETS["dark"])
    W, H = size
    img = Image.new("RGB", (W, H), preset["bg"])
    draw = ImageDraw.Draw(img)

    # Accent bar at top
    draw.rectangle([(0, 0), (W, 8)], fill=preset["accent"])

    # Main text
    font_size = 72 if len(text) < 60 else 56 if len(text) < 120 else 44
    font = _get_font(font_size)
    wrapped = textwrap.fill(text, width=38)
    # Center text
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (W - tw) // 2, (H - th) // 2 - (60 if subtitle else 0)
    draw.text((x + 2, y + 2), wrapped, fill=(0, 0, 0, 100), font=font)  # shadow
    draw.text((x, y), wrapped, fill=preset["text"], font=font)

    # Subtitle
    if subtitle:
        sfont = _get_font(32)
        sbbox = draw.textbbox((0, 0), subtitle, font=sfont)
        sw = sbbox[2] - sbbox[0]
        draw.text(((W - sw) // 2, y + th + 30), subtitle,
                  fill=preset["accent"], font=sfont)

    slide_id = uuid.uuid4().hex[:8]
    path = CLIP_CACHE / f"slide_{slide_id}.png"
    img.save(str(path))
    return str(path)


# ── Auto-captions (faster-whisper) ───────────────────────────────────────────

def transcribe_audio(audio_path: str, model_size: str = "tiny") -> list[dict]:
    """Returns list of {start, end, text} segments."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=False)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    except ImportError:
        print("[captions] faster-whisper not installed, skipping captions", flush=True)
        return []


def segments_to_srt(segments: list[dict], output_path: str):
    def fmt_time(t):
        h = int(t // 3600); m = int((t % 3600) // 60)
        s = int(t % 60); ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}", seg["text"], ""]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ── Thumbnail generator ───────────────────────────────────────────────────────

def generate_thumbnail_dalle(title: str, topic: str, api_key: str) -> str:
    """Generate thumbnail using DALL-E 3 via OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = (f"YouTube thumbnail for a video about '{topic}'. "
              f"Text overlay: '{title}'. "
              "Bold, vibrant, eye-catching. Professional quality. "
              "16:9 aspect ratio. No watermarks.")
    resp = client.images.generate(model="dall-e-3", prompt=prompt,
                                  size="1792x1024", quality="standard", n=1)
    image_url = resp.data[0].url
    img_data = requests.get(image_url, timeout=30).content
    path = THUMB_OUTPUT / f"thumb_{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(img_data)
    return str(path)


def generate_thumbnail_local(title: str, topic: str, bg_image: str | None = None) -> str:
    """Generate a simple styled thumbnail without external API."""
    W, H = 1280, 720
    # Background
    if bg_image and Path(bg_image).exists():
        img = Image.open(bg_image).convert("RGB").resize((W, H), Image.LANCZOS)
        # Dark overlay
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 160))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    else:
        img = Image.new("RGB", (W, H), (20, 10, 40))

    draw = ImageDraw.Draw(img)
    # Gradient-like side accent
    for i in range(20):
        alpha = int(255 * (1 - i / 20))
        draw.rectangle([(0, 0), (i * 5, H)], fill=(180, 60, 255, alpha))

    # Title text
    font = _get_font(80 if len(title) < 30 else 60)
    wrapped = textwrap.fill(title, width=22)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (W - tw) // 2, (H - th) // 2

    # Shadow
    draw.text((x + 3, y + 3), wrapped, fill=(0, 0, 0), font=font)
    draw.text((x, y), wrapped, fill=(255, 255, 255), font=font)

    # Topic tag
    tfont = _get_font(32)
    tag = f"  {topic.upper()}  "
    tbbox = draw.textbbox((0, 0), tag, font=tfont)
    tw2 = tbbox[2] - tbbox[0]
    draw.rectangle([(W // 2 - tw2 // 2 - 4, y + th + 20),
                    (W // 2 + tw2 // 2 + 4, y + th + 60)], fill=(255, 69, 0))
    draw.text((W // 2 - tw2 // 2, y + th + 22), tag, fill=(255, 255, 255), font=tfont)

    path = THUMB_OUTPUT / f"thumb_{uuid.uuid4().hex[:8]}.png"
    img.save(str(path))
    return str(path)


# ── Video assembly ────────────────────────────────────────────────────────────

def assemble_video(
    job_id: str,
    sequence: list[dict],      # [{type, src/text, duration, style}, ...]
    audio_path: str,
    output_path: str,
    burn_captions: bool = False,
    resolution: tuple = (1920, 1080),
    fps: int = 30,
) -> None:
    try:
        from moviepy.editor import (
            VideoFileClip, ImageClip, AudioFileClip,
            concatenate_videoclips, CompositeVideoClip, TextClip
        )
        import moviepy.config as mpconfig

        # Use imageio-ffmpeg if system ffmpeg not found
        try:
            import imageio_ffmpeg
            mpconfig.change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
        except Exception:
            pass

        _upd(job_id, "Loading audio…", 5)
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration

        # Calculate duration per clip
        _upd(job_id, "Planning clip durations…", 10)
        valid = [c for c in sequence if c.get("enabled", True)]
        if not valid:
            raise ValueError("No clips in sequence")

        # Clips with set duration keep theirs; others share remainder
        fixed_dur = sum(c.get("duration", 0) for c in valid if c.get("duration", 0) > 0)
        flexible = [c for c in valid if not c.get("duration", 0)]
        per_flexible = max(3.0, (total_duration - fixed_dur) / max(len(flexible), 1))
        for c in flexible:
            c["duration"] = per_flexible

        clips = []
        n = len(valid)
        for i, item in enumerate(valid):
            _upd(job_id, f"Processing clip {i+1}/{n}…", 10 + int(60 * i / n))
            dur = item.get("duration", 5)
            clip_type = item.get("type", "slide")
            W, H = resolution

            if clip_type == "video" and item.get("src"):
                try:
                    vc = VideoFileClip(item["src"]).subclip(0, min(dur, VideoFileClip(item["src"]).duration))
                    vc = vc.resize(resolution).set_duration(dur)
                    clips.append(vc)
                    continue
                except Exception as e:
                    print(f"[video] clip load failed: {e}, using slide fallback", flush=True)

            # Text slide (fallback or explicit)
            slide_path = generate_slide(
                text=item.get("text", ""),
                duration=dur,
                style=item.get("style", "dark"),
                subtitle=item.get("subtitle", ""),
                size=resolution,
            )
            ic = ImageClip(slide_path).set_duration(dur)
            clips.append(ic)

        _upd(job_id, "Concatenating clips…", 72)
        final = concatenate_videoclips(clips, method="compose")

        # Loop or trim to match audio
        if final.duration < total_duration:
            from moviepy.editor import concatenate_videoclips as cc
            loops = int(total_duration / final.duration) + 1
            final = cc([final] * loops, method="compose").subclip(0, total_duration)
        else:
            final = final.subclip(0, total_duration)

        _upd(job_id, "Mixing audio…", 80)
        final = final.set_audio(audio)

        # Burn captions
        if burn_captions:
            _upd(job_id, "Generating captions…", 85)
            segments = transcribe_audio(audio_path)
            if segments:
                caption_clips = []
                for seg in segments:
                    try:
                        tc = (TextClip(seg["text"], fontsize=44, color="white",
                                       stroke_color="black", stroke_width=2,
                                       method="caption", size=(int(W * 0.85), None))
                              .set_position(("center", 0.82), relative=True)
                              .set_start(seg["start"])
                              .set_duration(seg["end"] - seg["start"]))
                        caption_clips.append(tc)
                    except Exception:
                        pass
                if caption_clips:
                    final = CompositeVideoClip([final] + caption_clips)

        _upd(job_id, "Exporting MP4 (this takes a minute)…", 88)
        final.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="fast",
            logger=None,
        )
        final.close()
        audio.close()

        _upd(job_id, "Done", 100)
        _video_jobs[job_id]["status"] = "done"
        _video_jobs[job_id]["filename"] = Path(output_path).name

    except Exception as e:
        import traceback; traceback.print_exc()
        _video_jobs[job_id]["status"] = "error"
        _video_jobs[job_id]["error"] = str(e)


def start_assemble(job_id: str, **kwargs):
    _video_jobs[job_id] = {"status": "running", "msg": "Starting…", "pct": 0}
    t = threading.Thread(target=assemble_video, args=(job_id,), kwargs=kwargs, daemon=True)
    t.start()
