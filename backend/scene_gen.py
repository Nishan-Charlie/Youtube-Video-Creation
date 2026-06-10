"""
Scene generation: transcribe voiceover → group into scenes → generate images via HuggingFace.
"""
import re
import time
import uuid
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "output" / "scenes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Image style modifiers ─────────────────────────────────────────────────────

STYLES = {
    # ── Simple / hand-drawn ──────────────────────────────────────────────────────
    "stick_figure":  "__template__",   # uses STYLE_PROMPTS["stick_figure"] below
    "ms_paint":      "__template__",
    "whiteboard":    "__template__",
    "crayon":        "crayon drawing style, rough waxy lines, childlike coloring, simple shapes, pastel colors, scribble texture",
    "pencil_sketch": "pencil sketch, hand-drawn, rough graphite lines, simple shading, sketchbook style, black and white",
    "cartoon_flat":  "simple 2D cartoon, flat colors, bold outlines, no shadows, comic strip style, clean and friendly",
    "chalkboard":    "chalkboard drawing style, white chalk lines on dark green background, rough texture, educational diagram",
    "infographic":   "clean infographic style, flat icons, simple shapes, bright solid colors, modern explainer visual",
    # ── Artistic ────────────────────────────────────────────────────────────────
    "cinematic":     "cinematic photography, dramatic lighting, film grain, professional, 4K",
    "realistic":     "photorealistic, high detail, sharp focus, professional photography",
    "illustration":  "digital illustration, vibrant colors, artistic, concept art",
    "minimal":       "minimalist design, clean, modern, simple, flat design, vector art",
    "anime":         "anime style, cel shading, vibrant colors, studio quality",
    "documentary":   "documentary photography, natural lighting, candid, editorial",
    "3d_render":     "3D render, octane render, high quality, cinematic lighting",
    "watercolor":    "watercolor painting, soft colors, artistic, hand-painted",
    "oil_painting":  "oil painting, rich textures, expressive brushstrokes, classical art style",
    "pixel_art":     "pixel art, 16-bit retro style, vibrant colors, clean pixels, game art",
    "comic_book":    "comic book art, halftone dots, bold ink outlines, dramatic shading, superhero style",
    "neon_glow":     "neon glow aesthetic, dark background, vibrant neon lights, cyberpunk style, glowing outlines",
}

# ── Full prompt templates for styles that need more than a suffix ─────────────

def _ts(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _stick_figure_prompt(scene_text: str, start: float = 0, end: float = 5) -> str:
    return (
        f"A single frame from a stick-figure animation video at timestamp {_ts(start)}–{_ts(end)}. "
        "Minimal stick figure drawing in Microsoft Paint style. "
        "Black hand-drawn lines on pure white background.\n"
        f"Scene: {scene_text}\n\n"
        "A simple hand-drawn stick figure illustration in the style of Microsoft Paint doodles. "
        "Minimalist sketch on a pure white background. "
        "Black rough pen lines, slightly imperfect like hand-drawn. "
        "Stick figures only, no detailed anatomy. "
        "Simple objects (if needed) drawn as basic shapes (circles, rectangles, lines).\n\n"
        "Composition: centered, clean spacing, no clutter, no shading, no gradients, "
        "no color (only black line art).\n"
        "Style: early MS Paint / whiteboard sketch / doodle animation frame.\n"
        "Mood: simple, clear storytelling frame for animation.\n"
        "Must look like a single frame from a stick-figure explainer video.\n"
        "Background: pure white, empty, no texture.\n"
        "No text, no watermark, no logos.\n\n"
        "Style rules (IMPORTANT): "
        "Pure white background (#FFFFFF). Black line art only. "
        "Stick figures (circle head + line body + lines for limbs). "
        "No shading, no 3D, no gradients. No realistic humans. "
        "No detailed faces (optional: dot eyes only). "
        "Centered composition. Consistent character design across all frames."
    )


def _ms_paint_prompt(scene_text: str, start: float = 0, end: float = 5) -> str:
    return (
        f"Frame at {_ts(start)}–{_ts(end)}. "
        f"Microsoft Paint style digital drawing. Scene: {scene_text}. "
        "Thick pixelated lines, simple flat fill colors (red, blue, yellow, green), "
        "no anti-aliasing, no gradients, no shadows. "
        "Looks hand-drawn with the Paint brush tool. "
        "White canvas background. Naive, playful, retro digital art. "
        "Simple shapes: rectangles, circles, triangles. No text. "
        "Pure white background. Bold chunky lines."
    )


def _whiteboard_prompt(scene_text: str, start: float = 0, end: float = 5) -> str:
    return (
        f"Frame at {_ts(start)}–{_ts(end)}. "
        f"Whiteboard explainer sketch. Scene: {scene_text}. "
        "Black marker lines on a clean white background. "
        "Hand-drawn diagram style. Simple stick people, arrows, and labeled boxes. "
        "Rough pen strokes, slightly imperfect lines, educational illustration. "
        "No color, no shading — black lines on white only. "
        "Clean composition, centered layout. No text captions inside image."
    )


STYLE_PROMPTS: dict[str, callable] = {
    "stick_figure": _stick_figure_prompt,
    "ms_paint":     _ms_paint_prompt,
    "whiteboard":   _whiteboard_prompt,
}

# Gemini instructions per style (tells Gemini what kind of scene to describe)
_STYLE_GEMINI_INSTRUCTION: dict[str, str] = {
    "stick_figure": (
        "You are a storyboard artist creating a stick-figure explainer animation about \"{topic}\".\n\n"
        "For each narration segment, write ONE brief action description (max 15 words) of what a stick figure "
        "is doing to illustrate that moment.\n"
        "Rules:\n"
        "- Describe ONLY the stick figure's action or pose (e.g., 'A stick figure points at a large circle')\n"
        "- Use only basic objects: circles, rectangles, lines, arrows\n"
        "- Keep it visual and simple — no complex scenes\n"
        "- Output ONLY a numbered list (same numbering as input)\n\n"
        "Segments:\n{scene_texts}"
    ),
    "ms_paint": (
        "You are creating MS Paint-style thumbnails for a video about \"{topic}\".\n\n"
        "For each segment write ONE brief scene description (max 12 words) of what simple "
        "flat-color shapes and stick figures show in that moment.\n"
        "Output ONLY a numbered list.\n\nSegments:\n{scene_texts}"
    ),
    "whiteboard": (
        "You are creating whiteboard sketch frames for a video about \"{topic}\".\n\n"
        "For each segment write ONE brief description (max 15 words) of what a simple "
        "marker diagram or arrow-and-box sketch shows.\n"
        "Output ONLY a numbered list.\n\nSegments:\n{scene_texts}"
    ),
}

_DEFAULT_GEMINI_INSTRUCTION = (
    "You are a visual director creating a YouTube video about \"{topic}\".\n\n"
    "Convert each narration segment into a short, vivid image generation prompt.\n"
    "Rules:\n"
    "- Visually represent what is being said\n"
    "- Be specific, descriptive, cinematic\n"
    "- Do NOT include text/words in the image\n"
    "- Keep under 60 words each\n"
    "- Output ONLY a numbered list, one prompt per line (same numbering)\n\n"
    "Segments:\n{scene_texts}"
)


def build_scene_prompt(scene_text: str, style: str,
                       start: float = 0, end: float = 5) -> str:
    """Build the full image prompt for one scene, using style-specific templates."""
    fn = STYLE_PROMPTS.get(style)
    if fn:
        return fn(scene_text, start, end)
    style_hint = STYLES.get(style, STYLES["cinematic"])
    return f"{scene_text[:80]}, {style_hint}"

# HuggingFace models for image generation (all free with HF token)
HF_MODELS = {
    "SDXL (Best quality)":   "stabilityai/stable-diffusion-xl-base-1.0",
    "FLUX Schnell (Fast)":   "black-forest-labs/FLUX.1-schnell",
    "SD 2.1 (Lightweight)":  "stabilityai/stable-diffusion-2-1",
    "DreamShaper":           "Lykon/dreamshaper-xl-1-0",
}


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(audio_path: str, model_size: str = "tiny") -> list[dict]:
    """Returns [{start, end, text}, ...]"""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments]


# ── Scene grouping ────────────────────────────────────────────────────────────

def group_into_scenes(segments: list[dict], mode: str = "interval",
                       interval_secs: float = 8.0, max_scenes: int = 500) -> list[dict]:
    """
    Groups transcript segments into scenes.
    mode:
      'interval' — one scene per N seconds
      'sentence'  — one scene per sentence group
      'paragraph' — one scene per paragraph pause (gap > 0.8s)
    Returns [{start, end, text, duration}, ...]
    """
    if not segments:
        return []

    if mode == "interval":
        scenes = []
        buf_start = segments[0]["start"]
        buf_texts = []
        buf_end = segments[0]["start"]

        for seg in segments:
            buf_texts.append(seg["text"])
            buf_end = seg["end"]
            span = buf_end - buf_start
            if span >= interval_secs:
                scenes.append({"start": buf_start, "end": buf_end,
                                "text": " ".join(buf_texts), "duration": round(span, 1)})
                buf_start = buf_end
                buf_texts = []

        if buf_texts:
            span = buf_end - buf_start
            scenes.append({"start": buf_start, "end": buf_end,
                            "text": " ".join(buf_texts),
                            "duration": round(max(span, interval_secs), 2)})

    elif mode == "paragraph":
        scenes = []
        buf = [segments[0]]
        for seg in segments[1:]:
            gap = seg["start"] - buf[-1]["end"]
            if gap > 0.8:  # natural pause
                scenes.append(_merge_segs(buf))
                buf = [seg]
            else:
                buf.append(seg)
        if buf:
            scenes.append(_merge_segs(buf))

    else:  # sentence
        full_text = " ".join(s["text"] for s in segments)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        # Distribute timestamps proportionally
        total = segments[-1]["end"] - segments[0]["start"]
        total_chars = max(len(full_text), 1)
        scenes = []
        elapsed = segments[0]["start"]
        for sent in sentences:
            dur = total * len(sent) / total_chars
            scenes.append({"start": elapsed, "end": elapsed + dur,
                            "text": sent, "duration": round(dur, 1)})
            elapsed += dur

    # Cap to max_scenes
    if len(scenes) > max_scenes:
        # Merge tail scenes
        keep = scenes[:max_scenes - 1]
        tail = scenes[max_scenes - 1:]
        merged = {"start": tail[0]["start"], "end": tail[-1]["end"],
                  "text": " ".join(s["text"] for s in tail),
                  "duration": sum(s["duration"] for s in tail)}
        keep.append(merged)
        scenes = keep

    return scenes


def _merge_segs(segs: list[dict]) -> dict:
    return {"start": segs[0]["start"], "end": segs[-1]["end"],
            "text": " ".join(s["text"] for s in segs),
            "duration": round(segs[-1]["end"] - segs[0]["start"], 1)}


# ── Prompt enhancement ────────────────────────────────────────────────────────

def enhance_prompts_gemini(scenes: list[dict], topic: str, api_key: str,
                             style: str = "cinematic") -> list[dict]:
    """Use Gemini to generate scene descriptions, then wrap in style template."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    scene_texts = "\n".join(
        f"{i+1}. [{s['duration']}s] {s['text']}" for i, s in enumerate(scenes)
    )

    # Pick the right instruction for this style
    instruction_tmpl = _STYLE_GEMINI_INSTRUCTION.get(style, _DEFAULT_GEMINI_INSTRUCTION)
    gemini_prompt = instruction_tmpl.format(topic=topic, scene_texts=scene_texts)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
        )
        lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        raw_descs = []
        for line in lines:
            m = re.match(r'^\d+[.)]\s*(.*)', line)
            if m:
                raw_descs.append(m.group(1).strip())

        if len(raw_descs) == len(scenes):
            for i, s in enumerate(scenes):
                # For template styles, wrap Gemini's scene description in the template
                s["prompt"] = build_scene_prompt(
                    raw_descs[i], style,
                    s.get("start", 0), s.get("end", s.get("start", 0) + s.get("duration", 5))
                )
            return scenes
    except Exception as e:
        print(f"[scene_gen] Gemini prompt enhancement failed: {e}", flush=True)

    # Fallback: build prompt directly from transcript text
    for s in scenes:
        s["prompt"] = build_scene_prompt(
            s["text"], style,
            s.get("start", 0), s.get("end", s.get("start", 0) + s.get("duration", 5))
        )
    return scenes


# ── HuggingFace image generation ─────────────────────────────────────────────

HF_API_BASE = "https://api-inference.huggingface.co/models"


# ── Gemini image generation ───────────────────────────────────────────────────

_GEMINI_API_MODEL_IDS = {
    "gemini_flash_img": "gemini-2.0-flash-preview-image-generation",
    "gemini_imagen_3":  "imagen-3.0-generate-002",
}


def generate_image_gemini(
    prompt: str,
    model_key: str,
    gemini_key: str,
    output_dir: Path | None = None,
) -> str:
    """Generate image via Gemini API (Flash Image or Imagen 3). Returns saved file path."""
    if not gemini_key:
        raise RuntimeError("Gemini API key is required for Gemini image generation")

    from google import genai
    from google.genai import types
    import uuid as _uuid

    api_model = _GEMINI_API_MODEL_IDS.get(model_key, "gemini-2.0-flash-preview-image-generation")
    client = genai.Client(api_key=gemini_key)

    save_dir = output_dir or CACHE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / f"gemini_{_uuid.uuid4().hex[:8]}.png"

    if "imagen" in api_model:
        response = client.models.generate_images(
            model=api_model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
            ),
        )
        img_bytes = response.generated_images[0].image.image_bytes
    else:
        # gemini-2.0-flash-preview-image-generation
        response = client.models.generate_content(
            model=api_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )
        img_bytes = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                img_bytes = part.inline_data.data
                break
        if not img_bytes:
            raise RuntimeError("Gemini returned no image data")

    img_path.write_bytes(img_bytes)
    return str(img_path)


def generate_image_hf(prompt: str, model_id: str, hf_token: str,
                       width: int = 1024, height: int = 576,
                       output_dir: Path | None = None) -> str:
    """Generates one image, saves to output_dir (or CACHE_DIR), returns file path."""
    url = f"{HF_API_BASE}/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": width, "height": height}
    }

    for attempt in range(3):
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 503:
            wait = int(r.json().get("estimated_time", 20))
            print(f"[scene_gen] Model loading, wait {wait}s…", flush=True)
            time.sleep(min(wait, 30))
            continue
        if r.status_code == 200:
            break
        raise RuntimeError(f"HF API error {r.status_code}: {r.text[:200]}")

    r.raise_for_status()

    save_dir = output_dir or CACHE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    img_id = uuid.uuid4().hex[:8]
    path = save_dir / f"scene_{img_id}.png"
    path.write_bytes(r.content)
    return str(path)


def generate_all_scenes(
    scenes: list[dict],
    model_id: str,
    hf_token: str,
    progress_cb=None,
    width: int = 1024,
    height: int = 576,
) -> list[dict]:
    """Generate one image per scene, updates scene dict with 'img_path'."""
    total = len(scenes)
    for i, scene in enumerate(scenes):
        if progress_cb:
            progress_cb(f"Generating image {i+1}/{total}…", i + 1, total)
        prompt = scene.get("prompt", scene.get("text", "")[:80])
        try:
            scene["img_path"] = generate_image_hf(prompt, model_id, hf_token, width, height)
        except Exception as e:
            print(f"[scene_gen] Image {i+1} failed: {e}", flush=True)
            scene["img_path"] = None
    return scenes
