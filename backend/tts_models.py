"""
TTS model registry — Chatterbox, Kokoro, XTTS v2
All models share a common generate() interface.
"""
import re
import warnings
import torch
import torchaudio
from pathlib import Path

# Suppress harmless model-internal warnings
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*xFormers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

# ── Gemini TTS voice list ─────────────────────────────────────────────────────

GEMINI_VOICES = {
    "Aoede (Female)":   "Aoede",
    "Kore (Female)":    "Kore",
    "Zephyr (Female)":  "Zephyr",
    "Vale (Female)":    "Vale",
    "Sage (Female)":    "Sage",
    "Charon (Male)":    "Charon",
    "Fenrir (Male)":    "Fenrir",
    "Puck (Male)":      "Puck",
    "Orbit (Male)":     "Orbit",
    "Luca (Male)":      "Luca",
}

# ── Model registry ────────────────────────────────────────────────────────────

REGISTRY = {
    "chatterbox": {
        "name": "Chatterbox TTS",
        "author": "Resemble AI",
        "vram": "~6 GB",
        "description": "Expressive voice with emotion control. Best for dramatic/storytelling content.",
        "supports_clone": True,
        "voices": {"Default": "default"},
    },
    "kokoro": {
        "name": "Kokoro 82M",
        "author": "hexgrad",
        "vram": "~0.3 GB",
        "description": "Ultra-fast, 11 built-in voices. Best for high-volume or quick generation.",
        "supports_clone": False,
        "voices": {
            "Heart — US Female (warm)":    "af_heart",
            "Bella — US Female (bright)":  "af_bella",
            "Nicole — US Female (clear)":  "af_nicole",
            "Sarah — US Female (soft)":    "af_sarah",
            "Sky — US Female (airy)":      "af_sky",
            "Adam — US Male (deep)":       "am_adam",
            "Michael — US Male (firm)":    "am_michael",
            "Emma — UK Female (crisp)":    "bf_emma",
            "Isabella — UK Female":        "bf_isabella",
            "George — UK Male (rich)":     "bm_george",
            "Lewis — UK Male (smooth)":    "bm_lewis",
        },
    },
    "xtts2": {
        "name": "XTTS v2",
        "author": "Coqui AI",
        "vram": "~2 GB",
        "description": "Studio-quality voice cloning. Requires a reference WAV (5-30s) to generate.",
        "supports_clone": True,
        "voices": {"Clone from reference WAV": "clone"},
    },
    "gemini_tts": {
        "name": "Gemini TTS",
        "author": "Google",
        "vram": "Cloud API",
        "description": "Google Gemini's native voice synthesis. Uses your Gemini API key — no local GPU needed. 10 voices.",
        "supports_clone": False,
        "voices": GEMINI_VOICES,
        "requires_api_key": True,
    },
}

# ── Cached model instances ────────────────────────────────────────────────────

_loaded: dict = {}


def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"[tts] Using CUDA: {name} ({vram_mb} MB)", flush=True)
        return "cuda"
    return "cpu"


# ── Chatterbox ────────────────────────────────────────────────────────────────

def _load_chatterbox(device: str):
    if "chatterbox" in _loaded:
        return _loaded["chatterbox"]

    from chatterbox.tts import ChatterboxTTS
    import perth, safetensors.torch as st

    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    orig = st.load_file
    def _safe_load(filename, device_="cpu"):
        try:
            return orig(filename, device=device_)
        except OSError:
            import safetensors
            with open(filename, "rb") as f:
                data = f.read()
            t = safetensors.torch.load(data)
            if device_ not in ("cpu", None):
                t = {k: v.to(device_) for k, v in t.items()}
            return t
    st.load_file = _safe_load
    try:
        import chatterbox.tts as _ct
        _ct.load_file = _safe_load
    except Exception:
        pass

    direct = Path.home() / ".cache/huggingface/hub/models--ResembleAI--chatterbox/direct"
    required = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
    if direct.exists() and all((direct / f).exists() for f in required):
        print(f"[chatterbox] Loading from {direct}", flush=True)
        model = ChatterboxTTS.from_local(direct, device)
    else:
        print("[chatterbox] Downloading from HuggingFace…", flush=True)
        model = ChatterboxTTS.from_pretrained(device=device)

    _loaded["chatterbox"] = model
    print(f"[chatterbox] Loaded on {device}", flush=True)
    return model


def generate_chatterbox(text: str, output_path: str, voice: str = "default",
                         exaggeration: float = 0.5, cfg_weight: float = 0.5,
                         speed: float = 1.0, voice_ref: str | None = None,
                         chunk_mode: str = "paragraph", progress_cb=None) -> None:
    device = get_device()
    model = _load_chatterbox(device)
    chunks = _split(text, chunk_mode)
    total = len(chunks)
    _cb(progress_cb, "Chatterbox loaded — synthesizing…", 0, total)

    wavs = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        _cb(progress_cb, f"Synthesizing chunk {i+1}/{total}: {chunk[:50]}…", i + 1, total)
        wav = model.generate(chunk, audio_prompt_path=voice_ref,
                             exaggeration=exaggeration, cfg_weight=cfg_weight)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wavs.append(wav)
        if i < total - 1:
            wavs.append(torch.zeros(1, int(0.3 * model.sr)))

    combined = torch.cat(wavs, dim=1)
    combined = _apply_speed(combined, model.sr, speed, progress_cb)
    torchaudio.save(output_path, combined.cpu(), model.sr)
    _cb(progress_cb, "Done", total, total)


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def _load_kokoro(device: str):
    if "kokoro" in _loaded:
        return _loaded["kokoro"]
    from kokoro import KPipeline
    print(f"[kokoro] Loading pipeline…", flush=True)
    pipeline = KPipeline(lang_code='a', device=device)
    _loaded["kokoro"] = pipeline
    print(f"[kokoro] Loaded on {device}", flush=True)
    return pipeline


def generate_kokoro(text: str, output_path: str, voice: str = "af_heart",
                     speed: float = 1.0, chunk_mode: str = "paragraph",
                     progress_cb=None, **kwargs) -> None:
    import numpy as np
    import soundfile as sf

    device = get_device()
    pipeline = _load_kokoro(device)

    _cb(progress_cb, "Kokoro loaded — synthesizing…", 0, 1)
    all_audio = []
    chunks = _split(text, chunk_mode)
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        _cb(progress_cb, f"Kokoro chunk {i+1}/{total}: {chunk[:50]}…", i + 1, total)
        generator = pipeline(chunk, voice=voice, speed=speed)
        for _, _, audio in generator:
            all_audio.append(audio)

    if not all_audio:
        raise ValueError("No audio generated")

    combined = np.concatenate(all_audio, axis=0)
    sf.write(output_path, combined, 24000)
    _cb(progress_cb, "Done", total, total)


# ── XTTS v2 ──────────────────────────────────────────────────────────────────

def _load_xtts2(device: str):
    if "xtts2" in _loaded:
        return _loaded["xtts2"]
    from TTS.api import TTS
    print("[xtts2] Loading model…", flush=True)
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    _loaded["xtts2"] = model
    print(f"[xtts2] Loaded on {device}", flush=True)
    return model


def generate_xtts2(text: str, output_path: str, voice: str = "clone",
                    speed: float = 1.0, voice_ref: str | None = None,
                    chunk_mode: str = "paragraph", progress_cb=None, **kwargs) -> None:
    if not voice_ref:
        raise ValueError("XTTS v2 requires a Voice Reference WAV file. "
                         "Browse a 5–30s WAV in the Voice Settings.")

    device = get_device()
    model = _load_xtts2(device)

    chunks = _split(text, chunk_mode)
    total = len(chunks)
    import soundfile as sf, numpy as np

    all_audio = []
    sr = 24000

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        _cb(progress_cb, f"XTTS chunk {i+1}/{total}: {chunk[:50]}…", i + 1, total)
        tmp_path = output_path + f".chunk{i}.wav"
        model.tts_to_file(text=chunk, speaker_wav=voice_ref, language="en", file_path=tmp_path)
        audio, sr = sf.read(tmp_path)
        all_audio.append(audio)
        Path(tmp_path).unlink(missing_ok=True)

    combined_np = np.concatenate(all_audio, axis=0)
    wav = torch.from_numpy(combined_np).float().unsqueeze(0)
    wav = _apply_speed(wav, sr, speed, progress_cb)
    torchaudio.save(output_path, wav.cpu(), sr)
    _cb(progress_cb, "Done", total, total)


# ── Gemini TTS ────────────────────────────────────────────────────────────────

def generate_gemini_tts(text: str, output_path: str, voice: str = "Aoede",
                         speed: float = 1.0, chunk_mode: str = "paragraph",
                         gemini_api_key: str = "", progress_cb=None, **kwargs) -> None:
    if not gemini_api_key:
        raise ValueError("Gemini TTS requires your Gemini API key. Enter it in the left sidebar.")

    import base64, struct, wave
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key)
    chunks = _split(text, chunk_mode)
    total = len(chunks)
    all_pcm = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        _cb(progress_cb, f"Gemini TTS chunk {i+1}/{total}: {chunk[:50]}…", i + 1, total)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=chunk,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    )
                )
            )
            raw = response.candidates[0].content.parts[0].inline_data.data
            # Gemini returns base64-encoded PCM (24000 Hz, 16-bit, mono)
            pcm = base64.b64decode(raw) if isinstance(raw, str) else raw
            all_pcm.append(pcm)
        except Exception as e:
            raise RuntimeError(f"Gemini TTS error on chunk {i+1}: {e}") from e

    if not all_pcm:
        raise ValueError("No audio generated")

    # Add 300ms silence between chunks
    silence_300ms = bytes(int(24000 * 0.3) * 2)  # 16-bit mono
    combined_pcm = silence_300ms.join(all_pcm)

    # Write PCM → WAV
    _cb(progress_cb, "Saving audio…", total, total)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(24000)
        wf.writeframes(combined_pcm)

    # Apply speed if needed
    if speed != 1.0:
        wav_tensor, sr_val = torchaudio.load(output_path)
        wav_tensor = _apply_speed(wav_tensor, sr_val, speed, progress_cb)
        torchaudio.save(output_path, wav_tensor.cpu(), sr_val)

    _cb(progress_cb, "Done", total, total)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def generate(model_id: str, text: str, output_path: str, **kwargs) -> None:
    if model_id == "chatterbox":
        generate_chatterbox(text, output_path, **kwargs)
    elif model_id == "kokoro":
        generate_kokoro(text, output_path, **kwargs)
    elif model_id == "xtts2":
        generate_xtts2(text, output_path, **kwargs)
    elif model_id == "gemini_tts":
        generate_gemini_tts(text, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_id}")


def warmup(model_id: str = "chatterbox") -> None:
    try:
        device = get_device()
        if model_id == "chatterbox":
            _load_chatterbox(device)
        elif model_id == "kokoro":
            _load_kokoro(device)
    except Exception as e:
        print(f"[warmup] {model_id}: {e}", flush=True)


def is_installed(model_id: str) -> bool:
    try:
        if model_id == "chatterbox":
            import chatterbox; return True
        elif model_id == "kokoro":
            import kokoro; return True
        elif model_id == "xtts2":
            import TTS; return True
        elif model_id == "gemini_tts":
            from google import genai; return True  # noqa: F401
    except ImportError:
        return False
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(text: str, mode: str) -> list[str]:
    text = text.strip()
    if mode == "full":
        return [text]
    if mode == "paragraph":
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return parts or [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    groups, buf = [], ""
    for s in sentences:
        buf = (buf + " " + s).strip() if buf else s
        if len(buf) >= 60:
            groups.append(buf); buf = ""
    if buf: groups.append(buf)
    return groups or [text]


def _cb(fn, msg: str, chunk: int, total: int):
    print(f"[tts] {msg}", flush=True)
    if fn: fn(msg, chunk, total)


def _apply_speed(wav: torch.Tensor, sr: int, speed: float, progress_cb=None) -> torch.Tensor:
    if speed == 1.0:
        return wav
    import librosa, numpy as np
    _cb(progress_cb, f"Adjusting speed to {speed}x…", 0, 0)
    wav_np = wav.squeeze().cpu().numpy()
    wav_np = librosa.effects.time_stretch(wav_np, rate=float(speed))
    return torch.from_numpy(wav_np).unsqueeze(0)
