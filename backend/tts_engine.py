"""
Chatterbox TTS engine wrapper.
Handles chunking, model caching, and audio merging.
"""
import re
import torch
import torchaudio

_model = None
_device = None


def _get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"[tts] GPU: {name} ({vram} MB VRAM) — using CUDA", flush=True)
        return "cuda"
    print("[tts] No CUDA detected — falling back to CPU", flush=True)
    return "cpu"


def _local_model_path():
    from pathlib import Path
    direct = Path.home() / ".cache" / "huggingface" / "hub" / "models--ResembleAI--chatterbox" / "direct"
    required = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
    if direct.exists() and all((direct / f).exists() for f in required):
        return direct
    return None


def _patch_perth():
    """If PerthImplicitWatermarker failed to load (native ext missing), use the built-in DummyWatermarker."""
    import perth
    if perth.PerthImplicitWatermarker is None:
        print("[tts] resemble-perth native ext not available — using DummyWatermarker", flush=True)
        perth.PerthImplicitWatermarker = perth.DummyWatermarker


def _patch_safetensors_no_mmap():
    """Monkey-patch safetensors.torch.load_file to avoid mmap (fixes Windows paging file error)."""
    import safetensors.torch as st_torch
    import safetensors

    original_load_file = st_torch.load_file

    def load_file_no_mmap(filename, device="cpu"):
        try:
            return original_load_file(filename, device=device)
        except OSError:
            # Fall back to reading bytes directly — avoids Windows mmap paging error
            print(f"[tts] mmap failed for {filename}, loading via bytes…", flush=True)
            with open(filename, "rb") as f:
                data = f.read()
            tensors = safetensors.torch.load(data)
            if device not in ("cpu", None):
                tensors = {k: v.to(device) for k, v in tensors.items()}
            return tensors

    st_torch.load_file = load_file_no_mmap
    # Also patch inside chatterbox module if already imported
    try:
        import chatterbox.tts as cb_tts
        cb_tts.load_file = load_file_no_mmap
    except Exception:
        pass


def _load_model():
    global _model, _device
    if _model is not None:
        return _model

    from chatterbox.tts import ChatterboxTTS

    _device = _get_device()
    _patch_perth()
    _patch_safetensors_no_mmap()
    local_path = _local_model_path()

    if local_path:
        print(f"[tts] Loading from local path: {local_path}", flush=True)
        _model = ChatterboxTTS.from_local(local_path, _device)
    else:
        print(f"[tts] Downloading model from HuggingFace…", flush=True)
        _model = ChatterboxTTS.from_pretrained(device=_device)

    print(f"[tts] Model loaded on {_device}.", flush=True)
    return _model


def warmup_model():
    try:
        _load_model()
    except Exception as e:
        print(f"[tts] Warmup failed: {e}", flush=True)


def _split_text(text: str, mode: str) -> list[str]:
    """Split text into chunks by mode."""
    text = text.strip()

    if mode == "full":
        return [text]

    if mode == "paragraph":
        chunks = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return chunks if chunks else [text]

    if mode == "sentence":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Group short sentences (< 60 chars) with the next one
        groups = []
        buf = ""
        for s in sentences:
            buf = (buf + " " + s).strip() if buf else s
            if len(buf) >= 60:
                groups.append(buf)
                buf = ""
        if buf:
            groups.append(buf)
        return groups if groups else [text]

    return [text]


def generate_audio(
    text: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    voice_ref_path: str | None = None,
    chunk_mode: str = "paragraph",
    speed: float = 1.0,
    progress_cb=None,
) -> None:
    def _cb(msg: str, chunk: int = 0, total: int = 0):
        print(f"[tts] {msg}", flush=True)
        if progress_cb:
            progress_cb(msg, chunk, total)

    _cb("Loading TTS model on GPU…")
    model = _load_model()

    chunks = _split_text(text, chunk_mode)
    total = len(chunks)
    _cb(f"Starting synthesis — {total} chunk(s)", 0, total)

    all_wavs = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        preview = chunk[:55] + "…" if len(chunk) > 55 else chunk
        _cb(f"Synthesizing chunk {i+1}/{total}: {preview}", i + 1, total)

        wav = model.generate(
            chunk,
            audio_prompt_path=voice_ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        all_wavs.append(wav)

        if i < total - 1:
            silence_samples = int(0.3 * model.sr)
            all_wavs.append(torch.zeros(1, silence_samples))

    if not all_wavs:
        raise ValueError("No audio was generated")

    _cb("Merging audio segments…", total, total)
    combined = torch.cat(all_wavs, dim=1)

    # Speed adjustment (pitch-preserved time stretching via librosa)
    if speed != 1.0:
        import librosa, numpy as np
        _cb(f"Adjusting speed to {speed}x…", total, total)
        wav_np = combined.squeeze().cpu().numpy()
        wav_np = librosa.effects.time_stretch(wav_np, rate=float(speed))
        combined = torch.from_numpy(wav_np).unsqueeze(0)

    torchaudio.save(output_path, combined.cpu(), model.sr)
    _cb(f"Done — saved to {output_path}", total, total)
