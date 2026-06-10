import re
import time
from pathlib import Path

from google import genai
from google.genai import types

# ── OpenAI-compatible providers (Qwen DashScope, Ollama, LM Studio, etc.) ─────

OPENAI_PROVIDERS = {
    "qwen_dashscope": {
        "name": "Qwen (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct", "qwen2.5-14b-instruct"],
        "key_hint": "Get key at dashscope.aliyuncs.com",
    },
    "ollama": {
        "name": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "models": ["qwen2.5:7b", "qwen2.5:14b", "llama3.2", "mistral", "gemma2"],
        "key_hint": "No key needed. Install Ollama + run: ollama pull qwen2.5:7b",
    },
    "lmstudio": {
        "name": "LM Studio (local)",
        "base_url": "http://localhost:1234/v1",
        "models": ["local-model"],
        "key_hint": "No key needed. Start LM Studio server first.",
    },
}


def generate_script_openai_compat(
    topic: str, niche: str, duration: str, tone: str, context: str,
    api_key: str, model: str, base_url: str
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
    context_block = f"\n\nAdditional context:\n{context}" if context.strip() else ""

    prompt = f"""You are an expert YouTube scriptwriter. Write a complete voiceover script.

Topic: {topic}
Niche: {niche}
Duration: {duration}
Tone: {tone}{context_block}

Rules:
- Natural spoken English only — no stage directions
- Start with a powerful hook
- Use [INTRO] [MAIN CONTENT] [OUTRO] section headers

Write the full script now."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def check_ollama() -> list[str]:
    """Return list of installed Ollama models, empty if Ollama not running."""
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# Persist the last working model across restarts so we skip already-quota'd ones
_CACHE_FILE = Path(__file__).parent / ".last_model"
_DAILY_QUOTA_KEY = Path(__file__).parent / ".quota_date"


def _load_cached_model() -> str | None:
    """Return the last model that worked, if it was cached today."""
    try:
        today = time.strftime("%Y-%m-%d")
        if _DAILY_QUOTA_KEY.exists() and _DAILY_QUOTA_KEY.read_text().strip() == today:
            # Full daily quota hit was recorded today — skip directly to flash-lite/flash
            return "gemini-2.5-flash-lite"
        if _CACHE_FILE.exists():
            return _CACHE_FILE.read_text().strip() or None
    except Exception:
        pass
    return None


def _save_cached_model(model: str) -> None:
    try:
        _CACHE_FILE.write_text(model)
    except Exception:
        pass


def _record_daily_quota_exhausted() -> None:
    try:
        _DAILY_QUOTA_KEY.write_text(time.strftime("%Y-%m-%d"))
    except Exception:
        pass


def _parse_retry_delay(error_str: str) -> float:
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)", error_str, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 2, 65)
    return 30.0


def generate_script(
    topic: str,
    niche: str,
    duration: str,
    tone: str,
    context: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> str:
    client = genai.Client(api_key=api_key)

    context_block = f"\n\nAdditional context from creator:\n{context}" if context.strip() else ""

    prompt = f"""You are an expert YouTube scriptwriter with 10+ years of experience creating viral content.

Write a complete, ready-to-record YouTube voiceover script with these specifications:

**Topic:** {topic}
**Niche / Category:** {niche}
**Target Video Length:** {duration}
**Tone & Style:** {tone}{context_block}

**Script Requirements:**
- Write in natural spoken English — exactly as it should be read aloud
- No stage directions, no [pause here], no camera instructions — pure narration text only
- Start with a powerful hook in the first 15 seconds that creates curiosity
- Use short, punchy sentences mixed with longer ones for rhythm
- Include smooth transitions between sections
- End with a clear, compelling call-to-action

**Structure:**
[INTRO]
(Hook + topic introduction — ~10% of total)

[MAIN CONTENT]
(Core information in 3-5 clear sections — ~80% of total)

[OUTRO]
(Summary + CTA — ~10% of total)

Write the full script now. Keep the section headers exactly as shown above."""

    # Start from cached working model to skip already-exhausted ones
    cached = _load_cached_model()
    if cached and cached != model:
        models_to_try = [cached] + [m for m in MODELS if m != cached and m != model] + [model]
    else:
        models_to_try = [model] + [m for m in MODELS if m != model]

    last_error = None
    daily_quota_hits = 0

    for attempt_model in models_to_try:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                print(f"[script] Using {attempt_model}", flush=True)
                response = client.models.generate_content(
                    model=attempt_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        max_output_tokens=8192,
                    ),
                )
                _save_cached_model(attempt_model)
                print(f"[script] Done ({attempt_model})", flush=True)
                return response.text

            except Exception as e:
                err_str = str(e)
                last_error = e

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if "PerDay" in err_str or "per_day" in err_str.lower():
                        daily_quota_hits += 1
                        print(f"[script] Quota full for {attempt_model}, trying next…", flush=True)
                        break
                    delay = _parse_retry_delay(err_str)
                    print(f"[script] Rate limited, waiting {delay:.0f}s…", flush=True)
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue
                    break

                elif "API_KEY" in err_str or "invalid" in err_str.lower():
                    raise ValueError("Invalid Gemini API key. Check your key at aistudio.google.com") from e
                else:
                    raise

    if daily_quota_hits == len(models_to_try):
        _record_daily_quota_exhausted()

    err_msg = str(last_error)
    if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
        raise RuntimeError(
            "Gemini free-tier quota exhausted for all models. "
            "Options: (1) wait until tomorrow for the daily limit to reset, "
            "(2) upgrade to a paid Gemini API plan at aistudio.google.com."
        )
    raise last_error
