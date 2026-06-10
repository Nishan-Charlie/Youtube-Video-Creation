import os
import time
import requests
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("HF_TOKEN", "")
CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--ResembleAI--chatterbox" / "direct"
CACHE.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/ResembleAI/chatterbox/resolve/main"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

FILES = [
    "ve.safetensors",
    "s3gen.safetensors",
    "t3_cfg.safetensors",
    "t3_23lang.safetensors",
    "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v3.safetensors",
    "conds.pt",
    "tokenizer.json",
    "Cangjie5_TC.json",
    "grapheme_mtl_merged_expanded_v1.json",
    "mtl_tokenizer.json",
]

def download_file(filename):
    dest = CACHE / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [SKIP] {filename} already downloaded ({dest.stat().st_size // 1024**2} MB)", flush=True)
        return True

    url = f"{BASE_URL}/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        total_mb = total / 1024**2

        print(f"  Downloading {filename} ({total_mb:.1f} MB)...", flush=True)

        downloaded = 0
        last_print = time.time()
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 5:
                        pct = downloaded / total * 100 if total else 0
                        speed = downloaded / (now - last_print) / 1024 / 1024
                        print(f"    {downloaded//1024**2}/{total_mb:.0f} MB  ({pct:.0f}%)  {speed:.1f} MB/s", flush=True)
                        last_print = now

        tmp.rename(dest)
        print(f"  [DONE] {filename} ({dest.stat().st_size // 1024**2} MB)", flush=True)
        return True

    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  [FAIL] {filename}: {e}", flush=True)
        return False

print(f"Saving to: {CACHE}", flush=True)
print(f"HF_TOKEN: {'set' if TOKEN else 'NOT SET'}", flush=True)
print("", flush=True)

ok = 0
for i, f in enumerate(FILES, 1):
    print(f"[{i}/{len(FILES)}] {f}", flush=True)
    if download_file(f):
        ok += 1

print(f"\nComplete: {ok}/{len(FILES)} files in {CACHE}", flush=True)
