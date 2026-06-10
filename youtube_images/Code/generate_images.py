import os
import re
import sys
import requests
import base64
import time
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OUTPUT_DIR = r"C:\Users\nisha\OneDrive\Desktop\Youtube\youtube_images"
IMAGE_MODEL = "imagen-4.0-fast-generate-001"   # swap to imagen-4.0-generate-001 for higher quality
TEXT_MODEL  = "gemini-2.5-flash"

STYLE_PREFIX = (
    "MS Paint style drawing, extremely simple beginner art, white background, "
    "thick uneven black outlines, wobbly hand-drawn lines, stick figure humans "
    "with round circle heads and straight line bodies, simple dot eyes, "
    "basic facial expressions, flat solid colors only, no shading, no 3D effects, "
    "childish drawing, looks like a 5-year-old drew it in MS Paint: "
)
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── STEP 1: find transcript file ──────────────────────────────────────────────
def find_transcript(folder):
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            return os.path.join(folder, f)
    return None

script_dir = os.path.dirname(os.path.abspath(__file__))
transcript_path = find_transcript(script_dir)

if not transcript_path:
    print("ERROR: No .txt transcript file found in", script_dir)
    print("Place your transcript .txt file in the same folder as this script and run again.")
    sys.exit(1)

print(f"Reading transcript: {transcript_path}")
with open(transcript_path, "r", encoding="utf-8") as f:
    transcript_text = f.read()


# ── STEP 2: parse timestamps ──────────────────────────────────────────────────
# Matches patterns like (0:00) or (1:23) or 0:00 or 1:23
pattern = re.compile(r'\((\d+:\d+)\)(.*?)(?=\(\d+:\d+\)|$)', re.DOTALL)
matches = pattern.findall(transcript_text)

if not matches:
    print("ERROR: No timestamps found. Make sure your transcript uses format like (0:00) Text here.")
    sys.exit(1)

timestamps = []
for ts, text in matches:
    text = text.strip().replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    if text:
        timestamps.append((ts, text))

print(f"Found {len(timestamps)} timestamps.\n")


# ── STEP 3: generate image descriptions via Gemini text model ─────────────────
def get_image_description(timestamp, narration_text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{TEXT_MODEL}:generateContent?key={GEMINI_API_KEY}"

    system_prompt = (
        "You are helping create visuals for a YouTube video. "
        "For each line of narration, write a SHORT image description (max 25 words) "
        "for a simple figure drawing that visually illustrates what the narrator is saying. "
        "Only output the description, nothing else. No quotes, no explanation."
    )

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": f"Timestamp {timestamp}: {narration_text}"}]}]
    }

    r = requests.post(url, json=payload)
    if r.status_code == 200:
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    else:
        # Fallback: use narration text directly
        return narration_text[:120]


# ── STEP 4: generate image via Imagen ─────────────────────────────────────────
def generate_image(index, timestamp, description):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL}:predict?key={GEMINI_API_KEY}"

    prompt = STYLE_PREFIX + description

    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "16:9"}
    }

    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

    if r.status_code == 200:
        data = r.json()
        try:
            img_b64 = data["predictions"][0]["bytesBase64Encoded"]
            safe_ts = timestamp.replace(":", "_")
            img_path = os.path.join(OUTPUT_DIR, f"{index:03d}_{safe_ts}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            print(f"  [OK] saved -> {os.path.basename(img_path)}")
            return True
        except (KeyError, IndexError) as e:
            print(f"  [ERROR] parse error: {e}")
            return False
    else:
        print(f"  [ERROR] HTTP {r.status_code}: {r.text[:200]}")
        return False


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print(f"Generating {len(timestamps)} images into: {OUTPUT_DIR}\n")
success = 0
fail = 0

for i, (ts, narration) in enumerate(timestamps):
    print(f"[{i+1}/{len(timestamps)}] {ts} — {narration[:60]}...")

    # Get a focused image description
    description = get_image_description(ts, narration)
    print(f"  Prompt: {description[:80]}")

    # Generate the image
    ok = generate_image(i + 1, ts, description)
    if ok:
        success += 1
    else:
        fail += 1

    time.sleep(0.5)  # avoid rate limiting

print(f"\nDone! {success} succeeded, {fail} failed.")
print(f"Images saved to: {OUTPUT_DIR}")
