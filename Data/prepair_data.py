import json
import random
import wave
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset

# reazon-research/reazonspeech is a script-based dataset: needs datasets<4.0
# and an accepted gate on the Hub.
DATASET = "reazon-research/reazonspeech"
CONFIG = "tiny"
SPLIT = "train"
TEXT_KEYS = ("transcription", "text", "sentence")

SAMPLE_RATE = 16000
MAX_SECONDS  = 30.0     # Whisper's encoder window; longer clips are skipped
DEV_SIZE = 300
TEST_SIZE = 300
SEED = 0

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
MANIFEST_DIR = ROOT / "manifests"


def find_text_key(row):
    for key in TEXT_KEYS:
        if key in row:
            return key
    raise KeyError(f"no transcript column in {sorted(row)}; add it to TEXT_KEYS")


def write_wav(path, samples):
    """Write float samples in [-1, 1] as 16-bit mono PCM."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def build_rows():
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, trust_remote_code=True)
    # Audio(sampling_rate=...) makes datasets resample on decode.
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    rows, skipped = [], 0
    for i, row in enumerate(ds):
        text = row[find_text_key(row)].strip()
        samples = np.asarray(row["audio"]["array"], dtype=np.float32)
        duration = len(samples) / SAMPLE_RATE

        if not text or duration == 0 or duration > MAX_SECONDS:
            skipped += 1
            continue

        path = AUDIO_DIR / f"{i:06d}.wav"
        write_wav(path, samples)
        rows.append({"audio": str(path), "text": text, "duration": round(duration, 3)})

    print(f"kept {len(rows)} clips, skipped {skipped}")
    return rows


def write_manifest(name, rows):
    path = MANIFEST_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{path}: {len(rows)} clips, {hours:.2f} h")


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    if len(rows) <= DEV_SIZE + TEST_SIZE:
        raise SystemExit(f"only {len(rows)} clips; lower DEV_SIZE/TEST_SIZE")

    random.Random(SEED).shuffle(rows)
    write_manifest("dev", rows[:DEV_SIZE])
    write_manifest("test", rows[DEV_SIZE:DEV_SIZE + TEST_SIZE])
    write_manifest("train", rows[DEV_SIZE + TEST_SIZE:])

if __name__ == "__main__":
    main()
