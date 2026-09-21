import json
import wave
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset

# Adds extra TRAIN-only clips from the ReazonSpeech `small` config. Never touches
# dev/test and never reshuffles: writes a separate manifest + audio dir.
DATASET = "reazon-research/reazonspeech"
CONFIG = "small"
SPLIT = "train"
TEXT_KEYS = ("transcription", "text", "sentence")

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0      # Whisper's encoder window; longer clips are skipped
TARGET_HOURS = 8.0      # 8h extra on top of tiny's ~8h => the "2x" run; 24 => "4x"

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio_small"
MANIFEST_DIR = ROOT / "manifests"
OUT_MANIFEST = MANIFEST_DIR / "train_extra.jsonl"
EXISTING = ("train", "dev", "test")


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


def load_seen_texts():
    """Transcripts already in train/dev/test, so new clips can't leak into dev/test."""
    seen = set()
    for name in EXISTING:
        with open(MANIFEST_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["text"])
    return seen


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen_texts()
    print(f"{len(seen)} transcripts already in train/dev/test")

    # streaming=True: iterate lazily instead of downloading all ~100h up front.
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    target_seconds = TARGET_HOURS * 3600
    rows, total, skipped, dupes = [], 0.0, 0, 0
    for i, row in enumerate(ds):
        text = row[find_text_key(row)].strip()
        samples = np.asarray(row["audio"]["array"], dtype=np.float32)
        duration = len(samples) / SAMPLE_RATE

        if not text or duration == 0 or duration > MAX_SECONDS:
            skipped += 1
            continue
        if text in seen:
            dupes += 1
            continue
        seen.add(text)

        path = AUDIO_DIR / f"small_{i:07d}.wav"
        write_wav(path, samples)
        rows.append({"audio": str(path), "text": text, "duration": round(duration, 3)})
        total += duration

        if len(rows) % 500 == 0:
            print(f"{len(rows)} clips, {total / 3600:.2f} h")
        if total >= target_seconds:
            break

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{OUT_MANIFEST}: {len(rows)} clips, {total / 3600:.2f} h "
          f"(skipped {skipped}, duplicate transcripts {dupes})")


if __name__ == "__main__":
    main()
