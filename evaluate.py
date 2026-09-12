import json, sys, statistics, torch, whisper
from whisper.normalizers import BasicTextNormalizer
import jiwer

DEV_MANIFEST = "Data/manifests/dev.jsonl"
MODEL = "tiny"
SWEEP_LIMIT = 100          # clips per preset when sweeping (full run uses all)

BASE = dict(language="ja", task="transcribe", without_timestamps=True, fp16=False)

# decode presets to compare. "current" = the beam-5 defaults that gave 44.66%.
PRESETS = {
    "current":      {**BASE, "beam_size": 5},
    "greedy":       {**BASE, "beam_size": 1},
    "mild":         {**BASE, "beam_size": 5, "condition_on_previous_text": False},
    "no_fallback":  {**BASE, "beam_size": 5, "condition_on_previous_text": False,
                     "temperature": 0.0},
    "strict":       {**BASE, "beam_size": 5, "condition_on_previous_text": False,
                     "compression_ratio_threshold": 1.8, "logprob_threshold": -0.7,
                     "no_speech_threshold": 0.4},
}

# preset used by a normal (non-sweep) run
DECODE = PRESETS["current"]


def load_model(ckpt):
    model = whisper.load_model(MODEL)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def run(model, rows, decode, normalizer):
    refs, hyps = [], []
    for row in rows:
        result = model.transcribe(row["audio"], **decode)
        hyps.append(normalizer(result["text"]))
        refs.append(normalizer(row["text"]))
    return refs, hyps


def metrics(refs, hyps):
    mean = jiwer.cer(refs, hyps)
    per = [jiwer.cer(r, h) for r, h in zip(refs, hyps)]
    return mean, statistics.median(per), sum(c > 1 for c in per)


def main():
    args = sys.argv[1:]
    sweep = "--sweep" in args
    args = [a for a in args if a != "--sweep"]
    ckpt = args[0] if args else None

    model = load_model(ckpt)
    normalizer = BasicTextNormalizer()
    rows = [json.loads(l) for l in open(DEV_MANIFEST, encoding="utf-8")]
    tag = ckpt or "baseline (pretrained)"

    if sweep:
        subset = rows[:SWEEP_LIMIT]
        print(f"{MODEL} | {tag} | sweep on {len(subset)} clips\n")
        print(f"{'preset':<12} {'mean CER':>9} {'median':>8} {'CER>1':>6}")
        for name, decode in PRESETS.items():
            refs, hyps = run(model, subset, decode, normalizer)
            mean, med, cat = metrics(refs, hyps)
            print(f"{name:<12} {mean:>9.4f} {med:>8.3f} {cat:>6d}")
        return

    refs, hyps = run(model, rows, DECODE, normalizer)
    mean, med, cat = metrics(refs, hyps)
    print(f"{MODEL} | {tag} | dev clips {len(rows)} | "
          f"CER {mean:.4f} ({mean*100:.2f}%) | median {med:.3f} | CER>1: {cat}")

    per = sorted(((jiwer.cer(r, h), r, h) for r, h in zip(refs, hyps)), reverse=True)
    print("\n--- 10 worst clips ---")
    for c, r, h in per[:10]:
        print(f"CER {c:.2f}\n  ref: {r}\n  hyp: {h}\n")
    print("--- sampled (every 30th) ---")
    for c, r, h in per[::30]:
        print(f"CER {c:.2f}\n  ref: {r}\n  hyp: {h}\n")


if __name__ == "__main__":
    main()
