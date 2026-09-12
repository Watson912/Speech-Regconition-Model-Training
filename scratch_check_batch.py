"""
Throwaway Stage 3 verification.

Roadmap "done when": one batch prints the right shapes, and the tokens
decode back to the original transcript. This script does exactly that and
nothing else -- delete it once Stage 3 is ticked.

Run:  python scratch_check_batch.py
"""

import torch
from torch.utils.data import DataLoader

from dataset import SpeechDataset

MANIFEST = "Data/manifests/train.jsonl"
BATCH_SIZE = 4


def main():
    ds = SpeechDataset(MANIFEST)
    print(f"dataset size: {len(ds)}")

    # collate_fn is a bound method so it can see ds.tokenizer.
    # num_workers=0 keeps this simple and debuggable.
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ds.collate_fn,
        num_workers=0,
    )

    mels, input_ids, labels = next(iter(loader))

    # ---- 1. shapes ----
    print("\n--- shapes ---")
    print(f"mels      {tuple(mels.shape)}   expect ({BATCH_SIZE}, 80, 3000)")
    print(f"input_ids {tuple(input_ids.shape)}")
    print(f"labels    {tuple(labels.shape)}   (same L as input_ids)")
    print(f"mels dtype {mels.dtype}, input_ids dtype {input_ids.dtype}")

    assert mels.shape == (BATCH_SIZE, 80, 3000)
    assert input_ids.shape == labels.shape

    # ---- 2. padding / masking sanity ----
    eot = ds.tokenizer.eot
    print("\n--- padding / masking ---")
    for i in range(BATCH_SIZE):
        real_len = int((labels[i] != -100).sum())      # non-masked positions
        n_masked = int((labels[i] == -100).sum())
        print(
            f"row {i}: real tokens={real_len:3d}  masked pad={n_masked:3d}  "
            f"input_ids tail == eot: {bool((input_ids[i, real_len:] == eot).all())}"
        )

    # ---- 3. round-trip decode ----
    print("\n--- decode check (should match the manifest text) ---")
    for i in range(BATCH_SIZE):
        ids = input_ids[i].tolist()
        # drop special tokens (sot sequence, eot, and eot-padding) before decode
        text_ids = [t for t in ids if t < ds.tokenizer.eot]
        decoded = ds.tokenizer.decode(text_ids)
        print(f"row {i}: {decoded}")


if __name__ == "__main__":
    main()
