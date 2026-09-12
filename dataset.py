import json
import torch
import whisper
from pathlib import Path
import torch.nn.functional as F

class SpeechDataset(torch.utils.data.Dataset):
    """
    Class, take a seat. This object is the bridge between the files on your
    disk and the tensors that Whisper's training loop wants to eat.

    PyTorch asks only three things of a map-style dataset:
      - tell me how to set yourself up      -> __init__
      - tell me how many examples you hold  -> __len__
      - give me example number `i`          -> __getitem__

    Get those three right and the DataLoader (batching, shuffling, worker
    processes) comes for free. Everything below is just those three methods,
    done carefully.
    """

    def __init__(self, manifest_path, min_duration=0.0):
        # --- Runs ONCE, when you build the dataset. Do the cheap, one-time
        # setup here; never the heavy per-example work. ---

        self.manifest_path = manifest_path

        # We read a JSONL "manifest": one JSON object per line, each looking
        # like {"audio": "path/to/clip.wav", "text": "こんにちは"}.
        # We only load these little bookkeeping rows into memory here -- NOT
        # the audio itself. Audio is big; paths are tiny. Keep it that way.
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.data.append(row)

        if min_duration > 0.0:
            before = len(self.data)
            self.data = [r for r in self.data if r["duration"] >= min_duration]
            print(f"{manifest_path}: dropped {before - len(self.data)} clips " f"under {min_duration}s, {len(self.data)} left")

        # ***  IMPORTANT LINE  ***
        # The tokenizer turns Japanese text into the integer IDs Whisper was
        # trained on, and back again. The three arguments matter:
        #   multilingual=True  -> use the multilingual vocab (the .en models
        #                         have a different, English-only one)
        #   language="ja"      -> picks the <|ja|> language token
        #   task="transcribe"  -> <|transcribe|>, not <|translate|>
        # Build it ONCE here. Constructing it per example would be wasteful,
        # and mismatching these flags with your model is a classic silent
        # bug: training "works" but the loss never really drops.
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language="ja", task="transcribe"
        )


    def __len__(self):
        # Simple, but not decoration: the DataLoader's sampler uses this
        # number to know which indices are valid and when an epoch ends.
        return len(self.data)

    def __getitem__(self, idx):
        """
        ***  THIS IS THE HEART OF THE CLASS.  ***

        Runs every time the loader needs example number `idx` -- thousands of
        times per epoch, possibly in parallel worker processes. It takes ONE
        manifest row and returns ONE (input, target) pair.

        Two halves: turn the audio into the model's input, and turn the text
        into the model's target.
        """
        row = self.data[idx]

        # ---- Audio side: waveform -> log-mel spectrogram ----

        # load_audio: decode the file and resample to 16 kHz mono. Whisper is
        # hard-wired for 16 kHz; feed it anything else and every later step is
        # subtly wrong.
        audio = whisper.load_audio(row["audio"])

        # pad_or_trim: force the clip to exactly 30 seconds (480000 samples).
        # Short clips get zero-padded, long clips get cut. Whisper's encoder
        # only accepts this one fixed length -- there is no "variable length"
        # mode. This is also why we don't need to pad the mels in a collate fn.
        audio = whisper.pad_or_trim(audio)

        # ***  IMPORTANT LINE  ***
        # log_mel_spectrogram: the actual input to the model. NOT a cache, not
        # an optimization -- Whisper's encoder consumes log-mel, never raw
        # waveform. Output shape is (80, 3000): 80 mel frequency bins by 3000
        # time frames (10 ms per frame). If you ever pre-compute and cache
        # these to disk, that's fine for speed, but know that you've frozen
        # the audio -- no more waveform-level augmentation after this point.
        mel = whisper.log_mel_spectrogram(audio)

        # ---- Text side: string -> token IDs ----

        # Encode just the transcript text. These are the "content" tokens.
        text_tokens = self.tokenizer.encode(row["text"])

        # ***  IMPORTANT: the token layout must match how Whisper decodes.  ***
        # The decoder is trained on a specific prefix, not bare text:
        #
        #   <|startoftranscript|> <|ja|> <|transcribe|> <|notimestamps|>
        #        ... transcript tokens ...  <|endoftext|>
        #
        # sot_sequence_including_notimestamps gives you that whole 4-token
        # prefix. We append the text, then the EOT token that teaches the
        # model where to STOP -- without it, generation never terminates.
        # In the training loop you'll typically feed tokens[:-1] as decoder
        # input and predict tokens[1:] as the target (teacher forcing).
        tokens = (
            list(self.tokenizer.sot_sequence_including_notimestamps)
            + text_tokens
            + [self.tokenizer.eot]
        )

        # mel: a plain tensor already. tokens: a Python list, so we wrap it.
        # Note the two returned items have different lengths from example to
        # example on the token side -- your DataLoader will need a collate_fn
        # that pads the token sequences into a rectangular batch.
        return mel, torch.tensor(tokens)

    def collate_fn(self, batch):
        """
        The DataLoader builds a batch by calling __getitem__ several times,
        which hands us a LIST of single examples:
            [(mel_0, tokens_0), (mel_1, tokens_1), ...]
        Our job: pack that list into ONE stacked batch the model can run in
        a single forward pass. This is the "collate" (gather together) step.

        The mel side is trivial -- every mel is already (80, 3000), so they
        stack straight into (B, 80, 3000). All the real work is on the token
        side, and it comes down to two classic sequence-model ideas:
        PADDING and MASKING.
        """

        eot = self.tokenizer.eot
        mels, token_sequences = zip(*batch)

        # --- mel side: same shape already, just stack ---
        mels = torch.stack(mels)                       # (B, 80, 3000)

        # --- token side ---
        #
        # PADDING -- why:
        # A batch has to be one rectangular tensor. But the sentences differ
        # in length (one clip is 12 tokens, another is 30). You cannot stack
        # a length-12 row and a length-30 row into a clean grid. So we find
        # the longest sequence in THIS batch and stretch every other sequence
        # up to that length by appending filler tokens on the right. Now all
        # rows match and torch.stack works.
        max_len = max(t.size(0) for t in token_sequences)

        input_ids = []
        labels = []
        for t in token_sequences:
            pad_len = max_len - t.size(0)

            # input_ids: what we FEED the decoder. Pad with `eot`. The value
            # barely matters here -- it's just placeholder so the row is full
            # length -- but Whisper has no dedicated <pad> token, and `eot`
            # is the natural "nothing more to say" choice.
            input_ids.append(F.pad(t, (0, pad_len), value=eot))

            # labels: the ANSWER KEY the loss is graded against.
            #
            # MASKING -- why:
            # The padded positions are fake. We must not reward or punish the
            # model for whatever it predicts there, or it would waste capacity
            # learning to output filler. We mark those positions with -100,
            # which is torch's cross_entropy(ignore_index=-100) default: any
            # label == -100 contributes ZERO to the loss. That is the mask.
            labels.append(F.pad(t, (0, pad_len), value=-100))

        input_ids = torch.stack(input_ids)            # (B, max_len)
        labels = torch.stack(labels)                  # (B, max_len)

        # Note: input_ids and labels are the SAME sequence here, differing
        # only in pad value. The teacher-forcing shift (feed tokens[:-1],
        # predict tokens[1:]) happens in the training loop, not here -- don't
        # shift it twice.
        return mels, input_ids, labels


