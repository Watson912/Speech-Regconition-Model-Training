# Speech Recognition — Whisper Fine-Tuning (Personal Learning Project)

## Goal
Personal project to learn the process of fine-tuning a speech recognition model
from scratch, hands-on. The point is to build understanding by writing the code
myself, not to get a finished model quickly.

Working direction: fine-tune Whisper on Japanese speech, eventually aiming at
anime/expressive dialogue as a niche (exaggerated emotional delivery — screaming,
crying, whispering — that standard ASR benchmarks don't cover well). Not
committed to this niche yet; may evolve.

## How Claude should work in this project

**Do not write or create code, files, or scaffolding.** The user is doing all
implementation themselves as the learning exercise. Claude's role is limited to:
- Answering questions
- Debugging (reading errors, explaining what's wrong, suggesting fixes verbally)
- Explaining concepts, tradeoffs, and what to look into next

Only write/edit files if the user explicitly asks for it. Default to "explain,
don't implement."

**Keep answers short.** Answer what was asked, no more. Lead with the direct
answer, then only the detail needed to act on it. No exhaustive background,
no listing options that won't be used, no restating what I already know. Expand
only when I ask for depth.

## Key decisions made so far
- Using the **original OpenAI `openai/whisper` repo** (pip package
  `openai-whisper`), not Hugging Face `transformers`, to see the raw model and
  preprocessing without HF's abstractions.
  - Tradeoff: no built-in training loop — training loop must be written by hand.
  - Hugging Face `transformers` remains a fallback/comparison option later once
    the manual approach is understood.
- Plan is incremental:
  1. Run pretrained model for inference only, no training — validate
     environment and pipeline.
  2. Load a small dataset slice, write data pipeline by hand (resample to
     16kHz, chunk ≤30s, tokenize with Whisper's tokenizer/processor).
  3. Fine-tune a small model (tiny/base) on a tiny subset — get a training
     loop running end-to-end, not worried about quality yet.
  4. Scale up data/model size, learn to measure WER properly instead of
     eyeballing output.
  5. Chase the expressive-speech/anime niche once the basic pipeline works.

## Data sourcing notes
- Anime audio scraped from streaming/BDs is copyrighted; fansub subtitles are
  translations, not accurate transcripts of the spoken audio — bad training
  pairs. Avoid as a primary data source.
- **ReazonSpeech** — large Japanese corpus from real TV broadcast audio with
  actual transcripts (not translated subtitles). Best current candidate for
  the expressive/anime-adjacent niche since broadcast TV includes anime and
  varied emotional delivery.
- JSUT / JVS / Common Voice (Japanese) — cleaner, smaller, neutral-register
  speech. Good for an earlier, simpler baseline before ReazonSpeech.
