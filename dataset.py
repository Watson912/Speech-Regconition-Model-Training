import json
import torch
import whisper
from pathlib import Path

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.data.append(row)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="ja", task="transcribe")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        audio = whisper.load_audio(row["audio"])
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        text_tokens = self.tokenizer.encode(row["text"])
        tokens = (
            list(self.tokenizer.sot_sequence_including_notimestamps)

            + text_tokens
            + [self.tokenizer.eot]
        )

        return mel, torch.tensor(tokens)