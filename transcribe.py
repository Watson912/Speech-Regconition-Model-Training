import sys, torch, whisper

def main():
    ckpt, audio_path = sys.argv[1], sys.argv[2]

    model = whisper.load_model("tiny")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    result = model.transcribe(audio_path, language="ja", task="transcribe", fp16=False)
    print(result["text"])


if __name__ == "__main__":
    main()
