import csv, time
from pathlib import Path
import torch, torch.nn.functional as F, whisper
from torch.utils.data import DataLoader, ConcatDataset, Subset
from dataset import SpeechDataset

MODEL = "small"
EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-5
EVAL_EVERY = 200
CKPT = "ckpt/run3_best.pt"
RUN = "runs/run3"          # loss log -> runs/run3.csv, plot -> runs/run3.png

def main():

    device = "cuda" if torch.backends.mps.is_available() else "cpu"

    # make sure output dirs exist
    Path(CKPT).parent.mkdir(parents=True, exist_ok=True)
    Path(RUN).parent.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(MODEL).to(device)
    train_ds_main = SpeechDataset("Data/manifests/train.jsonl", min_duration=1.0)
    train_ds_extra = SpeechDataset("Data/manifests/train_extra.jsonl", min_duration=1.0)
    train_ds = ConcatDataset([train_ds_main, train_ds_extra])
    train_ds = Subset(train_ds, range(0, len(train_ds), 2))  # half the data, for now
    dev_ds = SpeechDataset("Data/manifests/dev.jsonl")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=train_ds_main.collate_fn)
    dev_loader = DataLoader(dev_ds, BATCH_SIZE, shuffle=False, collate_fn=dev_ds.collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def compute_loss(batch):
        mels, input_ids, labels = [x.to(device) for x in batch]
        logits = model(tokens=input_ids[:, :-1], mel=mels)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = [compute_loss(batch).item() for batch in dev_loader]
        model.train()
        return sum(losses) / len(losses)

    # history: one row per eval, written to CSV at the end for plotting
    history = []          # (step, train_loss, dev_loss)

    best = float("inf")
    step = 0
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            loss = compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} | train loss {loss.item():.4f}")

            if step % EVAL_EVERY == 0:
                dev = evaluate()
                history.append((step, loss.item(), dev))
                flag = ""
                if dev < best:
                    best = dev
                    torch.save(model.state_dict(), CKPT)
                    flag = " <-- saved"
                print(f" [eval] step {step} | dev loss {dev:.4f}{flag}")

    # write the loss log
    with open(f"{RUN}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("step", "train_loss", "dev_loss"))
        w.writerows(history)
    print(f"done. best dev loss {best:.4f}  |  loss log: {RUN}.csv")

if __name__ == "__main__":
    main()