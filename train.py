import csv, itertools, time
from pathlib import Path
import torch, torch.nn.functional as F, whisper
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SpeechDataset

MODEL = "tiny"
EPOCHS = 2
BATCH_SIZE = 4
LR = 1e-5
EVAL_EVERY = 200
CKPT = "ckpt/run3_best.pt"
RESUME_CKPT = "ckpt/run3_resume.pt"   # full training state, for continuing after a disconnect
RUN = "runs/run3"          # loss log -> runs/run3.csv, plot -> runs/run3.png
TB_DIR = "runs/run3_tb"    # tensorboard event files -> `tensorboard --logdir runs`

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

    total_steps = EPOCHS * len(train_loader)
    step = 0
    best = float("inf")
    history = []          # (step, train_loss, dev_loss)

    if Path(RESUME_CKPT).exists():
        state = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        best = state["best"]
        history = state["history"]
        print(f"resumed from {RESUME_CKPT} at step {step}/{total_steps}")

    writer = SummaryWriter(TB_DIR)

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

    model.train()
    batches = itertools.islice(itertools.chain.from_iterable(itertools.repeat(train_loader, EPOCHS)), step, total_steps)
    pbar = tqdm(batches, initial=step, total=total_steps, desc="train")
    for batch in pbar:
        loss = compute_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        writer.add_scalar("loss/train", loss.item(), step)

        if step % EVAL_EVERY == 0:
            dev = evaluate()
            history.append((step, loss.item(), dev))
            writer.add_scalar("loss/dev", dev, step)
            flag = ""
            if dev < best:
                best = dev
                torch.save(model.state_dict(), CKPT)
                flag = " <-- saved"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "best": best,
                "history": history,
            }, RESUME_CKPT)
            pbar.write(f" [eval] step {step} | dev loss {dev:.4f}{flag}")

    writer.close()

    # write the loss log
    with open(f"{RUN}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("step", "train_loss", "dev_loss"))
        w.writerows(history)
    print(f"done. best dev loss {best:.4f}  |  loss log: {RUN}.csv  |  tensorboard: {TB_DIR}")

if __name__ == "__main__":
    main()