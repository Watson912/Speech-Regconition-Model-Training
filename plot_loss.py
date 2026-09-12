"""
Plot the train/dev loss curve from a run's CSV.

Usage:  python plot_loss.py runs/run1
Needs:  pip install matplotlib
"""

import csv
import sys

import matplotlib.pyplot as plt

run = sys.argv[1] if len(sys.argv) > 1 else "runs/run1"

with open(f"{run}.csv") as f:
    rows = list(csv.reader(f))[1:]          # skip header

step = [int(r[0]) for r in rows]
train = [float(r[1]) for r in rows]
dev = [float(r[2]) for r in rows]

plt.figure(figsize=(7, 4))
plt.plot(step, train, marker="o", label="train loss")
plt.plot(step, dev, marker="o", label="dev loss")
plt.xlabel("training step")
plt.ylabel("cross-entropy loss (cost)")
plt.title(run)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out = f"{run}.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")

# what to read off the curve:
#   both falling then flat        -> good, roughly done
#   dev flat/rising, train falls  -> overfitting; the saved ckpt is from the low point
#   both still falling at the end -> train longer (more EPOCHS)
