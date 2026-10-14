"""Trace la loss du pré-entraînement à chaque step loggé, à partir de logs/train.log."""
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "train.log"
OUT = Path(__file__).resolve().parent / "loss_par_step.png"

text = LOG.read_text(encoding="utf-8", errors="ignore")

# Le log contient le run en double : on garde la dernière valeur vue pour chaque step.
steps_loss = {}
for step, total, loss in re.findall(r"step (\d+)/(\d+)\s+loss ([\d.]+)", text):
    steps_loss[int(step)] = float(loss)
total_steps = int(total)
steps = sorted(steps_loss)
losses = [steps_loss[s] for s in steps]

epochs = {int(e): float(l) for e, l in re.findall(r"epoch\s+(\d+)/\d+\s+loss ([\d.]+)", text)}
steps_per_epoch = total_steps // max(epochs)
warmup = int(re.search(r"(\d+) of warmup", text).group(1))

INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"

fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.axvspan(0, warmup, color=GRID, alpha=0.6, lw=0)
ax.text(warmup / 2, 0.98, "warmup", transform=ax.get_xaxis_transform(),
        ha="center", va="top", color=MUTED, fontsize=9)

ax.plot(steps, losses, color=BLUE, lw=2, marker="o", ms=3)

i_min = losses.index(min(losses))
ax.annotate(f"min {losses[i_min]:.3f} (step {steps[i_min]})",
            (steps[i_min], losses[i_min]), xytext=(25, -20), textcoords="offset points",
            color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate(f"fin {losses[-1]:.3f}", (steps[-1], losses[-1]), xytext=(-10, 15),
            textcoords="offset points", ha="right", color=INK, fontsize=9)

ax.set_xlim(0, total_steps)
ax.set_ylim(0, None)
ax.set_xlabel("Step", color=MUTED)
ax.set_ylabel("Loss (prédicteur vs encodeur cible)", color=MUTED)
ax.set_title(f"I-JEPA — loss par step ({max(epochs)} epochs, {total_steps} steps, loggée tous les 50 steps)",
             loc="left", color=INK, fontsize=12)

# Axe secondaire en epochs (même échelle, simple relabel de l'axe x).
sec = ax.secondary_xaxis("top", functions=(lambda s: s / steps_per_epoch, lambda e: e * steps_per_epoch))
sec.set_xlabel("Epoch", color=MUTED)
sec.tick_params(colors=MUTED)

ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)

fig.tight_layout()
fig.savefig(OUT, facecolor=fig.get_facecolor())
print(f"{len(steps)} points -> {OUT}")
