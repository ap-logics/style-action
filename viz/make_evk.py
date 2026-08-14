"""expression_vs_knowledge.pdf — consistency (activations) vs engram trace
(inside the network), with between-style baselines, random-weight control,
and CLIP reference line.
Data: results/engram_overlap_{mdm,t2mgpt,mld}.json (cluster exp. 1),
      random-W controls from results/engram_control_*.json on the cluster,
      consistency values from Table 1."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONS = {"MDM": 0.040, "T2M-GPT": 0.004, "MLD": 0.059}     # Table 1 (raw)
CTRL = {"MDM": 0.260, "T2M-GPT": 0.145, "MLD": 0.145}     # random-W control
CLIP_CONS = 0.644
ORDER = ["MLD", "MDM", "T2M-GPT"]                          # pipeline order

eng = {m: json.load(open(ROOT / f"results/engram_overlap_{k}.json"))
       for k, m in [("mdm", "MDM"), ("t2mgpt", "T2M-GPT"), ("mld", "MLD")]}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
x = np.arange(len(ORDER)); w = 0.36
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(x - w/2, [CONS[m] for m in ORDER], w, color="#b65f4a",
       label="activations: style consistency $\\mathcal{C}$")
ax.bar(x + w/2, [eng[m]["within_style_mean"] for m in ORDER], w, color="#6b7fae",
       label="engram trace $\\rho_{\\mathrm{within}}$")
first = True
for i, m in enumerate(ORDER):
    ax.text(i - w/2, CONS[m] + 0.012, f"{CONS[m]:.3f}", ha="center", fontsize=8.5)
    v = eng[m]["within_style_mean"]
    ax.text(i + w/2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    b = eng[m]["between_style_mean"]
    ax.plot([i + w/2 - 0.13, i + w/2 + 0.13], [b, b], color="k", lw=1.8, zorder=6,
            label="between-style baseline" if first else None)
    ax.plot([i + w/2 - 0.18, i + w/2 + 0.18], [CTRL[m], CTRL[m]], color="0.25",
            lw=1.6, ls=(0, (3, 2)), zorder=6,
            label="random-weight control" if first else None)
    first = False
ax.axhline(CLIP_CONS, color="#54588e", lw=1.2, ls="--")
ax.text(2.38, 0.605, f"CLIP input consistency ({CLIP_CONS:.2f})", ha="right",
        fontsize=8.5, color="#54588e")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_ylabel("style-axis coherence"); ax.set_ylim(-0.08, 0.72)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("The style axis: absent in activations, present inside the network", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT.parent / "overleaf/figures/expression_vs_knowledge.pdf")
print("wrote expression_vs_knowledge.pdf")
