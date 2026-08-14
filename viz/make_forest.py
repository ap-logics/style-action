"""bayes_forest.pdf — per-style escape posteriors, 4 systems x 13 styles.
Data: results/bayes_escape_summary.json (means + 94% HDIs from the
hierarchical model; rerun analysis/bayes_escape.py to refresh)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#0072B2", "MLD": "#56B4E9", "MDM": "#009E73", "T2M-GPT": "#E69F00"}
SYS = [("clip_v2", "CLIP"), ("mld_v2", "MLD"), ("mdm_v2", "MDM"), ("t2mgpt_v2", "T2M-GPT")]
OFF = {"CLIP": -0.27, "MLD": -0.09, "MDM": 0.09, "T2M-GPT": 0.27}

d = json.load(open(ROOT / "results/bayes_escape_summary.json"))
styles = sorted(d["clip_v2"], key=lambda s: d["t2mgpt_v2"][s]["mean"])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for key, name in SYS:
    xs = [d[key][s]["mean"] for s in styles]
    ys = [i + OFF[name] for i in range(len(styles))]
    lo = [d[key][s]["mean"] - d[key][s]["hdi94"][0] for s in styles]
    hi = [d[key][s]["hdi94"][1] - d[key][s]["mean"] for s in styles]
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4, color=COL[name],
                elinewidth=1.6, capsize=0, label=name)
for i in range(0, len(styles), 2):
    ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
ax.set_yticks(range(len(styles))); ax.set_yticklabels(styles)
ax.set_xlim(-0.01, 0.9); ax.set_ylim(-0.6, len(styles) - 0.4)
ax.set_xlabel("posterior escape probability (mean, 94% HDI)")
ax.legend(loc="lower right", frameon=False)
ax.set_title("Per-style basin escape: four systems, non-overlapping bands", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT.parent / "overleaf/figures/bayes_forest.pdf")
print("wrote bayes_forest.pdf")
