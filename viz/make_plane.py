"""
The invariance-equivariance plane: mean CKA (x) vs style consistency (y),
one point per system, on the v2 grid. The two axes of the framework are the
two axes of the plot, so a system's position reads directly as which
property it satisfies.

- Top-right (high CKA, high consistency): factored representation (CLIP).
- Top-left is empty: you cannot have a style axis without relational structure.
- Bottom-right (high CKA, low consistency): action structure intact but no
  style direction (MLD) -- the dissociation a single-metric probe misses.
- Bottom-left (low both): structure scrambled and no axis (MDM, T2M-GPT).

Generators get sd error bars (over seeds x templates) and an open-circle
marker at the noise-corrected consistency (Appendix "Sampling-noise
attenuation"; values hardcoded from that analysis).

Reads v2 reports / seed summaries already computed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#352a87", "MLD": "#0f5cdd", "MDM": "#2eb7a4",
       "T2M-GPT": "#9aab3a", "TMA": "#8888aa"}
CORRECTED = {"T2M-GPT": 0.009, "MDM": 0.102, "MLD": 0.162}   # C / reliability

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def seed_stats(model):
    ss = json.load(open(ROOT / f"results/{model}/seed_summary.json"))["per_run"]
    ck = [r["cka"] for sd in ss.values() for r in sd.values()]
    co = [r["consistency"] for sd in ss.values() for r in sd.values()]
    return np.mean(ck), np.std(ck), np.mean(co), np.std(co)


pts = {n: seed_stats(m) for m, n in [("t2mgpt_v2", "T2M-GPT"),
                                     ("mdm_v2", "MDM"), ("mld_v2", "MLD")]}
r = json.load(open(ROOT / "results/clip_v2/report.json"))
pts["CLIP"] = (r["cka_mean"], 0, r["consistency_mean"], 0)

tma_pts = {}
for label, dirn in [("H3D", "opentma_h3d_v2"), ("UniMoCap", "opentma_unimocap_v2"),
                    ("Motion-X", "opentma_motionx_v2")]:
    rr = json.load(open(ROOT / f"results/{dirn}/report.json"))
    tma_pts[label] = (rr["cka_mean"], rr["consistency_mean"])

fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.axhspan(0.33, 0.75, xmin=0.66, color="#eaf3ea", zorder=0)
ax.axhspan(-0.03, 0.13, xmin=0.66, color="#fdf3e3", zorder=0)
ax.axhspan(-0.03, 0.13, xmax=0.32, color="#fbebea", zorder=0)
ax.text(0.858, 0.715, "factored: both axes", fontsize=9, color="#3a7d3a",
        style="italic", va="top")
ax.text(0.858, 0.118, "coupled: structure kept,\nstyle axis lost", fontsize=9,
        color="#b07d2a", style="italic", va="top")
ax.text(0.556, 0.118, "scrambled: both axes lost", fontsize=9, color="#a44",
        style="italic", va="top")

label_off = {"CLIP": (-0.012, 0.0, "right"), "MLD": (0.013, -0.008, "left"),
             "MDM": (0.0, -0.038, "center"), "T2M-GPT": (0.013, -0.020, "left")}
for name, (cx, sx, cy, sy) in pts.items():
    if name in CORRECTED:
        ax.plot([cx, cx], [cy, CORRECTED[name]], color=COL[name], lw=0.9,
                alpha=0.6, zorder=4)
        ax.plot(cx, CORRECTED[name], "o", ms=8, mfc="white", mec=COL[name],
                mew=1.6, zorder=5)
    ax.errorbar(cx, cy, xerr=sx, yerr=sy, fmt="o", ms=8, color=COL[name],
                elinewidth=1.4, zorder=5)
    dx, dy, ha = label_off[name]
    ax.annotate(name, (cx, cy), xytext=(cx + dx, cy + dy), ha=ha, va="center",
                fontweight="bold", color=COL[name], fontsize=10)

tma_off = {"H3D": (0, -0.035), "UniMoCap": (0, 0.020), "Motion-X": (0, 0.020)}
for label, (cx, cy) in tma_pts.items():
    ax.plot(cx, cy, "s", ms=6, color=COL["TMA"], zorder=4)
    dx, dy = tma_off[label]
    ax.annotate(f"TMA {label}", (cx, cy), xytext=(cx + dx, cy + dy),
                ha="center", fontsize=8, color=COL["TMA"])

handles = [Line2D([], [], marker="o", ls="", color="0.3", label="measured (raw)"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="0.3",
                  label="noise-corrected")]
ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="center left")
ax.set_xlim(0.55, 1.0); ax.set_ylim(-0.03, 0.75)
ax.set_xlabel("relational invariance  (mean CKA)")
ax.set_ylabel("directional equivariance  (consistency $\\mathcal{C}$)")
ax.set_title("Invariance and equivariance are independent axes", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT.parent / "overleaf/figures/plane.pdf")
print("wrote plane.pdf")
