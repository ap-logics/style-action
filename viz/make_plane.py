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

Reads v2 reports / seed summaries already computed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# (results dir, label, marker colour, is_generator)
SYSTEMS = [
    ("clip_v2",            "CLIP (input)",     "#2a2a2a", False),
    ("opentma_h3d_v2",     "TMA, HumanML3D",   "#4878a8", False),
    ("opentma_motionx_v2", "TMA, Motion-X",    "#4878a8", False),
    ("opentma_unimocap_v2","TMA, UniMoCap",    "#4878a8", False),
    ("mld_v2",             "MLD",              "#e08214", True),
    ("mdm_v2",             "MDM",              "#c43030", True),
    ("t2mgpt_v2",          "T2M-GPT",          "#8c1010", True),
]


def cka_cons(name):
    seed = ROOT / "results" / name / "seed_summary.json"
    if seed.exists():
        s = json.loads(seed.read_text())["summary"]
        return s["cka"]["primary_template_mean"], s["consistency"]["primary_template_mean"]
    r = json.loads((ROOT / "results" / name / "report.json").read_text())
    return r["cka_mean"], r["consistency_mean"]


def main():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    # quadrant shading
    ax.axhspan(0.4, 1.0, xmin=0, xmax=1, color="#f2f7f2", zorder=0)
    ax.axvspan(0.85, 1.0, ymin=0, ymax=1, color="#f2f2f7", alpha=0.4, zorder=0)

    for name, label, col, gen in SYSTEMS:
        x, y = cka_cons(name)
        ax.scatter([x], [y], s=150 if gen else 90,
                   marker="s" if gen else "o", c=col,
                   edgecolors="white", linewidths=1.3, zorder=5)
        dy = 0.022 if label != "MLD" else -0.045
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(7, 6 if dy > 0 else -12), fontsize=8.5,
                    color=col, fontweight="bold" if gen else "normal", zorder=6)

    ax.set_xlabel("relational invariance  (mean CKA)", fontsize=10)
    ax.set_ylabel("directional equivariance  (consistency)", fontsize=10)
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(-0.03, 0.75)
    ax.axhline(0.0, color="#bbb", lw=0.7, ls=":")
    ax.text(0.995, 0.70, "factored", ha="right", fontsize=8, color="#3a7a3a",
            style="italic")
    ax.text(0.995, 0.02, "action structure kept,\nno style axis", ha="right",
            fontsize=7.5, color="#7a5a1a", style="italic")
    ax.text(0.57, 0.02, "scrambled,\nno style axis", ha="left",
            fontsize=7.5, color="#7a2a2a", style="italic")
    ax.set_title("The two axes are independent; systems fall in three corners",
                 fontsize=10)
    ax.grid(True, alpha=0.15)

    out = ROOT.parent / "overleaf" / "figures" / "plane.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
