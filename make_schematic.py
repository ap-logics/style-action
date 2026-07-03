"""
Conceptual figure: the separable-axes hypothesis vs what we measure.

Left panel: the idealised factored latent — actions arranged along a
low-dimensional action manifold, and one shared style direction delta_s
such that z(a, s) = z(a) + delta_s for every action a. All style arrows
are parallel; consistency = 1.

Right panel: what the diagnostics find in motion latents — style shifts
are large but their directions are action-specific (consistency ~ 0),
and some styled points cross into another action's basin (red).

Pure schematic — positions are illustrative, not data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

ACTIONS = ["walk", "run", "jump", "dance"]
# action manifold: gentle arc
tx = np.linspace(0.12, 0.88, len(ACTIONS))
ty = 0.28 + 0.10 * np.sin(np.linspace(0.4, 2.6, len(ACTIONS)))
DELTA = np.array([0.10, 0.30])          # the ideal shared style direction
DELTA = DELTA / np.linalg.norm(DELTA) * 0.30


def base(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # action manifold curve
    xs = np.linspace(0.06, 0.94, 100)
    ys = 0.28 + 0.10 * np.sin(np.linspace(0.2, 2.8, 100))
    ax.plot(xs, ys, c="#bbbbbb", lw=1.4, zorder=1)
    ax.annotate("action manifold", (0.72, 0.16), fontsize=8, color="#888")
    ax.scatter(tx, ty, s=70, c="#222", zorder=5)
    for x, y, n in zip(tx, ty, ACTIONS):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(2, -14),
                    fontsize=9, ha="center")
    ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

# ---- left: hypothesis ----
ax = axes[0]
base(ax, "Hypothesis: style is an axis\n$z(a,s) = z(a) + \\delta_s$")
for x, y in zip(tx, ty):
    ax.annotate("", xy=(x + DELTA[0], y + DELTA[1]), xytext=(x, y), zorder=4,
                arrowprops=dict(arrowstyle="->", color="#2a7ab8", lw=1.8))
    ax.scatter([x + DELTA[0]], [y + DELTA[1]], s=36, facecolors="white",
               edgecolors="#2a7ab8", linewidths=1.4, zorder=5)
# shared delta label
ax.annotate(r"$\delta_{\mathrm{happily}}$ (shared)",
            (tx[1] + DELTA[0] + 0.02, ty[1] + DELTA[1] + 0.02),
            fontsize=9, color="#2a7ab8")
ax.annotate("consistency $\\approx 1$", (0.05, 0.9), fontsize=9,
            color="#2a7ab8", fontweight="bold")

# ---- right: measurement ----
ax = axes[1]
base(ax, "Measured in motion latents:\ndirections decohere")
angles = [-0.5, 0.4, -0.9, 1.9]           # scattered directions
for i, (x, y, th) in enumerate(zip(tx, ty, angles)):
    d = np.array([np.cos(th), np.sin(th)]) * 0.26
    escaped = (i == 2)                    # jump's arrow lands near dance
    if escaped:
        d = np.array([tx[3] - x + 0.03, ty[3] - y + 0.05])
    col = "#c43030" if escaped else "#2a7ab8"
    ax.annotate("", xy=(x + d[0], y + d[1]), xytext=(x, y), zorder=4,
                arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax.scatter([x + d[0]], [y + d[1]], s=36, facecolors="white",
               edgecolors=col, linewidths=1.4, zorder=5)
ax.annotate("basin escape:\n\"jumping happily\" is\nnearer to dancing",
            (0.56, 0.66), fontsize=8, color="#c43030")
ax.annotate("consistency $\\approx 0$", (0.05, 0.9), fontsize=9,
            color="#c43030", fontweight="bold")

plt.tight_layout()
out = Path(__file__).parent.parent / "overleaf" / "figures" / "schematic.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
