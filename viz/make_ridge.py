"""posterior_ridge.pdf — 3D ridgeline of the four system-level escape
posteriors (talk figure, not in the paper).
NOTE: densities are logit-normal reconstructions from posterior mean + 94%
interval; the raw MCMC trace was not saved. For true KDE ridges, rerun
analysis/bayes_escape.py with trace saving enabled."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
PAR = LinearSegmentedColormap.from_list(
    "p", ["#54588e", "#6b7fae", "#06a4ca", "#5b8a72", "#87bf77", "#fec832"])
SYS = [("CLIP", 0.00434, 0.00077, 0.011),
       ("MLD", 0.0172, 0.008, 0.032),
       ("MDM", 0.309, 0.21, 0.43),
       ("T2M-GPT", 0.704, 0.63, 0.77)]

logit = lambda p: np.log(p / (1 - p))
x = np.linspace(1e-4, 0.85, 1200)
dens = []
for name, m, lo, hi in SYS:
    mu, sig = logit(m), (logit(hi) - logit(lo)) / (2 * 1.88)
    d = np.exp(-0.5 * ((logit(x) - mu) / sig) ** 2) / (x * (1 - x))
    dens.append(d / d.max())

fig = plt.figure(figsize=(9.2, 6.0))
ax = fig.add_subplot(111, projection="3d")
verts = [[(x[0], 0.), *zip(x, d), (x[-1], 0.)] for d in dens]
colors = [PAR(i / 3.5) for i in range(4)]
ax.add_collection3d(PolyCollection(verts, facecolors=colors, edgecolors="white",
                    linewidths=1.4, alpha=0.92), zs=[3, 2, 1, 0], zdir="y")
for i, d in enumerate(dens):
    ax.plot(x, np.full_like(x, 3 - i), d * 0.02, color=colors[i], lw=3, alpha=0.35)
for i, (name, m, lo, hi) in enumerate(SYS):
    ax.text(m, 3 - i, 1.12, name, fontsize=11, fontweight="bold",
            color=colors[i], ha="center")
ax.set_xlim(0, 0.85); ax.set_ylim(-0.4, 3.4); ax.set_zlim(0, 1.3)
ax.set_xlabel("posterior escape probability  $p_m$", fontsize=10, labelpad=12)
ax.set_ylabel("system  (pipeline order $\\rightarrow$)", fontsize=10, labelpad=10)
ax.set_zlabel("posterior density\n(scaled)", fontsize=9, labelpad=6)
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(["", "", "", ""])
ax.set_zticks([])
ax.view_init(elev=32, azim=-55)
ax.set_box_aspect((2.1, 1.9, 0.85))
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_alpha(0.04)
ax.grid(False)
ax.set_title("Hierarchical posteriors: the escape-probability ladder "
             "(94% HDIs disjoint)", fontsize=11)
plt.savefig(ROOT.parent / "overleaf/figures/posterior_ridge.pdf",
            bbox_inches="tight", dpi=220)
print("wrote posterior_ridge.pdf")
