"""coupling_landscape.pdf — two-panel talk/appendix figure in the muted theme.
Left: style-flow decoherence over T2M-GPT's latent PCA plane as a 3D surface
(height = 1 - directional coherence, real RBF field from measured vectors).
Right: stacked posterior escape densities for the four systems (true NUTS
draws), offset like value-law panels, ordered by pipeline depth."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#0072B2", "MLD": "#56B4E9", "MDM": "#009E73", "T2M-GPT": "#E69F00"}
MUTED = LinearSegmentedColormap.from_list("muted",
        ["#303864", "#6c7a9e", "#cec4aa", "#E69F00"])
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# ---------- left: turbulence surface from real t2mgpt field ----------
sub = ROOT / "results/t2mgpt/0"
Z_S = np.load(sub / "Z_S.npy"); Z_T = np.load(sub / "Z_T.npy")
if Z_T.ndim == 2: Z_T = Z_T.reshape(-1, Z_S.shape[0], Z_S.shape[1])
meta = json.load(open(sub / "meta.json"))
X = np.vstack([Z_S] + [Z_T[j] for j in range(Z_T.shape[0])])
mu = X.mean(0)
_, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
proj = lambda Z: (Z - mu) @ Vt[:2].T
P_S = proj(Z_S)
roots = np.vstack([P_S] * Z_T.shape[0])
vecs = np.vstack([proj(Z_T[j]) - P_S for j in range(Z_T.shape[0])])
pad = 0.25 * (P_S.max(0) - P_S.min(0))
gx = np.linspace(P_S[:,0].min()-pad[0], P_S[:,0].max()+pad[0], 90)
gy = np.linspace(P_S[:,1].min()-pad[1], P_S[:,1].max()+pad[1], 90)
XX, YY = np.meshgrid(gx, gy)
d = np.linalg.norm(roots[:,None]-roots[None], axis=-1)
sig = np.median(d[d>0]) * 0.45
P = np.stack([XX.ravel(), YY.ravel()], 1)
w = np.exp(-((P[:,None]-roots[None])**2).sum(-1) / (2*sig**2))
wsum = w.sum(1, keepdims=True) + 1e-12
U = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
coh = np.linalg.norm((w @ U) / wsum, axis=1).reshape(XX.shape)
turb = 1 - coh

fig = plt.figure(figsize=(11.6, 4.6))
ax = fig.add_subplot(121, projection="3d")
ax.computed_zorder = False
ax.plot_surface(XX, YY, turb, cmap=MUTED, rstride=2, cstride=2,
                linewidth=0.15, edgecolor="white", alpha=0.92, antialiased=True, zorder=1)
# neutral action embeddings dropped onto the surface
for i, a in enumerate([s.replace("a person is ", "") for s in meta["actions"]]):
    zi = turb[np.argmin(abs(gy - P_S[i,1])), np.argmin(abs(gx - P_S[i,0]))]
    ax.scatter(P_S[i,0], P_S[i,1], zi + 0.04, color="k", s=16, depthshade=False, zorder=5, edgecolors="white", linewidths=0.8)
    if a in ("walking", "dancing", "jumping", "sitting down"):
        ax.text(P_S[i,0], P_S[i,1], zi + 0.14, a, fontsize=8, ha="center", zorder=6)
ax.set_zlim(0, 1.05); ax.view_init(elev=38, azim=-60)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0, 0.5, 1])
ax.set_xlabel("latent PCA-1", fontsize=9, labelpad=-6)
ax.set_ylabel("latent PCA-2", fontsize=9, labelpad=-6)
ax.set_zlabel("style-flow turbulence\n$1-$coherence", fontsize=8.5, labelpad=2)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.pane.set_alpha(0.05)
ax.set_title("T2M-GPT latent: the coupling landscape", fontsize=10.5)

# ---------- right: stacked posterior densities ----------
ax2 = fig.add_subplot(122)
draws = np.load(ROOT / "results/bayes_p_model_draws.npy").reshape(-1, 4)
order = ["CLIP", "MLD", "MDM", "T2M-GPT"]          # pipeline depth, bottom-up
x = np.linspace(1e-4, 0.85, 900)
for k, name in enumerate(order):
    dvals = draws[:, k]
    bw = 1.06 * dvals.std() * len(dvals) ** -0.2
    dens = np.exp(-0.5*((x[:,None]-dvals[None])/bw)**2).sum(1)
    dens = dens / dens.max() * 0.85
    y0 = k * 1.0
    ax2.fill_between(x, y0, y0 + dens, color=COL[name], alpha=0.55, lw=0)
    ax2.plot(x, y0 + dens, color=COL[name], lw=1.3)
    ax2.axhline(y0, color="black", lw=0.6, alpha=0.5)
    ax2.text(0.86, y0 + 0.28, name, color=COL[name], fontsize=10,
             fontweight="bold", ha="left")
ax2.annotate("", xy=(0.985, 3.9), xytext=(0.985, 0.15),
             xycoords=("axes fraction", "data"),
             textcoords=("axes fraction", "data"),
             arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.1))
ax2.text(1.015, 2.0, "deeper in the pipeline,\nmore coupling", rotation=90,
         va="center", fontsize=8.5, color="0.4", transform=ax2.get_yaxis_transform())
ax2.set_xlim(0, 1.0); ax2.set_ylim(-0.15, 4.4)
ax2.set_yticks([])
ax2.set_xlabel("posterior escape probability")
ax2.set_title("Basin escape: four systems, one axis", fontsize=10.5)
for s in ("top", "right", "left"): ax2.spines[s].set_visible(False)
fig.tight_layout(w_pad=3)
fig.savefig(ROOT.parent / "overleaf/figures/coupling_landscape.pdf", bbox_inches="tight", dpi=220)
print("wrote coupling_landscape.pdf")
