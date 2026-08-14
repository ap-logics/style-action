"""ridge3d.pdf — the escape ladder as landscape, upgraded.
Four true posterior KDE ridges (NUTS draws) in pipeline depth order; on the
floor beneath each ridge, that system's REAL style vectors (2D PCA
directions of delta_j(a), subsampled) drawn as arrows: aligned at CLIP,
isotropic by T2M-GPT. Cause on the floor, effect in the mountains."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#0072B2", "MLD": "#56B4E9", "MDM": "#009E73", "T2M-GPT": "#E69F00"}
ORDER = ["CLIP", "MLD", "MDM", "T2M-GPT"]
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# ---- real 2D directions of style vectors per system ----
STYLE = "gracefully"   # one style's 24 vectors per rose: within-style coherence

def dirs2d(model, n=None, seed=0):
    if model == "CLIP":
        Z_S = np.load(ROOT / "results/clip_v2/0/Z_S.npy")
        Z_T = np.load(ROOT / "results/clip_v2/0/Z_T.npy")
        if Z_T.ndim == 2: Z_T = Z_T.reshape(13, 24, -1)
        import json
        styles = json.load(open(ROOT / "results/clip_v2/0/meta.json"))["styles"]
        d = (Z_T - Z_S[None])[styles.index(STYLE)]
    else:
        m = {"MLD": "mld_v2", "MDM": "mdm_v2", "T2M-GPT": "t2mgpt_v2"}[model]
        ZS = np.stack([np.load(q / "Z_S.npy") for q in sorted(ROOT.glob(f"results/{m}/seed4*/0"))])
        ZT = np.stack([np.load(q / "Z_T.npy") for q in sorted(ROOT.glob(f"results/{m}/seed4*/0"))])
        import json
        styles = json.load(open(next(ROOT.glob(f"results/{m}/seed4*/0")) / "meta.json"))["styles"]
        d = (ZT - ZS[:, None]).mean(0)[styles.index(STYLE)]
    # UNcentered SVD: PC1 keeps the shared direction, so a coherent bundle
    # stays a bundle in 2D instead of being centred into a star
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    v = d @ Vt[:2].T
    u = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    resultant_hd = float(np.linalg.norm(u.mean(0)))          # honest, high-dim
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9), resultant_hd

# ---- posterior KDEs from real draws ----
draws = np.load(ROOT / "results/bayes_p_model_draws.npy").reshape(-1, 4)
x = np.linspace(1e-4, 0.85, 900)
dens = []
for k in range(4):
    dv = draws[:, k]
    bw = 1.06 * dv.std() * len(dv) ** -0.2
    d = np.exp(-0.5 * ((x[:, None] - dv[None]) / bw) ** 2).sum(1)
    dens.append(d / d.max())

fig = plt.figure(figsize=(9.6, 6.2))
ax = fig.add_subplot(111, projection="3d")
ax.computed_zorder = False
verts = [[(x[0], 0.), *zip(x, d), (x[-1], 0.)] for d in dens]
colors = [COL[n] for n in ORDER]
# ridges back-to-front: CLIP deepest (y=3) ... T2M-GPT front (y=0)
ax.add_collection3d(PolyCollection(verts, facecolors=colors, edgecolors="white",
                    linewidths=1.2, alpha=0.9), zs=[3, 2, 1, 0], zdir="y")
for k, d in enumerate(dens):        # floor shadow tracks (the original's grounding)
    ax.plot(x, np.full_like(x, 3 - k), d * 0.018, color=colors[k], lw=1.6, alpha=0.35)
# floor arrow strips: real directions, drawn in the floor plane beneath each ridge
# compass rose per system: all measured style-vector directions from one origin
RX, RY = 0.085, 0.36
for k, name in enumerate(ORDER):
    y0 = 3 - k
    V, mn_hd = dirs2d(name)
    cx = 0.62 if name in ("CLIP", "MLD") else 0.14
    cy = y0 - 0.44
    for (u, w) in V:
        ax.plot([cx, cx + u * RX], [cy, cy + w * RY * 0.45], [0.012, 0.012],
                color=COL[name], alpha=0.45, lw=0.9, zorder=6)
    mu = V.mean(0); mn = mn_hd
    ax.plot([cx, cx + mu[0] * RX * 1.15], [cy, cy + mu[1] * RY * 0.5], [0.014, 0.014],
            color=COL[name], lw=2.6, zorder=7, solid_capstyle="round")
    ax.text(cx, cy - 0.40, 0.012, f"$|\\bar{{v}}|$ = {mn:.2f}", fontsize=7.5,
            color=COL[name], ha="center", zorder=8)
# peak labels (like the original) and out-of-scene annotations
import numpy as _np
means = [float((x * d).sum() / d.sum()) for d in dens]
for k, name in enumerate(ORDER):
    ax.text(means[k], 3 - k, 1.1, name, fontsize=11, fontweight="bold",
            color=COL[name], ha="center", zorder=9)
fig.text(0.5, 0.055,
         "floor roses: the 24 measured 'gracefully' vectors $\\delta_j(a)$ per system, one origin, uncentred PCA plane (display); "
         "$|\\bar{v}|$ is the mean resultant length in the full latent space. Tight rose, needle posterior; scattered rose, escape mass.",
         fontsize=8.5, color="0.32", ha="center")
ax.set_xlim(0, 0.9); ax.set_ylim(-0.5, 3.5); ax.set_zlim(0, 1.25)
ax.set_xlabel("posterior escape probability $p_m$", fontsize=10, labelpad=10)
ax.set_ylabel("pipeline depth $\\rightarrow$", fontsize=10, labelpad=8)
ax.set_zlabel("posterior density", fontsize=9, labelpad=2)
ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=32, azim=-55)
ax.set_box_aspect((2.15, 1.85, 0.8))
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_alpha(0.04)
ax.grid(False)
ax.set_title("The escape ladder, with its cause on the floor", fontsize=11)
plt.savefig(ROOT.parent / "overleaf/figures/ridge3d.pdf", bbox_inches="tight", dpi=220)
print("wrote ridge3d.pdf")
