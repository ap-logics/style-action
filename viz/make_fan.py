"""style_fan.pdf — the consistency collapse as geometry: each system's style
vectors form a cone around the style's mean direction; the fan half-angle IS
the consistency. Rays are the REAL angles of delta_j(a) to the style-mean
direction (data: /tmp cache or recomputed from results/*_v2 latents),
folded into 2D and mirrored for display. 90 deg = orthogonal = no axis."""
import numpy as np, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#352a87", "MLD": "#0f5cdd", "MDM": "#2eb7a4", "T2M-GPT": "#9aab3a"}

def fan_angles(Z_S, Z_T):
    delta = Z_T - Z_S[None]
    angs = []
    for j in range(delta.shape[0]):
        v = delta[j] / (np.linalg.norm(delta[j], axis=-1, keepdims=True) + 1e-9)
        mu = v.mean(0); mu /= (np.linalg.norm(mu) + 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(v @ mu, -1, 1))))
    return np.concatenate(angs)

data = {}
zs = np.load(ROOT / "results/clip_v2/0/Z_S.npy")
zt = np.load(ROOT / "results/clip_v2/0/Z_T.npy")
data["CLIP"] = fan_angles(zs, zt.reshape(13, 24, -1) if zt.ndim == 2 else zt)
for m, name in [("mld_v2", "MLD"), ("mdm_v2", "MDM"), ("t2mgpt_v2", "T2M-GPT")]:
    ZS = np.stack([np.load(p / "Z_S.npy") for p in sorted(ROOT.glob(f"results/{m}/seed4*/0"))])
    ZT = np.stack([np.load(p / "Z_T.npy") for p in sorted(ROOT.glob(f"results/{m}/seed4*/0"))])
    d = (ZT - ZS[:, None]).mean(0)
    data[name] = fan_angles(np.zeros_like(ZS[0]), d)

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), subplot_kw={"projection": "polar"})
rng = np.random.default_rng(0)
for ax, (name, angs) in zip(axes, data.items()):
    med = np.median(angs)
    # fold: mirror each ray left/right of the vertical mean direction
    sides = rng.choice([-1, 1], size=len(angs))
    th = np.radians(90 + sides * angs)           # 90 deg = up = mean direction
    ax.set_theta_zero_location("E")
    # median cone
    cone = np.radians(np.linspace(90 - med, 90 + med, 100))
    ax.fill_between(cone, 0, 1.0, color=COL[name], alpha=0.13, lw=0)
    for t in th:
        ax.plot([t, t], [0, 0.92], color=COL[name], lw=0.7, alpha=0.55)
    ax.plot([np.pi/2, np.pi/2], [0, 1.02], color="k", lw=1.8)   # mean direction
    ax.set_ylim(0, 1.1); ax.set_yticks([])
    ax.set_thetamin(-10); ax.set_thetamax(190)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180]))
    ax.set_xticklabels(["90°", "45°", "$\\bar{\\delta}_j$", "45°", "90°"], fontsize=8)
    ax.set_title(f"{name}\nmedian {med:.0f}°", fontsize=10, color=COL[name],
                 fontweight="bold", pad=12)
    ax.spines["polar"].set_visible(False)
fig.suptitle("The style cone: angle of each style vector $\\delta_j(a)$ to its style's mean direction "
             "(312 vectors per system; 90° = no shared axis)", fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(ROOT.parent / "overleaf/figures/style_fan.pdf", bbox_inches="tight")
print("wrote style_fan.pdf")
