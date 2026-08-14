"""cascade.pdf — the decoherence cascade: per-style consistency traced
through the pipeline. Stage 0: CLIP input. Stage 1: alignment (TMA-H3D,
canonical; other encoders as side markers). Stage 2: the three generator
branches (seed-averaged deltas; open markers = noise-corrected means).
Thin lines = the 13 individual styles; the collapse is universal, not
driven by outliers."""
import json, glob
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COL = {"CLIP": "#0072B2", "MLD": "#56B4E9", "MDM": "#009E73", "T2M-GPT": "#E69F00",
       "enc": "#5c6a8a", "hl": "#D55E00"}
CORR = {"MLD": 0.162, "MDM": 0.102, "T2M-GPT": 0.009}
plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

def per_style_from_report(model):
    r = json.load(open(ROOT / f"results/{model}/report.json"))
    c = r["consistency"]
    return {k: float(v) for k, v in c.items()}

def per_style_generator(model):
    ZS = np.stack([np.load(p / "Z_S.npy") for p in sorted(ROOT.glob(f"results/{model}/seed4*/0"))])
    ZT = np.stack([np.load(p / "Z_T.npy") for p in sorted(ROOT.glob(f"results/{model}/seed4*/0"))])
    meta = json.load(open(next(ROOT.glob(f"results/{model}/seed4*/0")) / "meta.json"))
    d = (ZT - ZS[:, None]).mean(0)                      # seed-avg (S,A,dim)
    out = {}
    for j, s in enumerate(meta["styles"]):
        v = d[j] / (np.linalg.norm(d[j], axis=-1, keepdims=True) + 1e-9)
        cs = v @ v.T
        A = v.shape[0]
        out[s] = float((cs.sum() - A) / (A * (A - 1)))
    return out

clip = per_style_from_report("clip_v2")
enc = per_style_from_report("opentma_h3d_v2")
gens = {"MLD": per_style_generator("mld_v2"),
        "MDM": per_style_generator("mdm_v2"),
        "T2M-GPT": per_style_generator("t2mgpt_v2")}
styles = list(clip.keys())
other_enc = {n: np.mean(list(per_style_from_report(m).values()))
             for m, n in [("opentma_unimocap_v2", "UniMoCap"), ("opentma_motionx_v2", "Motion-X")]}

X = {"clip": 0.0, "enc": 1.0, "MLD": 2.0, "MDM": 2.55, "T2M-GPT": 3.1}
HL = {"happily": "T2M-GPT's most destructive", "gracefully": None}

fig, ax = plt.subplots(figsize=(8.6, 4.6))
for s in styles:
    hl = s in HL
    c = COL["hl"] if hl else "0.62"
    lw = 1.4 if hl else 0.7
    a = 0.95 if hl else 0.5
    ax.plot([X["clip"], X["enc"]], [clip[s], enc[s]], color=c, lw=lw, alpha=a, zorder=3 if hl else 2)
    for g in gens:
        ax.plot([X["enc"], X[g]], [enc[s], gens[g][s]], color=c, lw=lw*0.85, alpha=a*0.8,
                zorder=3 if hl else 2)
    if hl:
        ax.annotate(f"\\textit{{{s}}}" if False else s, (X["clip"], clip[s]),
                    xytext=(-8, 0), textcoords="offset points", ha="right",
                    fontsize=8.5, style="italic", color=COL["hl"])
# system means, bold
m_clip = np.mean(list(clip.values())); m_enc = np.mean(list(enc.values()))
ax.plot([X["clip"], X["enc"]], [m_clip, m_enc], color="k", lw=2.6, zorder=5)
for g in gens:
    mg = np.mean(list(gens[g].values()))
    ax.plot([X["enc"], X[g]], [m_enc, mg], color=COL[g], lw=2.4, zorder=5)
    ax.plot(X[g], mg, "o", ms=7, color=COL[g], zorder=6)
    ax.plot(X[g], CORR[g], "o", ms=7, mfc="white", mec=COL[g], mew=1.5, zorder=6)
    ax.plot([X[g], X[g]], [mg, CORR[g]], color=COL[g], lw=0.8, alpha=0.6, zorder=4)
    ax.annotate(g, (X[g], -0.055), ha="center", fontsize=9.5, color=COL[g],
                fontweight="bold", annotation_clip=False)
ax.plot(X["clip"], m_clip, "o", ms=7, color=COL["CLIP"], zorder=6)
ax.plot(X["enc"], m_enc, "o", ms=7, color=COL["enc"], zorder=6)
# other encoders as context ticks at stage 1
for n, v in other_enc.items():
    ax.plot(X["enc"], v, "s", ms=5, mfc="white", mec=COL["enc"], mew=1.1, zorder=6)
    ax.annotate(n, (X["enc"], v), xytext=(7, -2), textcoords="offset points",
                fontsize=7.5, color=COL["enc"])
# stage labels + mean values
ax.annotate("CLIP\n(input)", (X["clip"], -0.055), ha="center", fontsize=9.5,
            color=COL["CLIP"], fontweight="bold", annotation_clip=False)
ax.annotate("alignment\n(TMA H3D)", (X["enc"], -0.055), ha="center", fontsize=9.5,
            color=COL["enc"], fontweight="bold", annotation_clip=False)
for xx, vv in [(X["clip"], m_clip), (X["enc"], m_enc)]:
    ax.annotate(f"{vv:.2f}", (xx, vv), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8.5, color="k")
# legend for corrected marker
ax.plot([], [], "o", ms=6, mfc="white", mec="0.3", label="noise-corrected mean")
ax.plot([], [], color="0.62", lw=0.8, label="individual styles (13)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_ylabel("style-direction consistency $\\mathcal{C}$")
ax.set_xlim(-0.55, 3.5); ax.set_ylim(-0.13, 0.92)
ax.set_xticks([])
ax.axhline(0, color="0.75", lw=0.6)
fig.tight_layout()
fig.savefig(ROOT.parent / "overleaf/figures/cascade.pdf", bbox_inches="tight", dpi=220)
print("wrote cascade.pdf")
