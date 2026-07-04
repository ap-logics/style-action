"""
The manifold shot: a hillshaded 3D potential landscape of the action space
with styled embeddings as particles, side-by-side for two systems.

Wells are Gaussian basins at each neutral action embedding (PCA plane).
Grey particles stayed in their own basin; red particles escaped (computed
in the full space). Domain is clipped to percentiles of the styled cloud
so outliers cannot flatten the wells.

Usage:
  python make_manifold.py --left clip --right t2mgpt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource

PARULA = LinearSegmentedColormap.from_list("parula", [
    "#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4",
    "#87bf77", "#d1bb59", "#fec832", "#f9fb0e"])


def load(system: str, template: int = 0):
    base = Path(__file__).resolve().parents[1] / "results" / system
    if (base / str(template)).exists():
        base = base / str(template)
    Z_S = np.load(base / "Z_S.npy")
    Z_T = np.load(base / "Z_T.npy")
    meta = json.loads((base / "meta.json").read_text())
    return Z_S, Z_T, [a.replace("a person is ", "") for a in meta["actions"]]


def escapes(Z_S, Z_T):
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    out = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        out.append((Zt @ Zs.T).argmax(axis=1) != np.arange(Z_S.shape[0]))
    return np.stack(out)


def panel(ax, system: str, elev: float, azim: float):
    Z_S, Z_T, names = load(system)
    esc = escapes(Z_S, Z_T)

    allZ = np.vstack([Z_S, Z_T.reshape(-1, Z_T.shape[-1])])
    mu = allZ.mean(0)
    _, _, Vt = np.linalg.svd(allZ - mu, full_matrices=False)
    proj = lambda Z: (Z - mu) @ Vt[:2].T
    P_S = proj(Z_S)
    P_T = np.stack([proj(Z_T[j]) for j in range(Z_T.shape[0])])

    # domain: neutral extent padded, styled cloud clipped to 5-95 pct
    flat = P_T.reshape(-1, 2)
    lo = np.minimum(P_S.min(0), np.percentile(flat, 12, axis=0))
    hi = np.maximum(P_S.max(0), np.percentile(flat, 88, axis=0))
    pad = 0.18 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    d = np.linalg.norm(P_S[:, None] - P_S[None], axis=-1)
    sigma = np.median(d[d > 0]) * 0.34

    n = 320
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    XX, YY = np.meshgrid(xs, ys)

    def potential(px, py):
        p = np.stack([np.atleast_1d(px), np.atleast_1d(py)], -1)
        r2 = ((p[..., None, :] - P_S) ** 2).sum(-1)
        out = -np.exp(-r2 / (2 * sigma ** 2)).sum(-1)
        return out if out.ndim > 1 else out.ravel()

    ZZ = potential(XX, YY)
    zspan = ZZ.max() - ZZ.min() + 1e-9

    # hillshaded facecolors
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(ZZ, cmap=PARULA, vert_exag=0.55, blend_mode="soft")
    ax.plot_surface(XX, YY, ZZ, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, alpha=0.96)
    ax.contour(XX, YY, ZZ, zdir="z", offset=ZZ.min() - 0.35 * zspan,
               levels=12, cmap=PARULA, linewidths=0.7)

    # action labels at well bottoms
    for a, name in enumerate(names):
        z = potential(P_S[a, 0:1], P_S[a, 1:2])[0]
        ax.text(P_S[a, 0], P_S[a, 1], z - 0.16 * zspan, name,
                fontsize=7.5, ha="center", color="#1a1a1a", zorder=20)

    # particles, clipped into the domain
    for j in range(Z_T.shape[0]):
        for a in range(Z_S.shape[0]):
            x, y = np.clip(P_T[j, a], lo, hi)
            z = potential(np.array([x]), np.array([y]))[0] + 0.03 * zspan
            if esc[j, a]:
                ax.scatter([x], [y], [z], s=55, c="#ff1f1f",
                           edgecolors="#5c0000", linewidths=0.6,
                           depthshade=False, zorder=30)
            else:
                ax.scatter([x], [y], [z], s=11, c="#f2f2f2",
                           edgecolors="#666", linewidths=0.35,
                           depthshade=False, zorder=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.3, 1.0, 0.42))
    ax.set_axis_off()
    n_esc = int(esc.sum())
    total = esc.size
    ax.set_title(f"{system.upper().replace('_', ' ')}   "
                 f"({n_esc}/{total} escapes)", fontsize=11, pad=-4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+",
                   default=["clip", "mdm", "t2mgpt"])
    p.add_argument("--elev", type=float, default=48)
    p.add_argument("--azim", type=float, default=-58)
    p.add_argument("--out", default="../overleaf/figures")
    args = p.parse_args()

    n = len(args.systems)
    fig = plt.figure(figsize=(5.6 * n, 4.6))
    for i, system in enumerate(args.systems):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        panel(ax, system, args.elev, args.azim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=-0.08)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / ("manifold_" + "_".join(args.systems) + ".pdf")
    plt.savefig(path, bbox_inches="tight", dpi=220)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
