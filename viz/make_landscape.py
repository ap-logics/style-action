"""
Physics-style visualisations of style flow.

1. Flow field (--kind flow): the 56 style vectors delta_j(a), projected to
   the PCA plane, interpolated (RBF) into a continuous vector field and
   drawn as streamlines over speed shading. Interpolation is illustrative:
   the field is measured at 8 points x 7 styles; streamlines show tendency,
   not trajectories.

2. Basin landscape (--kind basin, 3D): a potential surface with a Gaussian
   well at each neutral action embedding. Styled embeddings are particles
   dropped on the surface: grey when they remain in their own well, red
   when they have escaped to another action's basin (computed in the FULL
   space, not the projection).

Usage:
  python make_landscape.py --model t2mgpt --kind flow
  python make_landscape.py --model t2mgpt --kind basin --elev 55 --azim -60
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# MATLAB parula-inspired colormap
PARULA = LinearSegmentedColormap.from_list("parula", [
    "#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4",
    "#87bf77", "#d1bb59", "#fec832", "#f9fb0e"])


def load(root: Path, template: int):
    base = root / str(template) if (root / str(template)).exists() else root
    Z_S = np.load(base / "Z_S.npy")
    Z_T = np.load(base / "Z_T.npy")
    meta = json.loads((base / "meta.json").read_text())
    return Z_S, Z_T, meta


def pca2(Z_S, Z_T):
    allZ = np.vstack([Z_S, Z_T.reshape(-1, Z_T.shape[-1])])
    mu = allZ.mean(0)
    _, _, Vt = np.linalg.svd(allZ - mu, full_matrices=False)
    return lambda Z: (Z - mu) @ Vt[:2].T


def escapes(Z_S, Z_T):
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    out = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        out.append((Zt @ Zs.T).argmax(axis=1) != np.arange(Z_S.shape[0]))
    return np.stack(out)


def rbf_field(roots, vecs, XX, YY, sigma=None):
    """
    Gaussian-RBF interpolation of a vector field from scattered samples.
    Also returns local COHERENCE = |weighted mean vector| / weighted mean
    |vector| — 1 where nearby samples agree in direction (laminar), 0 where
    they cancel (turbulent). This is the field-level view of style-vector
    consistency.
    """
    pts = np.stack([XX.ravel(), YY.ravel()], 1)
    if sigma is None:
        d = np.linalg.norm(roots[:, None] - roots[None], axis=-1)
        sigma = np.median(d[d > 0]) * 0.35
    w = np.exp(-np.linalg.norm(pts[:, None] - roots[None], axis=-1) ** 2
               / (2 * sigma ** 2))
    w = w / (w.sum(1, keepdims=True) + 1e-12)
    F = w @ vecs
    mean_norm = np.linalg.norm(F, axis=1)
    norm_mean = w @ np.linalg.norm(vecs, axis=1)
    coherence = mean_norm / (norm_mean + 1e-12)
    return (F[:, 0].reshape(XX.shape), F[:, 1].reshape(XX.shape),
            coherence.reshape(XX.shape))


def flow(model, root, template, out):
    Z_S, Z_T, meta = load(root, template)
    proj = pca2(Z_S, Z_T)
    P_S, esc = proj(Z_S), escapes(Z_S, Z_T)
    short = [a.replace("a person is ", "") for a in meta["actions"]]

    roots, vecs = [], []
    for j in range(Z_T.shape[0]):
        P_T = proj(Z_T[j])
        roots.append(P_S)
        vecs.append(P_T - P_S)
    roots, vecs = np.vstack(roots), np.vstack(vecs)

    pad = 0.25 * (P_S.max(0) - P_S.min(0))
    n = 240
    x0, x1 = P_S[:, 0].min() - pad[0], P_S[:, 0].max() + pad[0]
    y0, y1 = P_S[:, 1].min() - pad[1], P_S[:, 1].max() + pad[1]
    # exactly uniform spacing — streamplot rejects linspace rounding error
    xs = x0 + (x1 - x0) / (n - 1) * np.arange(n)
    ys = y0 + (y1 - y0) / (n - 1) * np.arange(n)
    XX, YY = np.meshgrid(xs, ys)
    U, V, coherence = rbf_field(roots, vecs, XX, YY)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.pcolormesh(XX, YY, coherence, cmap=PARULA, shading="gouraud",
                       vmin=0, vmax=1, alpha=0.9, rasterized=True)
    ax.streamplot(xs, ys, U, V, color="white", density=1.4, linewidth=0.7,
                  arrowsize=0.9)
    ax.scatter(P_S[:, 0], P_S[:, 1], s=110, c="#111", zorder=6,
               edgecolors="white", linewidths=1.2)
    for a, name in enumerate(short):
        ax.annotate(name, P_S[a], textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="#111", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    n_esc = int(esc.sum())
    ax.set_title(f"{model.upper()}: style flow and local coherence "
                 f"({n_esc}/56 basin escapes)", fontsize=11)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    fig.colorbar(im, ax=ax, label="directional coherence (1 = laminar)",
                 shrink=0.85)
    path = out / f"{model}_flowfield.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"wrote {path}")


def basin(model, root, template, out, elev, azim):
    Z_S, Z_T, meta = load(root, template)
    proj = pca2(Z_S, Z_T)
    P_S, esc = proj(Z_S), escapes(Z_S, Z_T)
    short = [a.replace("a person is ", "") for a in meta["actions"]]

    d = np.linalg.norm(P_S[:, None] - P_S[None], axis=-1)
    sigma = np.median(d[d > 0]) * 0.35

    # grid must cover the styled points too, or escapees float off-surface
    P_all = np.vstack([P_S] + [proj(Z_T[j]) for j in range(Z_T.shape[0])])
    pad = 0.12 * (P_all.max(0) - P_all.min(0))
    xs = np.linspace(P_all[:, 0].min() - pad[0], P_all[:, 0].max() + pad[0], 260)
    ys = np.linspace(P_all[:, 1].min() - pad[1], P_all[:, 1].max() + pad[1], 260)
    XX, YY = np.meshgrid(xs, ys)

    def potential(x, y):
        p = np.stack([np.atleast_1d(x), np.atleast_1d(y)], -1)  # (..., 2)
        r2 = ((p[..., None, :] - P_S) ** 2).sum(-1)             # (..., 8)
        out = -np.exp(-r2 / (2 * sigma ** 2)).sum(-1)
        return out if out.ndim > 1 else out.ravel()

    ZZ = potential(XX, YY)

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, ZZ, cmap=PARULA, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, alpha=0.92)
    ax.contour(XX, YY, ZZ, zdir="z", offset=ZZ.min() - 0.25, cmap=PARULA,
               levels=10, linewidths=0.6)

    for a, name in enumerate(short):
        z = potential(P_S[a, 0:1], P_S[a, 1:2])[0]
        ax.text(P_S[a, 0], P_S[a, 1], z - 0.28, name, fontsize=8, ha="center")

    for j in range(Z_T.shape[0]):
        P_T = proj(Z_T[j])
        zt = potential(P_T[:, 0], P_T[:, 1])
        stay, run = ~esc[j], esc[j]
        ax.scatter(P_T[stay, 0], P_T[stay, 1], zt[stay] + 0.04, s=14,
                   c="#dddddd", edgecolors="#555", linewidths=0.4,
                   depthshade=False)
        ax.scatter(P_T[run, 0], P_T[run, 1], zt[run] + 0.04, s=26,
                   c="#e03030", edgecolors="#500", linewidths=0.5,
                   depthshade=False)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{model.upper()}: action potential landscape "
                 f"(red = escaped particles, {int(esc.sum())}/56)",
                 fontsize=11, pad=0)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("potential")
    ax.set_box_aspect((1.25, 1, 0.5))
    path = out / f"{model}_basin3d.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"wrote {path}")


def progression(models, template, out):
    """Single row of coherence fields, shared scale + one labelled colorbar."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.0))
    ims = []
    for ax, model in zip(axes, models):
        root = Path(__file__).resolve().parents[1] / "results" / model
        Z_S, Z_T, meta = load(root, template)
        proj = pca2(Z_S, Z_T)
        P_S, esc = proj(Z_S), escapes(Z_S, Z_T)
        short = [a.replace("a person is ", "") for a in meta["actions"]]

        roots, vecs = [], []
        for j in range(Z_T.shape[0]):
            P_T = proj(Z_T[j])
            roots.append(P_S); vecs.append(P_T - P_S)
        roots, vecs = np.vstack(roots), np.vstack(vecs)

        pad = 0.25 * (P_S.max(0) - P_S.min(0))
        n = 240
        x0, x1 = P_S[:, 0].min() - pad[0], P_S[:, 0].max() + pad[0]
        y0, y1 = P_S[:, 1].min() - pad[1], P_S[:, 1].max() + pad[1]
        xs = x0 + (x1 - x0) / (n - 1) * np.arange(n)
        ys = y0 + (y1 - y0) / (n - 1) * np.arange(n)
        XX, YY = np.meshgrid(xs, ys)
        U, V, coh = rbf_field(roots, vecs, XX, YY)

        im = ax.pcolormesh(XX, YY, coh, cmap=PARULA, shading="gouraud",
                           vmin=0, vmax=1, alpha=0.9, rasterized=True)
        ims.append(im)
        ax.streamplot(xs, ys, U, V, color="white", density=1.3,
                      linewidth=0.7, arrowsize=0.9)
        ax.scatter(P_S[:, 0], P_S[:, 1], s=100, c="#111", zorder=6,
                   edgecolors="white", linewidths=1.2)
        for a, name in enumerate(short):
            ax.annotate(name, P_S[a], textcoords="offset points",
                        xytext=(8, 8), fontsize=9.5, color="#000",
                        fontweight="medium", zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.4))
        n_esc = int(esc.sum())
        label = {"clip": "CLIP (input)", "mdm": "MDM",
                 "t2mgpt": "T2M-GPT"}.get(model, model.upper())
        ax.set_title(f"{label}   {n_esc}/{esc.size} escapes", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(ims[-1], ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("directional coherence\n0 = turbulent   1 = coherent",
                 fontsize=9)
    path = out / ("flowfield_" + "_".join(models) + ".pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False)
    p.add_argument("--models", nargs="+", help="multi-panel progression")
    p.add_argument("--kind", choices=["flow", "basin"], required=False)
    p.add_argument("--template", type=int, default=0)
    p.add_argument("--elev", type=float, default=55)
    p.add_argument("--azim", type=float, default=-60)
    p.add_argument("--out", default="../overleaf/figures")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.models:
        progression(args.models, args.template, out)
        return
    root = Path(__file__).resolve().parents[1] / "results" / args.model
    if args.kind == "flow":
        flow(args.model, root, args.template, out)
    else:
        basin(args.model, root, args.template, out, args.elev, args.azim)


if __name__ == "__main__":
    main()
