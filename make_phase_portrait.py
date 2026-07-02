"""
Phase portrait of style flow on the action manifold.

The 8 neutral embeddings are projected to 2D with PCA and drawn as labeled
basin centres. Each style vector delta_j(a) = z_j(a) - z_0(a) becomes an
arrow rooted at action a (projected through the same PCA). In a well-
factored representation the arrows form a coherent flow (all styles push
the same way everywhere); coupling appears as arrows swirling toward other
actions' basins. Arrows whose endpoint is nearer another action's neutral
embedding than its own (basin escapes, in the full space not the 2D
projection) are drawn red.

Usage:
  python make_phase_portrait.py --model clip --out ../overleaf/figures
  python make_phase_portrait.py --model mdm --template 0 --style tiredly
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(root: Path, template: int | None):
    base = root / str(template) if template is not None and \
        (root / str(template)).exists() else root
    Z_S = np.load(base / "Z_S.npy")
    Z_T = np.load(base / "Z_T.npy")
    meta = json.loads((base / "meta.json").read_text())
    return Z_S, Z_T, meta


def escapes(Z_S: np.ndarray, Z_T: np.ndarray) -> np.ndarray:
    """(n_styles, n_actions) bool — computed in the FULL space."""
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    out = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        out.append((Zt @ Zs.T).argmax(axis=1) != np.arange(Z_S.shape[0]))
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--template", type=int, default=0)
    p.add_argument("--style", default=None,
                   help="single style; omit for all styles overlaid")
    p.add_argument("--out", default="../overleaf/figures")
    args = p.parse_args()

    root = Path(__file__).parent / "results" / args.model
    Z_S, Z_T, meta = load(root, args.template)
    styles = meta["styles"]
    short = [a.replace("a person is ", "") for a in meta["actions"]]
    esc = escapes(Z_S, Z_T)

    # PCA fit on ALL embeddings so styled points project meaningfully
    allZ = np.vstack([Z_S, Z_T.reshape(-1, Z_T.shape[-1])])
    mu = allZ.mean(0)
    _, _, Vt = np.linalg.svd(allZ - mu, full_matrices=False)
    proj = lambda Z: (Z - mu) @ Vt[:2].T

    P_S = proj(Z_S)
    which = [styles.index(args.style)] if args.style else range(len(styles))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(P_S[:, 0], P_S[:, 1], s=90, c="#222", zorder=5)
    for a, name in enumerate(short):
        ax.annotate(name, P_S[a], textcoords="offset points",
                    xytext=(7, 7), fontsize=9, zorder=6)

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(styles)))
    for j in which:
        P_T = proj(Z_T[j])
        for a in range(len(short)):
            color = "#c43030" if esc[j, a] else cmap[j]
            ax.annotate(
                "", xy=P_T[a], xytext=P_S[a], zorder=4,
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.6 if esc[j, a] else 1.0,
                                alpha=0.95 if esc[j, a] else 0.65))
    if not args.style:
        for j, s in enumerate(styles):
            ax.plot([], [], color=cmap[j], label=s)
        ax.plot([], [], color="#c43030", lw=1.6, label="basin escape")
        ax.legend(fontsize=7, loc="best", framealpha=0.9)

    title_style = args.style or "all styles"
    ax.set_title(f"{args.model.upper()}: style flow ({title_style}), "
                 f"template {args.template}", fontsize=10)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_aspect("equal", adjustable="datalim")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    suffix = args.style or "all"
    path = out / f"{args.model}_phase_{suffix}.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"wrote {path}  (escapes shown in red: {int(esc[list(which)].sum())})")


if __name__ == "__main__":
    main()
