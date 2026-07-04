"""
Regenerate all paper figures from results/ artifacts.

Usage:
  python make_figures.py --model clip --out ../overleaf/figures

Produces:
  {model}_kernels.pdf       K_S vs K_T^{weakest style} heatmaps
  {model}_perstyle_cka.pdf  per-style CKA bars (+ template error bars if
                            robustness.json exists)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def kernel_figure(res: Path, out: Path, model: str):
    K_S = np.load(res / "K_S.npy")
    K_T = np.load(res / "K_T.npy")
    meta = json.loads((res / "meta.json").read_text())
    report = json.loads((res / "report.json").read_text())
    styles = meta["styles"]
    short = [a.replace("a person is ", "") for a in meta["actions"]]

    weakest = min(report["per_style_cka"], key=report["per_style_cka"].get)
    j = styles.index(weakest)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    vmin = min(K_S.min(), K_T[j].min())
    for ax, K, title in [
        (axes[0], K_S, r"$K_S$ (neutral)"),
        (axes[1], K_T[j], rf"$K_T^{{\mathrm{{{weakest}}}}}$"),
    ]:
        im = ax.imshow(K, cmap="viridis", vmin=vmin, vmax=1.0)
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
    path = out / f"{model}_kernels.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}  (weakest style: {weakest})")


def perstyle_figure(res: Path, out: Path, model: str):
    report = json.loads((res / "report.json").read_text())
    styles = list(report["per_style_cka"].keys())

    rob_path = res / "robustness.json"
    if rob_path.exists():
        rob = json.loads(rob_path.read_text())["summary"]
        means = [rob[s]["cka_mean"] for s in styles]
        stds = [rob[s]["cka_std"] for s in styles]
    else:
        means = [report["per_style_cka"][s] for s in styles]
        stds = None

    weakest = styles[int(np.argmin(means))]
    colors = ["#c44" if s == weakest else "#4878a8" for s in styles]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(range(len(styles)), means, yerr=stds, capsize=3, color=colors,
           error_kw={"lw": 0.9})
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels(styles, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"CKA($K_S, K_T^j$)")
    ax.set_ylim(min(0.85, min(means) - 0.05), 1.005)
    ax.axhline(np.mean(means), ls="--", c="gray", lw=0.8)
    ax.text(len(styles) - 0.55, np.mean(means) + 0.003, "mean",
            fontsize=7, color="gray", ha="right")
    path = out / f"{model}_perstyle_cka.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mdm", "t2mgpt", "clip"])
    p.add_argument("--out", default="../overleaf/figures")
    args = p.parse_args()

    res = Path(__file__).resolve().parents[1] / "results" / args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernel_figure(res, out, args.model)
    perstyle_figure(res, out, args.model)


if __name__ == "__main__":
    main()
