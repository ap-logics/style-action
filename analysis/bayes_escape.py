"""
Hierarchical Bayesian analysis of basin escape.

Likelihood: escape_{m,t,j,a} ~ Bernoulli(p_{m,j,a})   (t = template, replicate)
  logit(p_{m,j,a}) = alpha_m + beta_{m,j} + gamma_{m,a}

Partial pooling: style effects beta and action effects gamma are drawn from
model-specific Normal(0, sigma) priors whose scales are themselves inferred,
so sparse cells (7 or 8 binary observations each) are shrunk toward the
model mean in proportion to how little data supports them. This is the
estimation layer; the permutation nulls in the main pipeline remain the
calibration layer.

Outputs:
  results/bayes_escape_summary.json   posterior means + 94% HDIs
  ../overleaf/figures/bayes_forest.pdf  forest plot of per-style effects

Usage:
  python bayes_escape.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import os
MODELS = os.environ.get("BAYES_MODELS", "clip,mdm,t2mgpt").split(",")


def esc_matrix(zdir: Path) -> np.ndarray:
    Z_S = np.load(zdir / "Z_S.npy")
    Z_T = np.load(zdir / "Z_T.npy")
    Zs = Z_S / np.linalg.norm(Z_S, axis=1, keepdims=True)
    return np.stack([
        (Z_T[j] / np.linalg.norm(Z_T[j], axis=1, keepdims=True)
         @ Zs.T).argmax(axis=1) != np.arange(len(Z_S))
        for j in range(Z_T.shape[0])]).astype(int)


def find_runs(model: str) -> list[Path]:
    """All (seed x template) leaf dirs containing Z files, any layout."""
    root = ROOT / "results" / model
    leaves = sorted(root.glob("seed*/[0-9]")) or sorted(root.glob("[0-9]"))
    return [d for d in leaves if (d / "Z_S.npy").exists()] or (
        [root] if (root / "Z_S.npy").exists() else [])


def load_escapes():
    """Long-format arrays over every replicate run of every model."""
    m_idx, s_idx, a_idx, y = [], [], [], []
    styles = actions = None
    for mi, model in enumerate(MODELS):
        runs = find_runs(model)
        meta = json.loads((runs[0] / "meta.json").read_text())
        styles = meta["styles"]
        actions = [a.replace("a person is ", "") for a in meta["actions"]]
        for run in runs:
            mat = esc_matrix(run)
            for j in range(mat.shape[0]):
                for a in range(mat.shape[1]):
                    m_idx.append(mi); s_idx.append(j); a_idx.append(a)
                    y.append(int(mat[j, a]))
        print(f"{model}: {len(runs)} replicate runs")
    return (np.array(m_idx), np.array(s_idx), np.array(a_idx),
            np.array(y), styles, actions)


def main():
    m_idx, s_idx, a_idx, y, styles, actions = load_escapes()
    n_m, n_s, n_a = len(MODELS), len(styles), len(actions)
    print(f"{len(y)} Bernoulli observations "
          f"({n_m} models x templates x {n_s} styles x {n_a} actions)")

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0, shape=n_m)
        sigma_s = pm.HalfNormal("sigma_style", 1.0, shape=n_m)
        sigma_a = pm.HalfNormal("sigma_action", 1.0, shape=n_m)
        beta_z = pm.Normal("beta_z", 0, 1, shape=(n_m, n_s))
        gamma_z = pm.Normal("gamma_z", 0, 1, shape=(n_m, n_a))
        beta = pm.Deterministic("beta", beta_z * sigma_s[:, None])
        gamma = pm.Deterministic("gamma", gamma_z * sigma_a[:, None])

        logit_p = alpha[m_idx] + beta[m_idx, s_idx] + gamma[m_idx, a_idx]
        pm.Bernoulli("y", logit_p=logit_p, observed=y)

        # model-level escape probability at a typical style/action
        pm.Deterministic("p_model", pm.math.sigmoid(alpha))

        trace = pm.sample(2000, tune=2000, chains=4, cores=4,
                          target_accept=0.95, progressbar=False,
                          random_seed=7)

    summ = az.summary(trace, var_names=["p_model", "sigma_style",
                                        "sigma_action"], ci_prob=0.94)
    print(summ.to_string())

    # per-style escape probabilities per model (marginal over typical action)
    post = trace.posterior
    p_style = 1 / (1 + np.exp(-(post["alpha"].values[..., :, None]
                                + post["beta"].values)))  # (c, d, m, s)
    out = {}
    for mi, mname in enumerate(MODELS):
        out[mname] = {}
        for j, s in enumerate(styles):
            d = np.sort(p_style[..., mi, j].ravel())
            # narrowest window containing 94% of samples
            k = int(np.ceil(0.94 * len(d)))
            i = int(np.argmin(d[k - 1:] - d[:len(d) - k + 1]))
            out[mname][s] = {
                "mean": round(float(d.mean()), 4),
                "hdi94": [round(float(d[i]), 4), round(float(d[i + k - 1]), 4)],
            }
    (ROOT / "results" / "bayes_escape_summary.json").write_text(
        json.dumps(out, indent=1))

    # forest plot: per-style escape probability, three models side by side
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    palette = ["#4878a8", "#f0a13a", "#c43030"]
    colors = {m: palette[i % 3] for i, m in enumerate(MODELS)}
    for mi, mname in enumerate(MODELS):
        means = [out[mname][s]["mean"] for s in styles]
        lo = [out[mname][s]["hdi94"][0] for s in styles]
        hi = [out[mname][s]["hdi94"][1] for s in styles]
        ypos = np.arange(len(styles)) + (mi - 1) * 0.22
        ax.errorbar(means, ypos,
                    xerr=[np.array(means) - lo, np.array(hi) - means],
                    fmt="o", ms=4.5, lw=1.4, capsize=2.5,
                    color=colors[mname], label=mname.upper())
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels(styles, fontsize=9)
    ax.set_xlabel("posterior escape probability (94% HDI)")
    ax.set_xlim(0, 1)
    ax.axvline(1 / 8, ls=":", c="gray", lw=0.8)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Hierarchical posterior: per-style basin escape", fontsize=10)
    figpath = ROOT.parent / "overleaf" / "figures" / "bayes_forest.pdf"
    plt.savefig(figpath, bbox_inches="tight")
    print(f"wrote {figpath}")


if __name__ == "__main__":
    main()
