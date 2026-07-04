"""
Reviewer-proofing statistics: bootstrap confidence intervals and
tau-sensitivity for the headline metrics.

1. Bootstrap over actions: resample the action set with replacement
   (rebuilding kernels and style vectors from the resampled rows) 2000
   times, reporting 95% percentile intervals for mean CKA and mean
   consistency. This answers "would your numbers survive a different
   choice of actions?" directly.

2. Tau sensitivity: GED at every candidate quantile threshold, showing
   conclusions do not depend on the density-selected tau.

Usage:
  python robustness_stats.py --systems clip mdm t2mgpt
  python robustness_stats.py --systems clip_v2   (expanded grid)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))

from cka import linear_cka
from ged import graph_edit_distance
from tau_select import tau_selection, TAU_QUANTILES
from style_vectors import style_vectors, consistency
from base import LatentExtractor

ROOT = Path(__file__).resolve().parents[1]


def load(system: str):
    base = ROOT / "results" / system
    if (base / "0").exists():
        base = base / "0"
    return np.load(base / "Z_S.npy"), np.load(base / "Z_T.npy")


def metrics_for(Z_S, Z_T):
    K_S = LatentExtractor.cosine_kernel(Z_S)
    ckas = [linear_cka(K_S, LatentExtractor.cosine_kernel(z)) for z in Z_T]
    cons, _ = consistency(style_vectors(Z_S, Z_T))
    return float(np.mean(ckas)), float(cons.mean())


def subsample_ci(Z_S, Z_T, n_draws=2000, seed=0):
    """
    Leave-2-out subsampling WITHOUT replacement (a bootstrap with
    replacement duplicates actions, and duplicated rows have pairwise
    cosine 1, which inflates pairwise statistics). Exhaustive for small n.
    Answers: would the numbers survive a different action subset?
    """
    from itertools import combinations
    rng = np.random.default_rng(seed)
    n = Z_S.shape[0]
    k = n - 2
    combos = list(combinations(range(n), k))
    if len(combos) > n_draws:
        combos = [tuple(rng.choice(n, size=k, replace=False))
                  for _ in range(n_draws)]
    ckas, conss = [], []
    for idx in combos:
        idx = list(idx)
        c, s = metrics_for(Z_S[idx], Z_T[:, idx])
        ckas.append(c); conss.append(s)
    lo = np.percentile(ckas, [2.5, 97.5]); hi = np.percentile(conss, [2.5, 97.5])
    return (round(float(lo[0]), 4), round(float(lo[1]), 4)), \
           (round(float(hi[0]), 4), round(float(hi[1]), 4))


def tau_sensitivity(Z_S, Z_T):
    K_S = LatentExtractor.cosine_kernel(Z_S)
    n = K_S.shape[0]
    upper = np.triu_indices(n, k=1)
    out = {}
    for q in TAU_QUANTILES:
        tau = float(np.quantile(K_S[upper], q))
        geds = [graph_edit_distance(K_S, LatentExtractor.cosine_kernel(z), tau)
                for z in Z_T]
        out[f"q{q}"] = round(float(np.mean(geds)), 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    results = {}
    for s in args.systems:
        Z_S, Z_T = load(s)
        cka, cons = metrics_for(Z_S, Z_T)
        cka_ci, cons_ci = subsample_ci(Z_S, Z_T, n_draws=args.n_boot)
        taus = tau_sensitivity(Z_S, Z_T)
        results[s] = {"cka": round(cka, 4), "cka_ci95": cka_ci,
                      "consistency": round(cons, 4), "consistency_ci95": cons_ci,
                      "ged_by_tau_quantile": taus}
        print(f"{s:<18} CKA {cka:.3f} {cka_ci}   consistency {cons:.3f} {cons_ci}")
        print(f"{'':<18} GED by tau quantile: {taus}")
    (ROOT / "results" / "robustness_stats.json").write_text(
        json.dumps(results, indent=1))
    print("\nsaved results/robustness_stats.json")


if __name__ == "__main__":
    main()
