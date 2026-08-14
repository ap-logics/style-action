"""Permutation null for MoMask, matching the procedure used for the other systems.

Shuffle the action labels, recompute CKA/GED/consistency 1000 times, take the
2.5/97.5 percentiles. The null is system-specific because it depends on the
geometry of that system's own kernel, so MoMask's observed CKA cannot be read
against another model's interval.

Writes both the full RVQ latent and the base-layer-only latent, so the
residual-depth comparison has a null on each side.

Usage:
  python3 null_momask.py --run results/momask_v2/seed42/0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))

from cka import linear_cka
from ged import graph_edit_distance
from tau_select import tau_selection
from style_vectors import style_vectors, consistency
from base import LatentExtractor

kern = LatentExtractor.cosine_kernel


def null_interval(Z_S, Z_T, n_permutations=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = Z_S.shape[0]
    K_S = kern(Z_S)
    tau, _ = tau_selection(K_S)

    cka_n, ged_n, cons_n = [], [], []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        KSp = K_S[np.ix_(perm, perm)]
        K_T = kern(Z_T[rng.integers(Z_T.shape[0])])
        cka_n.append(linear_cka(KSp, K_T))
        ged_n.append(graph_edit_distance(KSp, K_T, tau))
        cons_n.append(float(np.mean(consistency(style_vectors(Z_S[perm], Z_T))[0])))

    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"cka": q(cka_n), "ged": q(ged_n), "cons": q(cons_n)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="one seed x template directory")
    p.add_argument("--perms", type=int, default=1000)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = (root / args.run) if not Path(args.run).is_absolute() else Path(args.run)

    out = {}
    for label, suffix in [("MoMask", ""), ("MoMask-base", "_base")]:
        Z_S = np.load(run / f"Z_S{suffix}.npy")
        Z_T = np.load(run / f"Z_T{suffix}.npy")
        out[label] = null_interval(Z_S, Z_T, args.perms)
        n = out[label]
        print(f"{label}:  cka [{n['cka'][0]:.3f}, {n['cka'][1]:.3f}]  "
              f"ged [{n['ged'][0]:.3f}, {n['ged'][1]:.3f}]  "
              f"cons [{n['cons'][0]:.4f}, {n['cons'][1]:.4f}]")

    path = root / "results" / "null_intervals.json"
    existing = json.loads(path.read_text())
    existing.update(out)
    path.write_text(json.dumps(existing, indent=1))
    print(f"merged into {path}")


if __name__ == "__main__":
    main()
