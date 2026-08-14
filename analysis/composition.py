"""
Do style operations compose, and which model of styling predicts the compound?

The main paper measures one modifier at a time and declines to test composition.
This asks the question directly, on held-out compound prompts of the form
"a person is <action> <adverb_j> and <adverb_k>", and it pits the paper's two
transformations against each other as *predictive* models rather than as
descriptive statistics.

Given neutral z_0(a), singles z_j(a), z_k(a) and the compound z_jk(a), write
delta_x(a) = z_x(a) - z_0(a). Candidate predictors of delta_jk:

  additive   delta_j + delta_k        styling is a translation, so composing
                                      two styles adds their displacements
  rotation   Q_k Q_j z_0  - z_0       styling is an orthogonal map, so composing
                                      two styles composes their rotations
  first      delta_j                  the compound is just the first modifier
  second     delta_k                  ... or just the second
  mean       (delta_j + delta_k)/2    additive up to scale (identical cosine)

Every predictor is scored two ways: cosine with the true compound displacement
(direction only) and relative residual ||pred - true|| / ||true|| (direction and
magnitude).

THE CONTROL THAT MATTERS. In these spaces the ambient anisotropy makes almost
any geometric statistic look impressive against a naive null, so each predictor
is also evaluated with the *wrong* singles: predict delta_jk from a different
randomly chosen pair (j', k'). A predictor is only informative to the extent it
beats its own shuffled-pair version. Reported as obs, shuffled, and the gap.

Order sensitivity is measured separately as cos(delta_jk, delta_kj) against the
same shuffled-pair baseline: if the two orders agree no better than unrelated
pairs do, composition is not even well defined at the prompt level.

Usage:
  python analysis/composition.py --root results_compound
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def _fit_Q(Z0: np.ndarray, Z1: np.ndarray):
    """Orthogonal map carrying the neutral configuration to the styled one,
    fitted inside the span of the neutral configuration (outside it 24 points
    determine nothing). Returns Q and the basis E."""
    A, B = _unit(Z0), _unit(Z1)
    U, s, _ = np.linalg.svd(A.T, full_matrices=False)
    k = int((s > 1e-8).sum())
    E = U[:, :k]
    a, b = A @ E, B @ E
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ vt, E


def _cos(P: np.ndarray, T: np.ndarray) -> float:
    return float((_unit(P) * _unit(T)).sum(axis=-1).mean())


def _resid(P: np.ndarray, T: np.ndarray) -> float:
    return float(np.linalg.norm(P - T) / (np.linalg.norm(T) + 1e-12))


def analyse(run_dir: Path, n_shuffle: int = 20, seed: int = 0) -> dict:
    Z_S = np.load(run_dir / "Z_S.npy")
    Z_T = np.load(run_dir / "Z_T.npy")
    meta = json.loads((run_dir / "meta.json").read_text())
    labels = meta["styles"]
    n_singles = meta.get("n_singles", sum(1 for l in labels if "|" not in l))
    singles = {labels[i]: i for i in range(n_singles)}
    pairs = [(i, *labels[i].split("|")) for i in range(n_singles, len(labels))]

    D = Z_T - Z_S[None]                      # (rows, actions, d)
    rng = np.random.default_rng(seed)

    # rotation model needs a Q per single, fitted once
    Qs, E = {}, None
    for s, i in singles.items():
        Qs[s], E = _fit_Q(Z_S, Z_T[i])
    A0 = _unit(Z_S) @ E                      # neutral configuration in the span

    preds = {"additive": [], "rotation": [], "first": [], "second": [], "mean": []}
    shuf = {k: [] for k in preds}
    order_obs, order_null = [], []

    for row, j, k in pairs:
        true = D[row]                                        # (actions, d)
        dj, dk = D[singles[j]], D[singles[k]]
        # rotation model: apply Q_j then Q_k in the span, lift back
        rot = ((A0 @ Qs[j] @ Qs[k]) - A0) @ E.T
        cand = {"additive": dj + dk, "rotation": rot, "first": dj,
                "second": dk, "mean": 0.5 * (dj + dk)}
        for name, P in cand.items():
            preds[name].append((_cos(P, true), _resid(P, true)))

        # shuffled control: same predictor built from a different random pair
        for _ in range(n_shuffle):
            r2, j2, k2 = pairs[rng.integers(len(pairs))]
            dj2, dk2 = D[singles[j2]], D[singles[k2]]
            rot2 = ((A0 @ Qs[j2] @ Qs[k2]) - A0) @ E.T
            c2 = {"additive": dj2 + dk2, "rotation": rot2, "first": dj2,
                  "second": dk2, "mean": 0.5 * (dj2 + dk2)}
            for name, P in c2.items():
                shuf[name].append((_cos(P, true), _resid(P, true)))

        # order sensitivity
        rev = f"{k}|{j}"
        if rev in labels:
            order_obs.append(_cos(D[labels.index(rev)], true))
            r3 = pairs[rng.integers(len(pairs))][0]
            order_null.append(_cos(D[r3], true))

    out = {"n_pairs": len(pairs), "n_actions": int(Z_S.shape[0]), "models": {}}
    for name in preds:
        o = np.array(preds[name]); s = np.array(shuf[name])
        out["models"][name] = {
            "cos": round(float(o[:, 0].mean()), 4),
            "cos_shuffled": round(float(s[:, 0].mean()), 4),
            "cos_gap": round(float(o[:, 0].mean() - s[:, 0].mean()), 4),
            "resid": round(float(o[:, 1].mean()), 4),
            "resid_shuffled": round(float(s[:, 1].mean()), 4),
        }
    out["order"] = {
        "cos_forward_vs_reverse": round(float(np.mean(order_obs)), 4),
        "cos_vs_unrelated_pair": round(float(np.mean(order_null)), 4),
        "n": len(order_obs),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_compound")
    p.add_argument("--systems", nargs="*",
                   default=["clip", "opentma_h3d", "opentma_motionx", "opentma_unimocap"])
    args = p.parse_args()

    root = ROOT / args.root
    allout = {}
    for sysname in args.systems:
        d = root / sysname / "0"
        if not (d / "Z_S.npy").exists():
            print(f"{sysname:18s} SKIP (no Z_S.npy at {d})")
            continue
        r = analyse(d)
        allout[sysname] = r
        print(f"\n=== {sysname}  ({r['n_pairs']} ordered pairs x {r['n_actions']} actions)")
        print(f"  {'model':10s} {'cos':>7s} {'shuffled':>9s} {'gap':>7s} {'resid':>7s} {'shuf':>7s}")
        for name, m in sorted(r["models"].items(), key=lambda kv: -kv[1]["cos_gap"]):
            print(f"  {name:10s} {m['cos']:7.3f} {m['cos_shuffled']:9.3f} "
                  f"{m['cos_gap']:+7.3f} {m['resid']:7.3f} {m['resid_shuffled']:7.3f}")
        o = r["order"]
        print(f"  order: cos(j|k, k|j) = {o['cos_forward_vs_reverse']:.3f}  "
              f"vs unrelated pair {o['cos_vs_unrelated_pair']:.3f}")

    (ROOT / "results" / "composition.json").write_text(json.dumps(allout, indent=1))
    print("\nsaved results/composition.json")


if __name__ == "__main__":
    main()
