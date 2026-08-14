"""
How far is styling from being a rotation, and from being a translation?

Section 3 of the paper says relational invariance holds exactly when the styled
action configuration is an orthogonal image of the neutral one, and (exact)
equivariance exactly when it is a translate. CKA and consistency each report a
normalised score for one of those, on scales that cannot be compared. This
script reports both as residuals on one scale: how far the styled configuration
is from satisfying each definition exactly.

For style j, with A = unit-normalised neutral embeddings and B = unit-normalised
styled embeddings (both n x d), and n_A = n_B = n since rows are unit norm:

    r_rot  = min_{Q in O(d)} ||B - A Q||_F^2  = 2n - 2 * sum_i sigma_i(A^T B)
    r_tra  = min_{delta}     ||B - (A + 1 delta^T)||_F^2

Two normalisations are reported, and they answer different questions.

  eps_* = r_* / (2n)      distance to the definition, on a fixed denominator
  rho_* = 1 - r_* / M     fraction of the movement M = ||B - A||_F^2 explained

eps is the one to trust. Its denominator does not depend on which styled
embedding is paired with which neutral one, so the permutation null (shuffle the
pairing, as everywhere else in the paper) is meaningful: every observed eps in
every system falls below its own null, as a residual should.

rho is reported because it is the more intuitive reading ("86% of the movement is
one global rotation"), but its denominator M grows under permutation, which
inflates rho for the shuffled pairing. rho_rot consequently sits at or BELOW its
own permutation null in five of the eight systems and must not be read as
evidence. rho_tra behaves, and closely tracks the paper's consistency (0.640 vs
0.644 for CLIP), which is what it is a relative of.

Two caveats on eps, both reported in the paper:
  1. O(d) carries d(d-1)/2 free parameters against the translation group's d, so
     eps_rot < eps_tra is expected a priori. The two are distances to each
     definition, not a model comparison.
  2. eps_tra is a distance, not a magnitude-free coherence, so comparing it
     across systems is confounded by how far styling moves the embeddings at
     all. Consistency remains the reported directional measure.

The nonzero singular values of A^T B are the square roots of the nonzero
eigenvalues of G_A G_B with G = A A^T the n x n Gram matrix, so the orthogonal
fit is a function of the two Gram matrices alone -- which is what Definition 1
asserts -- and the computation is an n x n eigenproblem, not a d x d SVD.

Usage:
  python analysis/procrustes.py --nulls
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SYSTEMS = [
    ("CLIP (input)",   "clip_v2/{t}",             None),
    ("TMA, HumanML3D", "opentma_h3d_v2/{t}",      None),
    ("TMA, Motion-X",  "opentma_motionx_v2/{t}",  None),
    ("TMA, UniMoCap",  "opentma_unimocap_v2/{t}", None),
    ("MLD",            "mld_v2/seed{s}/{t}",      [42, 43, 44, 45, 46]),
    ("MDM",            "mdm_v2/seed{s}/{t}",      [42, 43, 44, 45, 46]),
    ("T2M-GPT",        "t2mgpt_v2/seed{s}/{t}",   [42, 43, 44, 45, 46]),
    ("MoMask",         "momask_v2/seed{s}/{t}",   [42, 43, 44, 45, 46]),
]
PRIMARY_TEMPLATE = 0   # adverb-final, the template Table 1 reports


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _nuclear_cross(A: np.ndarray, B: np.ndarray) -> float:
    ev = np.linalg.eigvals((A @ A.T) @ (B @ B.T)).real
    return float(np.sqrt(np.clip(ev, 0.0, None)).sum())


def residuals(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float, float]:
    """(eps_rot, eps_tra, rho_rot, rho_tra) for one unit-normalised pair."""
    n = A.shape[0]
    r_rot = max(2.0 * n - 2.0 * _nuclear_cross(A, B), 0.0)
    D = B - A
    r_tra = float(((D - D.mean(axis=0, keepdims=True)) ** 2).sum())
    M = float((D ** 2).sum())
    den = 2.0 * n
    if M <= 1e-12:
        return r_rot / den, r_tra / den, 1.0, 1.0
    return r_rot / den, r_tra / den, 1.0 - r_rot / M, 1.0 - r_tra / M


def score_run(run_dir: Path, perm: np.ndarray | None = None) -> np.ndarray:
    Z_S = np.load(run_dir / "Z_S.npy")
    Z_T = np.load(run_dir / "Z_T.npy")
    A = _unit(Z_S)
    if perm is not None:
        A = A[perm]
    return np.array([residuals(A, _unit(Z_T[j])) for j in range(Z_T.shape[0])]).mean(axis=0)


def null_run(run_dir: Path, n_perm: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = np.load(run_dir / "Z_S.npy").shape[0]
    draws = np.array([score_run(run_dir, rng.permutation(n)) for _ in range(n_perm)])
    q = lambda k: [round(float(np.percentile(draws[:, k], 2.5)), 3),
                   round(float(np.percentile(draws[:, k], 97.5)), 3)]
    return {"eps_rot_null95": q(0), "eps_tra_null95": q(1),
            "rho_rot_null95": q(2), "rho_tra_null95": q(3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", action="store_true")
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()

    out = {}
    hdr = f"{'system':16s} {'eps_rot':>8s} {'null95':>16s} {'eps_tra':>8s} {'null95':>16s}"
    print(hdr + f" {'rho_rot':>8s} {'rho_tra':>8s}")
    for label, pattern, seeds in SYSTEMS:
        dirs = ([RES / pattern.format(s=s, t=PRIMARY_TEMPLATE) for s in seeds]
                if seeds else [RES / pattern.format(t=PRIMARY_TEMPLATE)])
        dirs = [d for d in dirs if (d / "Z_S.npy").exists()]
        if not dirs:
            print(f"{label:16s} SKIP (no Z_S.npy)")
            continue
        v = np.array([score_run(d) for d in dirs])
        rec = {"eps_rot": round(float(v[:, 0].mean()), 4),
               "eps_tra": round(float(v[:, 1].mean()), 4),
               "rho_rot": round(float(v[:, 2].mean()), 4),
               "rho_tra": round(float(v[:, 3].mean()), 4),
               "n_runs": len(dirs)}
        if len(dirs) > 1:
            for i, k in enumerate(["eps_rot", "eps_tra", "rho_rot", "rho_tra"]):
                rec[k + "_sd"] = round(float(v[:, i].std(ddof=1)), 4)
        if args.nulls:
            rec.update(null_run(dirs[0], args.n_perm))
        out[label] = rec
        print(f"{label:16s} {rec['eps_rot']:8.3f} "
              f"{str(rec.get('eps_rot_null95','-')):>16s} {rec['eps_tra']:8.3f} "
              f"{str(rec.get('eps_tra_null95','-')):>16s} "
              f"{rec['rho_rot']:8.3f} {rec['rho_tra']:8.3f}")

    (RES / "procrustes.json").write_text(json.dumps(out, indent=1))
    print("\nsaved results/procrustes.json")


if __name__ == "__main__":
    main()
