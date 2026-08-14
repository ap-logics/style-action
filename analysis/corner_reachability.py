"""
Is the high-consistency / low-CKA corner of the invariance-equivariance plane
empirically empty, or geometrically unreachable?

Figure 2a marks that corner as unoccupied and the text explains it as needing
"a shared displacement large enough to reorganise the angles it acts on".
That explanation assumes the corner is reachable in principle. This script
checks whether it is, by constructing styled configurations directly and
reading both metrics off them.

Three families are constructed from each system's real neutral embeddings:

  translation      delta(a) = mu, one shared vector, swept over magnitude.
                   Consistency is 1 by construction.

  scaled           delta(a) = c_a * mu, parallel but with action-dependent
                   magnitude, swept over the spread of c_a. Consistency is
                   STILL 1 (cosine ignores magnitude) and this is permitted by
                   Definition 2, which asks only that directions agree. This
                   is the family that could occupy the corner.

  rotation         the styled configuration is an orthogonal image of the
                   neutral one, swept over angle. This is the opposite corner
                   and serves as a positive control that CKA moves at all.

Usage:
  python analysis/corner_reachability.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"

SYSTEMS = {
    "clip_v2": "CLIP (input)",
    "opentma_h3d_v2": "TMA, HumanML3D",
    "opentma_unimocap_v2": "TMA, UniMoCap",
    "mld_v2": "MLD",
    "mdm_v2": "MDM",
    "t2mgpt_v2": "T2M-GPT",
    "momask_v2": "MoMask",
}


def cosine_kernel(Z):
    Zn = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-12)
    return Zn @ Zn.T


def linear_cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    denom = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float((Kc * Lc).sum() / (denom + 1e-12))


def consistency_of(deltas):
    d = deltas / (np.linalg.norm(deltas, axis=-1, keepdims=True) + 1e-12)
    g = d @ d.T
    iu = np.triu_indices(len(deltas), k=1)
    return float(g[iu].mean())


def load_neutral(model):
    for cand in [RESULTS / model / "0" / "Z_S.npy",
                 *sorted((RESULTS / model).glob("seed*/0/Z_S.npy"))]:
        if Path(cand).exists():
            return np.load(cand)
    return None


def random_rotation(d, angle, rng):
    """Rotation by `angle` in a random 2-plane, identity elsewhere."""
    Q = np.eye(d)
    i, j = rng.choice(d, size=2, replace=False)
    c, s = np.cos(angle), np.sin(angle)
    Q[i, i] = c; Q[i, j] = -s; Q[j, i] = s; Q[j, j] = c
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "corner_reachability.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    report = {}
    print("Observed style-vector size, per system\n")
    print(f"{'system':22}{'||d||/||z||':>13}{'||d||/||z-zbar||':>19}")
    for m, label in SYSTEMS.items():
        Z_S = load_neutral(m)
        if Z_S is None:
            continue
        run = (RESULTS / m / "0" / "Z_T.npy")
        if not run.exists():
            run = sorted((RESULTS / m).glob("seed*/0/Z_T.npy"))[0]
        D = np.load(run) - Z_S[None, :, :]
        dn = np.linalg.norm(D, axis=-1).mean()
        zn = np.linalg.norm(Z_S, axis=-1).mean()
        zc = np.linalg.norm(Z_S - Z_S.mean(0), axis=-1).mean()
        report.setdefault(m, {})["norm_ratio"] = float(dn / zn)
        report[m]["norm_ratio_centred"] = float(dn / zc)
        print(f"{label:22}{dn/zn:>13.3f}{dn/zc:>19.3f}")

    # ---------------------------------------------------------- families
    Z = load_neutral("clip_v2")
    n, d = Z.shape
    K_S = cosine_kernel(Z)
    zn = np.linalg.norm(Z, axis=-1).mean()

    print("\n\n(1) TRANSLATION: one shared displacement, swept over magnitude")
    print(f"{'||d||/||z||':>13}{'consistency':>14}{'CKA':>10}")
    rows = []
    for scale in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        mu = rng.normal(size=d); mu = mu / np.linalg.norm(mu) * scale * zn
        Zt = Z + mu[None, :]
        c, k = consistency_of(Zt - Z), linear_cka(K_S, cosine_kernel(Zt))
        rows.append(dict(scale=scale, consistency=c, cka=k))
        print(f"{scale:>13.2f}{c:>14.3f}{k:>10.4f}")
    report["translation"] = rows

    print("\n(2) SCALED: parallel displacements with action-dependent magnitude")
    print("    (consistency is still 1 -- Definition 2 permits this)")
    print(f"{'mag. spread':>13}{'consistency':>14}{'CKA':>10}")
    rows = []
    for spread in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        mu = rng.normal(size=d); mu = mu / np.linalg.norm(mu) * zn
        c_a = np.exp(rng.normal(0, spread, size=n))
        Zt = Z + c_a[:, None] * mu[None, :]
        c, k = consistency_of(Zt - Z), linear_cka(K_S, cosine_kernel(Zt))
        rows.append(dict(spread=spread, consistency=c, cka=k))
        print(f"{spread:>13.2f}{c:>14.3f}{k:>10.4f}")
    report["scaled"] = rows

    print("\n(3) ROTATION control: orthogonal image, swept over angle")
    print(f"{'angle (deg)':>13}{'consistency':>14}{'CKA':>10}")
    rows = []
    for deg in [5, 20, 45, 90]:
        Q = random_rotation(d, np.deg2rad(deg), rng)
        Zt = Z @ Q.T
        c, k = consistency_of(Zt - Z), linear_cka(K_S, cosine_kernel(Zt))
        rows.append(dict(angle_deg=deg, consistency=c, cka=k))
        print(f"{deg:>13}{c:>14.3f}{k:>10.4f}")
    report["rotation"] = rows

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
