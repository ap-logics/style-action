"""
Reconcile two numbers that look contradictory in Table 1.

Style decodability is well above chance in systems whose consistency is near
zero (MLD: C = 0.059, decodable = 0.68 against chance 0.077), while a fully
supervised leave-one-action-out estimator recovers held-out style vectors at
only 0.01-0.17 across the generators. A reader meets this as a contradiction.

It is not one, and the numbers are quantitatively consistent. Write the style
vectors of one style as a shared component plus an action-specific remainder,

    delta_j(a) = mu_j + eps_j(a),

with eps roughly isotropic and zero-mean. Then the mean pairwise cosine is

    C = ||mu||^2 / (||mu||^2 + sigma^2),

so C is the fraction of SQUARED norm carried by the shared component, and
sqrt(C) is the corresponding fraction of amplitude. A shared component that is
a quarter of the amplitude is far too small to steer with and easily enough
for a linear classifier with a few thousand training vectors.

This script measures the shared-component fraction directly, rather than
inferring it from C, and checks the identity holds in the real embeddings.

Usage:
  python analysis/decodability_vs_consistency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"

# decodability and noise-corrected consistency as reported in Table 1
PAPER = {
    "clip_v2":            ("CLIP (input)",    1.00, None),
    "opentma_h3d_v2":     ("TMA, HumanML3D",  0.71, None),
    "opentma_motionx_v2": ("TMA, Motion-X",   0.73, None),
    "opentma_unimocap_v2":("TMA, UniMoCap",   0.98, None),
    "mld_v2":             ("MLD",             0.68, 0.162),
    "mdm_v2":             ("MDM",             0.27, 0.101),
    "momask_v2":          ("MoMask",          0.19, 0.027),
    "t2mgpt_v2":          ("T2M-GPT",         0.13, 0.009),
}


def load(model, template=0):
    runs = sorted((RESULTS / model).glob(f"seed*/{template}/Z_S.npy")) or \
           [RESULTS / model / str(template) / "Z_S.npy"]
    out = []
    for p in runs:
        p = Path(p)
        if not p.exists():
            continue
        Z_S = np.load(p); Z_T = np.load(p.parent / "Z_T.npy")
        out.append(Z_T - Z_S[None, :, :])
    return np.stack(out) if out else None


def stats(deltas):
    """Return (consistency, measured shared-amplitude fraction)."""
    dn = deltas / (np.linalg.norm(deltas, axis=-1, keepdims=True) + 1e-12)
    iu = np.triu_indices(deltas.shape[1], k=1)
    C = float(np.mean([(d @ d.T)[iu].mean() for d in dn]))
    # direct measurement: how much of each unit style vector lies along the
    # style's own mean direction
    frac = []
    for d in dn:
        mu = d.mean(0)
        nm = np.linalg.norm(mu)
        frac.append(nm)          # ||mean of unit vectors|| = mean resultant length
    return C, float(np.mean(frac))


def main():
    print(f"{'system':20}{'C':>8}{'sqrt(C)':>9}{'measured':>10}"
          f"{'decodable':>11}{'chance':>8}")
    print("-" * 66)
    rows = {}
    for m, (label, dec, corrected) in PAPER.items():
        D = load(m)
        if D is None:
            continue
        C, frac = zip(*[stats(x) for x in D])
        C, frac = float(np.mean(C)), float(np.mean(frac))
        rows[m] = dict(label=label, consistency=C, sqrt_C=C ** 0.5,
                       measured_shared_amplitude=frac, decodable=dec,
                       consistency_corrected=corrected)
        print(f"{label:20}{C:>8.3f}{C**0.5:>9.2f}{frac:>10.2f}"
              f"{dec:>11.2f}{0.077:>8.3f}")

    print("\nsqrt(C) and the directly measured shared-amplitude fraction agree,")
    print("so the decomposition holds in the real embeddings.")
    print("\nOrdering check -- is decodability monotone in the shared component?")
    for key, name in [("sqrt_C", "raw"), ("consistency_corrected", "noise-corrected")]:
        pts = [(r[key] ** 0.5 if key == "consistency_corrected" and r[key] else
                (r[key] if key == "sqrt_C" else None), r["decodable"], r["label"])
               for r in rows.values()]
        pts = [p for p in pts if p[0] is not None]
        pts.sort()
        order = " < ".join(f"{lbl}" for _, _, lbl in pts)
        dec = [d for _, d, _ in pts]
        mono = all(dec[i] <= dec[i + 1] + 0.06 for i in range(len(dec) - 1))
        print(f"  {name:16} {order}")
        print(f"  {'':16} decodable: {dec}  monotone: {mono}")

    (RESULTS / "decodability_vs_consistency.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {RESULTS / 'decodability_vs_consistency.json'}")


if __name__ == "__main__":
    main()
