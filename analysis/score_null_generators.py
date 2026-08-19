"""
Score the non-manner (modifier-presence) grid inside the generators.

The text-side control shows that swapping the manner adverb for a non-manner
one of the same syntactic slot still returns consistency 0.577 at the CLIP
input, against 0.644 for the manner modifiers: most of the input's directional
coherence is generic. That control could not say whether the generic component
also dies inside a generator, which left the comparison between the manner
excess at the input and a generator's total consistency open to challenge.

This scores the same grid through the generators, so the floor and the signal
can be read on the same axis in the same system.

Usage:
  python analysis/score_null_generators.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))
from cka import linear_cka                            # noqa: E402
from style_vectors import style_vectors, consistency  # noqa: E402

# manner-grid reference values, template 0, from results/*/seed_summary.json
MANNER = {"mld": 0.059, "t2mgpt": 0.004, "momask": 0.018, "mdm": 0.040}
LABEL = {"mld": "MLD", "t2mgpt": "T2M-GPT", "momask": "MoMask", "mdm": "MDM"}


def cosine_kernel(Z):
    Zn = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-8)
    return Zn @ Zn.T


def escape(Z_S, Z_T):
    a = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    out = []
    for j in range(Z_T.shape[0]):
        b = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        out.append(float(((b @ a.T).argmax(1) != np.arange(len(Z_S))).mean()))
    return float(np.mean(out))


def main():
    base = ROOT / "results_null"
    report = {}
    print(f"{'system':10}{'non-manner C':>14}{'manner C':>11}{'CKA':>8}{'escape':>9}{'seeds':>7}")
    print("-" * 59)
    for m in ["mld", "mdm", "t2mgpt", "momask"]:
        runs = sorted(base.glob(f"{m}/seed*/0"))
        runs = [r for r in runs if (r / "Z_S.npy").exists()]
        if not runs:
            print(f"{LABEL[m]:10}{'(not run)':>14}")
            continue
        cons, ckas, escs = [], [], []
        for r in runs:
            Z_S, Z_T = np.load(r / "Z_S.npy"), np.load(r / "Z_T.npy")
            cons.append(float(consistency(style_vectors(Z_S, Z_T))[0].mean()))
            K_S = cosine_kernel(Z_S)
            ckas.append(float(np.mean([linear_cka(K_S, cosine_kernel(z)) for z in Z_T])))
            escs.append(escape(Z_S, Z_T))
        report[m] = dict(non_manner_consistency=round(float(np.mean(cons)), 4),
                         non_manner_consistency_sd=round(float(np.std(cons)), 4),
                         manner_consistency=MANNER[m],
                         cka=round(float(np.mean(ckas)), 4),
                         escape=round(float(np.mean(escs)), 4),
                         n_seeds=len(runs))
        print(f"{LABEL[m]:10}{np.mean(cons):>14.3f}{MANNER[m]:>11.3f}"
              f"{np.mean(ckas):>8.3f}{np.mean(escs):>9.3f}{len(runs):>7}")

    print("\nInput reference: non-manner 0.577, manner 0.644 (template 0).")
    print("If the generators' non-manner values sit at their manner values,")
    print("the generic modifier-presence component dies with the manner one")
    print("and the input floor does not carry through to them.")
    (ROOT / "results" / "null_generators.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {ROOT / 'results' / 'null_generators.json'}")


if __name__ == "__main__":
    main()
