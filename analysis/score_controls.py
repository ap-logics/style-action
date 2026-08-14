"""
Score the two control grids built by prompts/grid_controls.py.

NULL: consistency of non-manner adverbs occupying the style slot. This is the
      floor produced by appending an adverb at all, and is what the input's
      0.644 should be read against rather than zero.

PACE: consistency and basin escape for pace adverbs alongside the thirteen
      manner modifiers, extracted in the same run. Tests whether excluding
      pace removed the best candidate for a coherent direction, and measures
      the action drift that motivated the exclusion instead of assuming it.

Usage:
  python analysis/score_controls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))
from cka import linear_cka                      # noqa: E402
from style_vectors import style_vectors, consistency  # noqa: E402

PACE = ["slowly", "quickly", "briskly", "rapidly", "swiftly"]


def cosine_kernel(Z):
    Zn = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-8)
    return Zn @ Zn.T


def escape(Z_S, Z_T):
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    out = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        out.append(float(((Zt @ Zs.T).argmax(1) != np.arange(len(Z_S))).mean()))
    return np.array(out)


def load(d):
    Z_S = np.load(Path(d) / "Z_S.npy")
    Z_T = np.load(Path(d) / "Z_T.npy")
    meta = json.load(open(Path(d) / "meta.json"))
    return Z_S, Z_T, meta


def main():
    report = {}

    # ------------------------------------------------------------- NULL
    Z_S, Z_T, meta = load(ROOT / "results_controls_null/clip/0")
    cons, _ = consistency(style_vectors(Z_S, Z_T))
    esc = escape(Z_S, Z_T)
    K_S = cosine_kernel(Z_S)
    cka = np.array([linear_cka(K_S, cosine_kernel(z)) for z in Z_T])
    print("NULL CONTROL — non-manner adverbs in the style slot (template 0)\n")
    print(f"{'adverb':14}{'consistency':>13}{'CKA':>8}{'escape':>9}")
    for a, c, k, e in sorted(zip(meta["styles"], cons, cka, esc),
                             key=lambda x: -x[1]):
        print(f"{a:14}{c:>13.3f}{k:>8.3f}{e:>9.3f}")
    print(f"{'MEAN':14}{cons.mean():>13.3f}{cka.mean():>8.3f}{esc.mean():>9.3f}")
    report["null"] = dict(mean_consistency=float(cons.mean()),
                          mean_cka=float(cka.mean()),
                          mean_escape=float(esc.mean()),
                          per_adverb=dict(zip(meta["styles"],
                                              [float(x) for x in cons])))

    # main-grid CLIP reference, same template
    Z_S0, Z_T0, meta0 = load(ROOT / "results/clip_v2/0")
    cons0, _ = consistency(style_vectors(Z_S0, Z_T0))
    print(f"\n  manner modifiers, same template : {cons0.mean():.3f}")
    print(f"  non-manner floor                : {cons.mean():.3f}")
    print(f"  manner above the floor          : {cons0.mean() - cons.mean():+.3f}")
    report["null"]["manner_consistency"] = float(cons0.mean())
    report["null"]["excess_over_floor"] = float(cons0.mean() - cons.mean())

    # ------------------------------------------------------------- PACE
    print("\n\nPACE CONTROL — pace adverbs run alongside the manner modifiers")
    rows, per_t = {}, []
    for t in (0, 1, 2):
        Z_S, Z_T, meta = load(ROOT / f"results_controls_pace/clip/{t}")
        cons, _ = consistency(style_vectors(Z_S, Z_T))
        esc = escape(Z_S, Z_T)
        K_S = cosine_kernel(Z_S)
        cka = np.array([linear_cka(K_S, cosine_kernel(z)) for z in Z_T])
        per_t.append((meta["styles"], cons, cka, esc))
        for s, c, k, e in zip(meta["styles"], cons, cka, esc):
            rows.setdefault(s, []).append((c, k, e))

    styles = per_t[0][0]
    ispace = [s in PACE for s in styles]
    C = np.array([[np.mean([v[0] for v in rows[s]])] for s in styles]).ravel()
    K = np.array([[np.mean([v[1] for v in rows[s]])] for s in styles]).ravel()
    E = np.array([[np.mean([v[2] for v in rows[s]])] for s in styles]).ravel()

    print(f"\n{'modifier':14}{'kind':>8}{'consistency':>13}{'CKA':>8}{'escape':>9}")
    for s, c, k, e, p in sorted(zip(styles, C, K, E, ispace), key=lambda x: -x[1]):
        print(f"{s:14}{'PACE' if p else 'manner':>8}{c:>13.3f}{k:>8.3f}{e:>9.3f}")

    m, p = ~np.array(ispace), np.array(ispace)
    print(f"\n{'manner (13)':14}{'':>8}{C[m].mean():>13.3f}{K[m].mean():>8.3f}{E[m].mean():>9.3f}")
    print(f"{'pace (5)':14}{'':>8}{C[p].mean():>13.3f}{K[p].mean():>8.3f}{E[p].mean():>9.3f}")
    report["pace"] = dict(
        manner=dict(consistency=float(C[m].mean()), cka=float(K[m].mean()),
                    escape=float(E[m].mean())),
        pace=dict(consistency=float(C[p].mean()), cka=float(K[p].mean()),
                  escape=float(E[p].mean())),
        per_modifier={s: dict(consistency=float(c), cka=float(k), escape=float(e))
                      for s, c, k, e in zip(styles, C, K, E)})

    out = ROOT / "results/controls.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
