"""
Per-style association between the two axes, within each system.

For each system we compute, for each of the thirteen modifiers, that modifier's
CKA against the neutral kernel and its style-vector consistency, then correlate
the two across modifiers. This is the "do the two axes move together within a
system?" check quoted in Section 4.3.

Note on what this replaced: the manuscript previously quoted a single r = 0.001
as a within-system result. That figure is the CLIP row -- the input, not one of
the probed systems. Computed per system the picture is different: six of the
eight correlations are negative, none is individually determined at n = 13, and
the defensible claim is that the two axes are nowhere positively coupled rather
than that they are independent.

Usage:
  python analysis/per_style_corr.py
Writes results/per_style_correlation.json.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "metrics"))
from cka import linear_cka                       # noqa: E402
from style_vectors import style_vectors, consistency  # noqa: E402

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
PRIMARY_TEMPLATE = 0


def _kernel(Z: np.ndarray) -> np.ndarray:
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Zn @ Zn.T


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> list[float]:
    if abs(r) >= 1.0 or n < 4:
        return [float("nan"), float("nan")]
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    crit = 1.959963985
    return [round((math.exp(2 * x) - 1) / (math.exp(2 * x) + 1), 3)
            for x in (z - crit * se, z + crit * se)]


def score_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    Z_S = np.load(run_dir / "Z_S.npy")
    Z_T = np.load(run_dir / "Z_T.npy")
    K_S = _kernel(Z_S)
    cka = np.array([linear_cka(K_S, _kernel(Z_T[j])) for j in range(Z_T.shape[0])])
    cons, _ = consistency(style_vectors(Z_S, Z_T))
    return cka, cons


def main() -> None:
    out = {}
    print(f"{'system':16s} {'r':>7s} {'95% CI':>18s} {'n':>4s}")
    for label, pattern, seeds in SYSTEMS:
        dirs = ([RES / pattern.format(s=s, t=PRIMARY_TEMPLATE) for s in seeds]
                if seeds else [RES / pattern.format(t=PRIMARY_TEMPLATE)])
        dirs = [d for d in dirs if (d / "Z_S.npy").exists()]
        if not dirs:
            print(f"{label:16s} SKIP")
            continue
        rs = []
        for d in dirs:
            cka, cons = score_run(d)
            rs.append(float(np.corrcoef(cka, cons)[0, 1]))
        r = float(np.mean(rs))
        n = 13
        rec = {"r": round(r, 3), "ci95_fisher": fisher_ci(r, n),
               "n_modifiers": n, "n_runs": len(dirs)}
        if len(rs) > 1:
            rec["r_sd_across_seeds"] = round(float(np.std(rs, ddof=1)), 3)
        out[label] = rec
        print(f"{label:16s} {rec['r']:+7.3f} {str(rec['ci95_fisher']):>18s} {n:4d}")

    rr = [v["r"] for v in out.values()]
    out["_summary"] = {
        "range": [round(min(rr), 3), round(max(rr), 3)],
        "n_negative": int(sum(1 for x in rr if x < 0)),
        "n_systems": len(rr),
        "note": ("no correlation is individually determined at n=13; the "
                 "defensible claim is that the two axes are nowhere positively "
                 "coupled, not that they are independent"),
    }
    (RES / "per_style_correlation.json").write_text(json.dumps(out, indent=1))
    print(f"\nrange {out['_summary']['range']}, "
          f"{out['_summary']['n_negative']} of {out['_summary']['n_systems']} negative")
    print("saved results/per_style_correlation.json")


if __name__ == "__main__":
    main()
