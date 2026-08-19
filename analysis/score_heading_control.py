"""
Score the heading control against manner on a matched action set.

The heading grid (prompts/grid_positive.py) is emitted over twelve locomotion
actions, only eight of which come from the primary 24-action grid. Comparing
heading on twelve actions against manner on twenty-four is not like for like:
it mixes an action-set change into an attribute change, and it inflates the
ratio between them. Everything here is therefore restricted to the eight
shared actions, with manner rescored on those same eight.

Consistency is a mean over pairs, so the reference band depends on the number
of actions. Eight actions give 28 pairs against the primary grid's 276, and the
independent-directions band widens accordingly; it is simulated here rather
than scaled analytically.

Usage:
  python analysis/score_heading_control.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prompts"))
sys.path.insert(0, str(ROOT / "metrics"))
from grid_v2 import ACTIONS_V2                        # noqa: E402
from grid_positive import ACTIONS_DIR                 # noqa: E402
from style_vectors import style_vectors, consistency  # noqa: E402

SYSTEMS = [("MLD", "mld", "mld_v2"), ("MoMask", "momask", "momask_v2"),
           ("T2M-GPT", "t2mgpt", "t2mgpt_v2")]


def cons(Z_S, Z_T, idx):
    return float(consistency(style_vectors(Z_S[idx], Z_T[:, idx]))[0].mean())


def load(p: Path):
    return np.load(p / "Z_S.npy"), np.load(p / "Z_T.npy")


def band(n_vec, d, rng, trials=20000):
    iu = np.triu_indices(n_vec, 1)
    out = np.empty(trials)
    for t in range(trials):
        V = rng.normal(size=(n_vec, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        out[t] = (V @ V.T)[iu].mean()
    return np.percentile(out, [2.5, 97.5])


def main():
    grid = [a for a, _ in ACTIONS_V2]
    shared = [a for a in ACTIONS_DIR if a in grid]
    i_head = [ACTIONS_DIR.index(a) for a in shared]
    i_man = [grid.index(a) for a in shared]
    print(f"{len(shared)} shared actions: {', '.join(shared)}\n")

    rows = {}
    hS, hT = load(ROOT / "results_positive/clip/0")
    mS, mT = load(ROOT / "results/clip_v2/0")
    rows["CLIP"] = (cons(hS, hT, i_head), cons(mS, mT, i_man))
    for label, hdir, mdir in SYSTEMS:
        hr = [r for r in sorted((ROOT / "results_positive" / hdir).glob("seed*/0"))
              if (r / "Z_S.npy").exists()]
        mr = [r for r in sorted((ROOT / "results" / mdir).glob("seed*/0"))
              if (r / "Z_S.npy").exists()]
        rows[label] = (float(np.mean([cons(*load(r), i_head) for r in hr])),
                       float(np.mean([cons(*load(r), i_man) for r in mr])))

    print(f"{'system':10}{'heading':>10}{'manner':>9}{'ratio':>8}")
    print("-" * 37)
    for k, (h, m) in rows.items():
        print(f"{k:10}{h:>10.3f}{m:>9.3f}{(h / m if m > 1e-6 else float('nan')):>8.1f}x")

    rng = np.random.default_rng(0)
    print(f"\nindependent-directions band at n = {len(shared)}:")
    bands = {}
    for d in (256, 512):
        lo, hi = band(len(shared), d, rng)
        bands[d] = [float(lo), float(hi)]
        print(f"   d = {d}: [{lo:+.3f}, {hi:+.3f}]")
    print("\nMLD is d=256; MoMask and T2M-GPT are d=512.")

    (ROOT / "results/heading_control_matched.json").write_text(json.dumps(
        {"shared_actions": shared,
         "consistency": {k: {"heading": h, "manner": m} for k, (h, m) in rows.items()},
         "null_band": bands}, indent=2))
    print(f"\nwritten to {ROOT / 'results/heading_control_matched.json'}")


if __name__ == "__main__":
    main()
