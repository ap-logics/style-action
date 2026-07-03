"""
Leave-one-action-out linear style recovery.

The strongest version of "style is linearly usable": estimate the style
direction for modifier j from SEVEN actions with full supervision,
    delta_hat_j = mean_{a != b} [z_j(a) - z_0(a)],
then apply it to the held-out action b and ask two questions:

  direction recovery  cos( delta_hat_j , z_j(b) - z_0(b) )
      how well does the supervised estimate predict the true shift?
  steering success    is z_0(b) + delta_hat_j nearer to z_j(b) than to
      the styled embedding of ANY other action? (1-NN among styled points)

If even this supervised estimator fails on held-out actions, no zero-shot
latent arithmetic can succeed, and the negative result is causal rather
than correlational. Where it partially works, this doubles as a free
post-hoc linear steering method with a measured success rate.

Usage:  python linear_probe.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SYSTEMS = ["clip", "opentma_h3d", "opentma_motionx", "opentma_unimocap",
           "mdm", "t2mgpt"]


def load(system: str):
    base = ROOT / "results" / system
    if (base / "0").exists():
        base = base / "0"
    Z_S = np.load(base / "Z_S.npy")
    Z_T = np.load(base / "Z_T.npy")
    meta = json.loads((base / "meta.json").read_text())
    return Z_S, Z_T, meta["styles"]


def probe(Z_S: np.ndarray, Z_T: np.ndarray):
    n_styles, n_actions, d = Z_T.shape
    deltas = Z_T - Z_S[None]
    dir_rec = np.zeros((n_styles, n_actions))
    steer_ok = np.zeros((n_styles, n_actions), dtype=bool)

    for j in range(n_styles):
        for b in range(n_actions):
            train = [a for a in range(n_actions) if a != b]
            d_hat = deltas[j, train].mean(axis=0)

            true = deltas[j, b]
            dir_rec[j, b] = (d_hat @ true) / (
                np.linalg.norm(d_hat) * np.linalg.norm(true) + 1e-8)

            z_pred = Z_S[b] + d_hat
            # nearest styled embedding (this style, all actions)
            sims = (Z_T[j] @ z_pred) / (
                np.linalg.norm(Z_T[j], axis=1) * np.linalg.norm(z_pred) + 1e-8)
            steer_ok[j, b] = sims.argmax() == b

    return dir_rec, steer_ok


def main():
    print(f"{'system':<20} {'direction recovery':>20} {'steering success':>18}")
    out = {}
    for s in SYSTEMS:
        Z_S, Z_T, styles = load(s)
        dir_rec, steer_ok = probe(Z_S, Z_T)
        out[s] = {
            "direction_recovery_mean": round(float(dir_rec.mean()), 4),
            "steering_success_rate": round(float(steer_ok.mean()), 4),
            "per_style_recovery": {st: round(float(dir_rec[j].mean()), 4)
                                   for j, st in enumerate(styles)},
        }
        print(f"{s:<20} {dir_rec.mean():>20.3f} {steer_ok.mean():>18.3f}")
    (ROOT / "results" / "linear_probe.json").write_text(
        json.dumps(out, indent=1))
    print("\nsaved results/linear_probe.json")


if __name__ == "__main__":
    main()
