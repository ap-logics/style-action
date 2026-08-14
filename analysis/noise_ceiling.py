"""
Empirical noise ceiling for style-vector consistency.

The concern this answers: generator consistency C is measured on stochastic
latents (cross-seed reliability r = 0.36-0.68) and compared against CLIP's
0.644, which is deterministic (r = 1). The disattenuated C/r is offered as a
sensitivity analysis, but classical attenuation correction assumes additive,
isotropic, signal-independent measurement error, and diffusion decoding noise
plausibly violates all three.

So instead of correcting the generators upward, we degrade CLIP downward:
take a system KNOWN to carry a partially coherent style direction, inject
noise calibrated to each generator's measured r, and read C back off. That
gives the value CLIP would post if it were measured through that generator's
noise -- an empirical ceiling that the generator must be compared against
instead of against zero.

Two noise models are run:

  gaussian   isotropic additive noise. This is the classical model, and its
             prediction is C_obs ~= r * C_true. Running it tests whether that
             formula actually holds in CLIP's (strongly anisotropic) geometry.

  empirical  residuals resampled from a real generator's seed-to-seed spread,
             which carries whatever anisotropy, heteroscedasticity and
             signal-dependence the real decoding noise has. This is the model
             that does NOT assume the classical conditions. Only available for
             generators whose latent dimension matches CLIP's 512.

In both cases the noise scale is calibrated by bisection so that the achieved
cross-replicate reliability matches the target r, and C is then computed per
replicate and averaged, exactly as the generator numbers are.

Usage:
  python analysis/noise_ceiling.py
  python analysis/noise_ceiling.py --template 0 --replicates 5 --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"

GENERATORS = {
    "mdm_v2": "MDM, pose diffusion",
    "mld_v2": "MLD, latent diffusion",
    "t2mgpt_v2": "T2M-GPT, VQ tokens",
    "momask_v2": "MoMask, residual VQ",
}


# --------------------------------------------------------------- primitives
def consistency(deltas: np.ndarray) -> float:
    """Mean pairwise cosine among the style vectors of each style, then over styles.

    deltas: (n_styles, n_actions, d)
    """
    normed = deltas / (np.linalg.norm(deltas, axis=2, keepdims=True) + 1e-12)
    grams = np.einsum("sad,sbd->sab", normed, normed)
    iu = np.triu_indices(deltas.shape[1], k=1)
    return float(np.mean([g[iu].mean() for g in grams]))


def reliability(reps: np.ndarray) -> float:
    """Cross-replicate reliability of the style vectors.

    reps: (n_reps, n_styles, n_actions, d). For each (style, action) take the
    mean pairwise cosine of that vector across replicates, then average. This
    is test-retest reliability of the measurement, which is the quantity the
    attenuation correction divides by.
    """
    normed = reps / (np.linalg.norm(reps, axis=3, keepdims=True) + 1e-12)
    n_reps = reps.shape[0]
    iu = np.triu_indices(n_reps, k=1)
    # (n_reps, n_reps) cosine per (style, action), upper triangle averaged
    gram = np.einsum("rsad,qsad->saqr", normed, normed)
    return float(np.mean([gram[s, a][iu].mean()
                          for s in range(reps.shape[1])
                          for a in range(reps.shape[2])]))


def load_deltas(model: str, template: int) -> np.ndarray | None:
    """Return (n_reps, n_styles, n_actions, d) style vectors, one rep per seed."""
    seed_dirs = sorted((RESULTS / model).glob("seed*"))
    if seed_dirs:
        runs = [d / str(template) for d in seed_dirs]
    else:
        runs = [RESULTS / model / str(template)]
    out = []
    for r in runs:
        zs, zt = r / "Z_S.npy", r / "Z_T.npy"
        if not zs.exists():
            continue
        Z_S, Z_T = np.load(zs), np.load(zt)
        out.append(Z_T - Z_S[None, :, :])
    return np.stack(out) if out else None


# ------------------------------------------------------------ noise models
def add_noise(delta: np.ndarray, scale: float, rng, pool: np.ndarray | None) -> np.ndarray:
    """One noisy replicate of a clean style-vector field.

    pool=None gives isotropic Gaussian noise. Otherwise residuals are drawn
    with replacement from `pool`, preserving the real noise geometry.
    """
    if pool is None:
        eps = rng.normal(size=delta.shape)
    else:
        idx = rng.integers(0, len(pool), size=delta.shape[:2])
        eps = pool[idx]
    # scale is expressed relative to the typical style-vector norm, so the
    # calibration is dimensionless and comparable across systems
    unit = np.linalg.norm(delta, axis=2, keepdims=True).mean()
    eps = eps / (np.linalg.norm(eps, axis=2, keepdims=True).mean() + 1e-12) * unit
    return delta + scale * eps


def calibrate(delta, target_r, rng, pool, n_reps, tol=2e-3, iters=40):
    """Bisect the noise scale until achieved reliability matches target_r."""
    lo, hi = 0.0, 8.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        reps = np.stack([add_noise(delta, mid, rng, pool) for _ in range(n_reps)])
        r = reliability(reps)
        if abs(r - target_r) < tol:
            return mid, r
        if r > target_r:      # too little noise
            lo = mid
        else:
            hi = mid
    return mid, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=5,
                    help="noisy replicates per trial; matches the 5 generator seeds")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "noise_ceiling.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    clip = load_deltas("clip_v2", args.template)
    assert clip is not None, "clip_v2 embeddings not found"
    clip = clip[0]
    c_clip = consistency(clip)
    print(f"CLIP (input): consistency {c_clip:.3f}, deterministic (r = 1)\n")

    # measured reliability and raw consistency for each generator
    measured = {}
    for m, label in GENERATORS.items():
        reps = load_deltas(m, args.template)
        if reps is None or len(reps) < 2:
            print(f"  {label}: insufficient seeds, skipped")
            continue
        r = reliability(reps)
        c = float(np.mean([consistency(x) for x in reps]))
        resid = (reps - reps.mean(0, keepdims=True)).reshape(-1, reps.shape[-1])
        measured[m] = dict(label=label, r=r, c_raw=c, d=reps.shape[-1],
                           residuals=resid)
        print(f"  {label:24} r = {r:.3f}   raw C = {c:.3f}   d = {reps.shape[-1]}")

    print(f"\n{'':26}{'CLIP under that noise':>34}")
    print(f"{'system':26}{'r':>7}{'gaussian':>13}{'empirical':>12}"
          f"{'raw C':>9}{'ratio':>8}")
    print("-" * 76)

    report = {"clip_consistency": c_clip, "template": args.template, "systems": {}}
    for m, info in measured.items():
        row = {"r": info["r"], "c_raw": info["c_raw"], "d": info["d"]}

        # (a) classical isotropic Gaussian
        vals = []
        for _ in range(max(1, args.trials // 20)):
            s, _ = calibrate(clip, info["r"], rng, None, args.replicates)
            reps = np.stack([add_noise(clip, s, rng, None)
                             for _ in range(args.replicates)])
            vals.append(np.mean([consistency(x) for x in reps]))
        row["ceiling_gaussian"] = float(np.mean(vals))
        row["ceiling_gaussian_sd"] = float(np.std(vals))

        # (b) empirical residuals, only where the dimension matches CLIP
        if info["d"] == clip.shape[-1]:
            vals = []
            for _ in range(max(1, args.trials // 20)):
                s, _ = calibrate(clip, info["r"], rng, info["residuals"],
                                 args.replicates)
                reps = np.stack([add_noise(clip, s, rng, info["residuals"])
                                 for _ in range(args.replicates)])
                vals.append(np.mean([consistency(x) for x in reps]))
            row["ceiling_empirical"] = float(np.mean(vals))
            row["ceiling_empirical_sd"] = float(np.std(vals))
            emp = f"{row['ceiling_empirical']:.3f}"
        else:
            row["ceiling_empirical"] = None
            emp = "  n/a"

        ceiling = row.get("ceiling_empirical") or row["ceiling_gaussian"]
        row["ratio_ceiling_over_raw"] = float(ceiling / max(info["c_raw"], 1e-9))
        report["systems"][m] = row

        print(f"{info['label']:26}{info['r']:>7.2f}"
              f"{row['ceiling_gaussian']:>13.3f}{emp:>12}"
              f"{info['c_raw']:>9.3f}{row['ratio_ceiling_over_raw']:>8.1f}x")

    print("\nReading: 'CLIP under that noise' is what a genuinely partially")
    print("coherent style direction reads as, once measured through the same")
    print("amount of noise as that generator. Compare the generator's raw C")
    print("against that number, not against zero.")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
