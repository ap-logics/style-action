from __future__ import annotations
import numpy as np
from ged import _adjacency

# Quantiles of the off-diagonal K_S values used as tau candidates.
# A fixed grid (Chowdhury & Land use tau=0.4 for word embeddings) does not
# transfer across representation spaces: CLIP text embeddings, for example,
# are anisotropic — all pairwise cosines land in roughly [0.83, 0.94] — so
# any fixed tau below that band yields a complete graph and degenerate GED.
TAU_QUANTILES = [0.2, 0.35, 0.5, 0.65, 0.8]


def tau_selection(
    K_S: np.ndarray,
    quantiles: list[float] = TAU_QUANTILES,
) -> tuple[float, dict[float, float]]:
    """
    Select tau by maximising the discriminativeness of the thresholded
    neutral graph G_S. Candidate taus are quantiles of the off-diagonal
    similarity values, so the sweep adapts to the similarity range of the
    representation being probed. Each candidate is scored by d*(1-d) where
    d is the resulting edge density: 0 at the degenerate extremes
    (empty/complete graph), maximal near d = 0.5.

    Returns (best_tau, {tau: score} for all candidates).
    """
    n = K_S.shape[0]
    upper = np.triu_indices(n, k=1)
    off_diag = K_S[upper]

    scores: dict[float, float] = {}
    for q in quantiles:
        tau = round(float(np.quantile(off_diag, q)), 4)
        adj = _adjacency(K_S, tau)
        d = float(adj[upper].mean())
        scores[tau] = round(d * (1.0 - d), 4)
    best_tau = max(scores, key=scores.__getitem__)
    return best_tau, scores


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--K_S", required=True, help="Path to K_S .npy file")
    args = p.parse_args()

    K_S = np.load(args.K_S)
    best, scores = tau_selection(K_S)
    print(f"Best tau: {best}")
    print("Score curve (d*(1-d), higher = more discriminative):")
    for tau, s in sorted(scores.items()):
        marker = " <--" if tau == best else ""
        print(f"  tau={tau:.4f}  score={s:.4f}{marker}")
