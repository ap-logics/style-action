from __future__ import annotations
import numpy as np


def _adjacency(K: np.ndarray, tau: float) -> np.ndarray:
    adj = (K >= tau).astype(np.uint8)
    np.fill_diagonal(adj, 0)   # no self-loops
    return adj


def graph_edit_distance(K_S: np.ndarray, K_T: np.ndarray, tau: float) -> float:
    """
    Normalised GED from Chowdhury & Land (2026) eq. (3).
    = 2 * |E(G_S) △ E(G_T)| / (n*(n-1))

    Since G_S and G_T have the same node set, GED reduces to the normalised
    Hamming distance between their upper-triangular adjacency matrices.
    Returns a value in [0, 1]; higher = more topological mismatch.
    """
    n = K_S.shape[0]
    A_S = _adjacency(K_S, tau)
    A_T = _adjacency(K_T, tau)

    # count differing edges in upper triangle only (undirected)
    upper = np.triu_indices(n, k=1)
    diff = int(np.sum(A_S[upper] != A_T[upper]))
    max_edges = n * (n - 1) // 2
    return 2 * diff / (n * (n - 1))   # equiv to diff / max_edges * 2 / ... see paper


def null_ged(
    K_S: np.ndarray,
    K_T: np.ndarray,
    tau: float,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Observed GED + empirical p-value via row/col permutation of K_S.
    Returns (observed, p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    observed = graph_edit_distance(K_S, K_T, tau)
    idx = np.arange(len(K_S))
    null = np.array([
        graph_edit_distance(K_S[np.ix_(perm := rng.permutation(idx), perm)], K_T, tau)
        for _ in range(n_permutations)
    ])
    p = float(np.mean(null >= observed))
    return observed, p, null
