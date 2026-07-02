from __future__ import annotations
import numpy as np


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H)) / (n - 1) ** 2


def linear_cka(K: np.ndarray, L: np.ndarray) -> float:
    """
    Centered Kernel Alignment between two (N, N) cosine-similarity matrices.
    Returns a value in [0, 1]; higher = more geometrically aligned.
    """
    return _hsic(K, L) / np.sqrt(_hsic(K, K) * _hsic(L, L) + 1e-10)


def null_cka(
    K_S: np.ndarray,
    K_T: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Observed CKA + empirical p-value via row/col permutation of K_S.
    Returns (observed, p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    observed = linear_cka(K_S, K_T)
    idx = np.arange(len(K_S))
    null = np.array([
        linear_cka(K_S[np.ix_(perm := rng.permutation(idx), perm)], K_T)
        for _ in range(n_permutations)
    ])
    p = float(np.mean(null >= observed))
    return observed, p, null
