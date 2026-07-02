"""
Style-vector geometry: tests the linear separability hypothesis directly.

For style j and action a, the style vector is
    delta_j(a) = z_j(a) - z_0(a).

If the representation encodes style as an axis independent of action,
delta_j(a) should point in (roughly) the same direction for every action.
We measure this as the mean pairwise cosine similarity of the 8 style
vectors for each style ("consistency"). Consistency near 1 means style is
a global direction; near 0 means the style shift is action-specific,
which is coupling stated geometrically.

The causal counterpart (transfer: decode z_0(b) + delta_j(a) and check the
action survives while the style appears) needs a decoder and runs on HPC.
"""
from __future__ import annotations
import numpy as np


def style_vectors(Z_S: np.ndarray, Z_T: np.ndarray) -> np.ndarray:
    """
    Z_S: (n_actions, d) neutral embeddings
    Z_T: (n_styles, n_actions, d) styled embeddings
    Returns deltas: (n_styles, n_actions, d)
    """
    return Z_T - Z_S[None, :, :]


def consistency(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-style consistency: mean pairwise cosine among that style's
    action-wise delta vectors.

    Returns:
        per_style : (n_styles,) mean pairwise cosine per style
        matrices  : (n_styles, n_actions, n_actions) full pairwise cosines
    """
    n_styles, n_actions, _ = deltas.shape
    normed = deltas / (np.linalg.norm(deltas, axis=2, keepdims=True) + 1e-8)
    matrices = np.einsum("sad,sbd->sab", normed, normed)
    upper = np.triu_indices(n_actions, k=1)
    per_style = np.array([m[upper].mean() for m in matrices])
    return per_style, matrices


def null_consistency(
    Z_S: np.ndarray,
    Z_T: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Null: shuffle which neutral embedding is subtracted from each styled
    embedding (breaking the action pairing), recompute consistency.
    Returns (observed_per_style, p_values_per_style).
    """
    rng = np.random.default_rng(seed)
    observed, _ = consistency(style_vectors(Z_S, Z_T))
    n_styles, n_actions, _ = Z_T.shape
    null = np.zeros((n_permutations, n_styles))
    for i in range(n_permutations):
        perm = rng.permutation(n_actions)
        null[i], _ = consistency(Z_T - Z_S[None, perm, :])
    p = (null >= observed[None, :]).mean(axis=0)
    return observed, p


def transfer_targets(Z_S: np.ndarray, deltas: np.ndarray,
                     source_action: int) -> np.ndarray:
    """
    Latent arithmetic for the causal test: apply the style vector measured
    on `source_action` to every other action's neutral embedding.

    Returns edited: (n_styles, n_actions, d) where
        edited[j, b] = Z_S[b] + deltas[j, source_action]
    Decode these and check (1) action classifier still outputs b,
    (2) style scorer detects style j.
    """
    return Z_S[None, :, :] + deltas[:, source_action, None, :]
