from __future__ import annotations
import numpy as np


def action_preservation(
    motions: "np.ndarray",   # (N, T, J*3)
    labels: list[int],
    classifier,              # humanml3d.HumanML3DClassifier
) -> tuple[float, np.ndarray]:
    """
    Returns (AP score, per-sample correctness array).
    AP = fraction of generated motions whose predicted action label
    matches the ground-truth action label.
    """
    preds = classifier.predict(motions)   # (N,) int array
    labels_arr = np.array(labels)
    correct = preds == labels_arr
    return float(correct.mean()), correct


def basin_escape_rate(
    Z_S: np.ndarray,   # (N, d) neutral embeddings
    Z_T: np.ndarray,   # (N, d) styled embeddings
    n_actions: int = 8,
    n_styles: int = 8,
) -> tuple[float, np.ndarray]:
    """
    BER: fraction of (action, style) pairs where the styled embedding
    is closer to a different action centroid than the neutral embedding.

    Centroids are computed from Z_S (neutral space).
    Returns (BER, boolean mask of escaping pairs).
    """
    from sklearn.decomposition import PCA

    # action centroids in neutral space
    centroids = np.stack([
        Z_S[ai * n_styles:(ai + 1) * n_styles].mean(axis=0)
        for ai in range(n_actions)
    ])  # (n_actions, d)

    def nearest(Z: np.ndarray) -> np.ndarray:
        # cosine nearest centroid
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        C_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        return np.argmax(Z_norm @ C_norm.T, axis=1)   # (N,)

    true_labels = np.array([ai for ai in range(n_actions) for _ in range(n_styles)])
    neutral_nearest = nearest(Z_S)
    styled_nearest  = nearest(Z_T)

    # escape = styled point lands in a different basin than the neutral point
    escaped = styled_nearest != true_labels
    # skip neutral-baseline style (style index 0, every n_styles-th row)
    neutral_mask = np.array([si == 0 for _ in range(n_actions) for si in range(n_styles)])
    escaped[neutral_mask] = False

    ber = float(escaped[~neutral_mask].mean())
    return ber, escaped
