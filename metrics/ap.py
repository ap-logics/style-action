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
    Z_S: np.ndarray,   # (n_actions, d) neutral embeddings, one per action
    Z_T: np.ndarray,   # (n_styles, n_actions, d) styled embeddings
) -> tuple[float, np.ndarray]:
    """
    BER: fraction of styled embeddings whose cosine-nearest NEUTRAL embedding
    belongs to a different action (the paper's definition; matches the inline
    implementations in pipeline.py / score_hpc.py / score_seeds.py).

    Returns (BER, boolean escape mask of shape (n_styles, n_actions)).
    """
    n_styles, n_actions, _ = Z_T.shape
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    escaped = np.zeros((n_styles, n_actions), dtype=bool)
    for j in range(n_styles):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        escaped[j] = (Zt @ Zs.T).argmax(axis=1) != np.arange(n_actions)
    return float(escaped.mean()), escaped
