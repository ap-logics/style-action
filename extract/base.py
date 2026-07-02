from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LatentExtractor(ABC):
    """
    Interface that both MDM and T2M-GPT extractors must satisfy.

    encode : text prompts → latent vectors  (used to build K_S / K_T)
    decode : latent vectors → motion seqs   (used for AP generation)
    """

    @abstractmethod
    def encode(self, prompts: list[str]) -> np.ndarray:
        """
        Args:
            prompts: list of N text strings
        Returns:
            Z: np.ndarray of shape (N, d), float32
        """

    @abstractmethod
    def decode(self, Z: np.ndarray, n_seeds: int = 1) -> np.ndarray:
        """
        Args:
            Z: (N, d) latent vectors
            n_seeds: number of stochastic samples per latent
        Returns:
            motions: (N * n_seeds, T, J*3), float32
        """

    @staticmethod
    def cosine_kernel(Z: np.ndarray) -> np.ndarray:
        """(N, d) → (N, N) cosine similarity matrix."""
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Z_norm = Z / norms
        return (Z_norm @ Z_norm.T).astype(np.float32)
