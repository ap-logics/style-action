"""
CLIP text-encoder control.

Runs the same CKA/GED pipeline on raw CLIP-ViT-B/32 text embeddings,
with no motion model involved. This establishes whether coupling in the
motion latent space is inherited from the text encoder or introduced by
the model itself.

If a motion model's CKA(K_S, K_T) is lower than CLIP's, the coupling
is the model's fault. If it matches CLIP, the model is not adding
anything beyond what the text encoder already does.
"""
from __future__ import annotations
import numpy as np
import torch

from base import LatentExtractor


class CLIPControlExtractor(LatentExtractor):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import clip
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")
        self._clip = clip
        model, _ = clip.load("ViT-B/32", device=self.device)
        model.eval()
        self._model = model

    def encode(self, prompts: list[str]) -> np.ndarray:
        """Raw CLIP text embeddings, L2-normalised. Shape (N, 512)."""
        self._load()
        tokens = self._clip.tokenize(prompts, truncate=True).to(self.device)
        with torch.no_grad():
            embs = self._model.encode_text(tokens).float()
            embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy().astype(np.float32)

    def decode(self, Z: np.ndarray, n_seeds: int = 1) -> np.ndarray:
        raise NotImplementedError("CLIP control has no decoder.")

    def generate_from_prompt(self, prompt: str, n_seeds: int = 1) -> np.ndarray:
        raise NotImplementedError("CLIP control has no motion generator.")
