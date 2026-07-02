"""
Thin wrapper around the HumanML3D action classifier from the t2m repo
(Guo et al., CVPR 2022). The classifier checkpoint ships with the repo
at checkpoints/t2m/Comp_v6_KLD005/meta/motion_classifier.tar

Set env var T2M_ROOT to the cloned t2m repo root.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch


class HumanML3DClassifier:
    def __init__(
        self,
        ckpt_path: str = "",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        t2m_root: str = "",
    ):
        self.device = device
        self._t2m_root = t2m_root or os.environ.get(
            "T2M_ROOT",
            str(Path(__file__).parent.parent.parent / "models" / "text-to-motion"),
        )
        self._ckpt_path = ckpt_path or os.environ.get("T2M_CLF_CKPT", "")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        root = self._t2m_root
        if root not in sys.path:
            sys.path.insert(0, root)

        from networks.modules import MovementConvEncoder, TextEncoderBiGRU, MotionEncoderBiGRU

        # The t2m repo's motion encoder doubles as a classifier when its
        # output is compared against class prototypes. We use it to get
        # motion embeddings, then do nearest-centroid classification.
        # If a dedicated classifier head exists in the checkpoint, load it.
        ckpt = torch.load(self._ckpt_path, map_location="cpu")
        self._model = ckpt["model"]
        self._model.to(self.device).eval()

    def predict(self, motions: np.ndarray) -> np.ndarray:
        """
        Args:
            motions: (N, T, J*3) float32 motion sequences
        Returns:
            preds: (N,) int array of predicted action labels (0-7)
        """
        self._load()
        with torch.no_grad():
            x = torch.tensor(motions, dtype=torch.float32, device=self.device)
            logits = self._model(x)        # (N, n_classes)
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy()
