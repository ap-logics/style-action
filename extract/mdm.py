"""
MDM latent extractor.

Extraction site:
  encode → CLIP-ViT-B/32 text embedding z ∈ R^512, then run DDIM-50
           denoising to get x0 ∈ R^(T×263). Mean-pool over T → z ∈ R^263.
  decode → full DDIM-50 generation from the conditioning text embedding.

Requires:
  - MDM repo cloned at $MDM_ROOT (default: ../../models/motion-diffusion-model)
  - HumanML3D checkpoint at $MDM_CKPT
  - env vars or config passed via MDMConfig
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from base import LatentExtractor


@dataclass
class MDMConfig:
    mdm_root: str = os.environ.get(
        "MDM_ROOT",
        str(Path(__file__).parent.parent.parent / "models" / "motion-diffusion-model"),
    )
    ckpt_path: str = os.environ.get("MDM_CKPT", "")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ddim_steps: int = 50
    guidance_param: float = 2.5
    max_frames: int = 196


class MDMExtractor(LatentExtractor):
    def __init__(self, cfg: MDMConfig | None = None):
        self.cfg = cfg or MDMConfig()
        self._model = None
        self._diffusion = None
        self._clip = None

    def _load(self):
        if self._model is not None:
            return
        root = self.cfg.mdm_root
        if root not in sys.path:
            sys.path.insert(0, root)

        from utils.model_util import create_model_and_diffusion, load_model_wo_clip
        from utils.parser_util import get_cond_mode
        import clip as openai_clip

        # load CLIP separately so we can use it directly for encoding
        self._clip_model, _ = openai_clip.load("ViT-B/32", device=self.cfg.device)
        self._clip_model.eval()

        # load MDM
        args_path = Path(self.cfg.ckpt_path).parent / "args.json"
        import json
        with open(args_path) as f:
            args_dict = json.load(f)

        from argparse import Namespace
        args = Namespace(**args_dict)
        args.device = self.cfg.device

        model, diffusion = create_model_and_diffusion(args, None)
        state = torch.load(self.cfg.ckpt_path, map_location="cpu")
        load_model_wo_clip(model, state)
        model.to(self.cfg.device)
        model.eval()

        self._model = model
        self._diffusion = diffusion

    def encode(self, prompts: list[str]) -> np.ndarray:
        """
        Returns mean-pooled denoised x0 for each prompt: (N, 263).
        Using DDIM-50 denoising conditioned on text; mean-pooling captures
        the action-style structure that the denoiser has settled on.
        """
        self._load()
        import clip as openai_clip

        results = []
        for prompt in prompts:
            tokens = openai_clip.tokenize([prompt]).to(self.cfg.device)
            with torch.no_grad():
                text_emb = self._clip_model.encode_text(tokens)  # (1, 512)

            # DDIM sample to get x0
            shape = (1, 1, 263, self.cfg.max_frames)
            model_kwargs = {"y": {"text": [prompt]}}

            sample = self._diffusion.ddim_sample_loop(
                self._model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                device=self.cfg.device,
            )  # (1, 1, 263, T)

            x0 = sample[0, 0].permute(1, 0).cpu().numpy()  # (T, 263)
            results.append(x0.mean(axis=0))                 # (263,)

        return np.stack(results).astype(np.float32)          # (N, 263)

    def decode(self, Z: np.ndarray, n_seeds: int = 5) -> np.ndarray:
        """
        Z is ignored here — MDM generates directly from text. We store the
        prompts alongside Z in the pipeline and re-generate for AP scoring.
        This is called by the pipeline with the neutral prompts.
        Raises NotImplementedError to signal the pipeline to use generate_from_prompts.
        """
        raise NotImplementedError(
            "MDM decode takes text, not latent vectors. "
            "Use pipeline.generate_motions_mdm() instead."
        )

    def generate_from_prompt(self, prompt: str, n_seeds: int = 5) -> np.ndarray:
        """Generate n_seeds motions from a single text prompt. Returns (n_seeds, T, 263)."""
        self._load()
        motions = []
        for _ in range(n_seeds):
            shape = (1, 1, 263, self.cfg.max_frames)
            model_kwargs = {"y": {"text": [prompt]}}
            sample = self._diffusion.ddim_sample_loop(
                self._model, shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None,
                progress=False, dump_steps=None,
                noise=None, const_noise=False,
                device=self.cfg.device,
            )
            motions.append(sample[0, 0].permute(1, 0).cpu().numpy())  # (T, 263)
        return np.stack(motions)  # (n_seeds, T, 263)
