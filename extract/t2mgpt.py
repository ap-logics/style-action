"""
T2M-GPT latent extractor.

Extraction site:
  encode → VQ-VAE encoder output *before* quantisation → continuous
           z ∈ R^(T × 512). Mean-pool over T → z ∈ R^512.
           Pre-quantisation keeps vectors continuous for cosine similarity.
  decode → full autoregressive generation (text → token seq → VQ decoder).

Requires:
  - T2M-GPT repo cloned at $T2MGPT_ROOT
    (default: ../../models/T2M-GPT)
  - VQ-VAE checkpoint at $T2MGPT_VQVAE_CKPT
  - GPT checkpoint at  $T2MGPT_GPT_CKPT
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from base import LatentExtractor


@dataclass
class T2MGPTConfig:
    root: str = os.environ.get(
        "T2MGPT_ROOT",
        str(Path(__file__).parent.parent.parent / "models" / "T2M-GPT"),
    )
    vqvae_ckpt: str = os.environ.get("T2MGPT_VQVAE_CKPT", "")
    gpt_ckpt: str   = os.environ.get("T2MGPT_GPT_CKPT", "")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_frames: int = 196
    top_k: int = 1          # deterministic decoding; set > 1 for stochastic


class T2MGPTExtractor(LatentExtractor):
    def __init__(self, cfg: T2MGPTConfig | None = None):
        self.cfg = cfg or T2MGPTConfig()
        self._vqvae = None
        self._gpt   = None
        self._clip  = None

    def _load(self):
        if self._vqvae is not None:
            return
        root = self.cfg.root
        if root not in sys.path:
            sys.path.insert(0, root)

        import clip as openai_clip
        from models.vq.model import VQVAE
        from models.t2m_trans import Text2Motion_Transformer
        import options.option_transformer as option_trans

        self._clip_model, _ = openai_clip.load("ViT-B/32", device=self.cfg.device)
        self._clip_model.eval()

        # VQ-VAE
        args = option_trans.get_args_parser()
        vqvae = VQVAE(args, os.path.join(root, "checkpoints", "kit", "Comp_v6_KLD005"))
        vqvae_ckpt = torch.load(self.cfg.vqvae_ckpt, map_location="cpu")
        vqvae.load_state_dict(vqvae_ckpt["net"])
        vqvae.to(self.cfg.device).eval()
        self._vqvae = vqvae

        # GPT
        gpt = Text2Motion_Transformer(
            num_vq=args.nb_code,
            embed_dim=args.embed_dim_gpt,
            clip_dim=args.clip_dim,
            block_size=args.block_size,
            num_layers=args.num_layers,
            n_head=args.n_head_gpt,
            drop_out_rate=args.drop_out_rate,
            fc_rate=args.ff_rate,
        )
        gpt_ckpt = torch.load(self.cfg.gpt_ckpt, map_location="cpu")
        gpt.load_state_dict(gpt_ckpt["net"])
        gpt.to(self.cfg.device).eval()
        self._gpt = gpt

    def encode(self, prompts: list[str]) -> np.ndarray:
        """
        For each prompt: generate a token sequence via the GPT, look up
        the VQ-VAE codebook embeddings, mean-pool → z ∈ R^512.
        Returns (N, 512).
        """
        self._load()
        import clip as openai_clip

        results = []
        for prompt in prompts:
            tokens = openai_clip.tokenize([prompt], truncate=True).to(self.cfg.device)
            with torch.no_grad():
                text_feat = self._clip_model.encode_text(tokens).float()   # (1, 512)

                # generate index sequence autoregressively
                index_seq = self._gpt.sample(
                    text_feat,
                    if_categorial=False,  # greedy
                )  # (1, L) int

                # look up codebook embeddings (pre-quantisation continuous vectors)
                codebook = self._vqvae.vq_layer.embedding.weight  # (nb_code, code_dim)
                z_codes = codebook[index_seq[0]]  # (L, code_dim)
                z_mean = z_codes.mean(dim=0).cpu().numpy()  # (code_dim,)
            results.append(z_mean)

        return np.stack(results).astype(np.float32)  # (N, code_dim)

    def decode(self, Z: np.ndarray, n_seeds: int = 1) -> np.ndarray:
        """
        Not meaningful for T2M-GPT: generation is text-conditioned, not
        latent-conditioned. Pipeline calls generate_from_prompt for AP.
        """
        raise NotImplementedError(
            "T2M-GPT decode takes text, not latents. "
            "Use generate_from_prompt() instead."
        )

    def generate_from_prompt(self, prompt: str, n_seeds: int = 5) -> np.ndarray:
        """Generate n_seeds motions from a single text prompt. Returns (n_seeds, T, J*3)."""
        self._load()
        import clip as openai_clip

        tokens = openai_clip.tokenize([prompt], truncate=True).to(self.cfg.device)
        motions = []
        with torch.no_grad():
            text_feat = self._clip_model.encode_text(tokens).float()
            for _ in range(n_seeds):
                index_seq = self._gpt.sample(
                    text_feat,
                    if_categorial=(n_seeds > 1),   # stochastic if multiple seeds
                )
                motion = self._vqvae.forward_decoder(index_seq)  # (1, T, J*3)
                motions.append(motion[0].cpu().numpy())

        return np.stack(motions)  # (n_seeds, T, J*3)
