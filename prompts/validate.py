"""
Screens the 8x8 prompt grid for near-synonymy between action and style terms
in CLIP embedding space. Pairs with cosine similarity > THRESHOLD are flagged
— they risk the text encoder treating the style modifier as semantically
redundant with the action, which would suppress the style delta we want to measure.

Requires: pip install git+https://github.com/openai/CLIP.git torch
"""
from __future__ import annotations
import sys
import numpy as np

THRESHOLD = 0.90  # flag if action-token and style-token cosine similarity exceeds this

from grid import ACTIONS, STYLES


def _embed(texts: list[str], device: str = "cpu"):
    try:
        import clip
        import torch
    except ImportError:
        sys.exit("Install CLIP: pip install git+https://github.com/openai/CLIP.git torch")

    model, preprocess = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embs = model.encode_text(tokens).float()
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy()


def validate(threshold: float = THRESHOLD, device: str = "cpu") -> list[tuple]:
    # embed just the bare action nouns and style adverbs
    action_terms = [a.replace("a person ", "") for a in ACTIONS]   # "walking", "running" …
    style_terms  = [s for s in STYLES if s]                        # skip neutral ""

    all_terms = action_terms + style_terms
    embs = _embed(all_terms, device)
    a_embs = embs[: len(action_terms)]
    s_embs = embs[len(action_terms) :]

    flagged = []
    sims = a_embs @ s_embs.T   # (8, 7)

    for ai, action in enumerate(action_terms):
        for si, style in enumerate(style_terms):
            sim = float(sims[ai, si])
            if sim > threshold:
                flagged.append((action, style, sim))
                print(f"  WARN  '{action}' ↔ '{style}'  sim={sim:.3f} > {threshold}")

    if not flagged:
        print(f"All pairs OK (max cosine similarity < {threshold})")
    return flagged


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    flagged = validate(args.threshold, args.device)
    sys.exit(1 if flagged else 0)
