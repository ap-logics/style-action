"""
CLIP text embeddings for an arbitrary prompt grid (same schema as grid_v2.py).

Mirrors extract/clip_control.py exactly -- ViT-B/32 text encoder, L2-normalised
embeddings -- but consumes a grid JSON so it can be pointed at the compound
grid. Writes {out}/clip/{template}/Z_S.npy, Z_T.npy, meta.json.

Usage:
  python extract/clip_extract_grid.py --grid grid_compound.json \
      --out /data/pmyap24/sac/results_compound
"""
from __future__ import annotations
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    import clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=dev)
    model.eval()

    def encode(prompts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(prompts), args.batch):
            tok = clip.tokenize(prompts[i:i + args.batch], truncate=True).to(dev)
            with torch.no_grad():
                e = model.encode_text(tok).float()
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy().astype(np.float32))
        return np.concatenate(out)

    for g in json.load(open(args.grid)):
        t = g["template"]
        d = Path(args.out) / "clip" / str(t)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        Z_S = encode(g["neutral"])
        Z_T = np.stack([encode(row) for row in g["styled"]])
        np.save(d / "Z_S.npy", Z_S)
        np.save(d / "Z_T.npy", Z_T)
        meta = {"actions": g["actions"], "styles": g["styles"], "template": t}
        if "n_singles" in g:
            meta["n_singles"] = g["n_singles"]
        (d / "meta.json").write_text(json.dumps(meta))
        print(f"  [template {t}] {time.time()-t0:.0f}s  Z_S {Z_S.shape}  Z_T {Z_T.shape}",
              flush=True)


if __name__ == "__main__":
    main()
