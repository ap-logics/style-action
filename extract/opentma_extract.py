"""
OpenTMA text-encoder probe: the data-controlled text-encoder comparison.

Loads the DistilbertActorAgnosticEncoder from each OpenTMA checkpoint
(same architecture, three training corpora: HumanML3D / Motion-X / UniMoCap)
and encodes the full prompt grid. The embedding is the mean (mu) of the
text latent distribution, shape (N, 256).

Question this answers: does contrastive text-motion alignment preserve the
style axis the way plain CLIP does, and does training-corpus richness
(Motion-X vs HumanML3D) change that?

Run from the OpenTMA repo root:
  python opentma_extract.py --grid grid_all_templates.json \
      --out /data/pmyap24/sac/results
Outputs results/opentma_{h3d,motionx,unimocap}/{t}/Z_S.npy etc.,
same layout as the other extractors, scoreable by score_hpc.py.
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from tma.models.architectures.temos.textencoder.distillbert_actor import (
    DistilbertActorAgnosticEncoder)

CKPTS = {
    "opentma_h3d": "checkpoints/humanml3d.ckpt",
    "opentma_motionx": "checkpoints/motionx.ckpt",
    "opentma_unimocap": "checkpoints/unimocap.ckpt",
}


def _tolerant_load(path):
    """torch.load, stubbing any missing modules the pickle references
    (we only need the state_dict tensors, not the pickled config objects)."""
    import sys, types

    def make_stub(name):
        stub = types.ModuleType(name)
        stub.__path__ = []          # behave like a package for submodule imports

        def _getattr(attr, _n=name):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return type(attr, (), {})
        stub.__getattr__ = _getattr
        sys.modules[name] = stub

    for _ in range(20):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            make_stub(e.name)
    raise RuntimeError(f"could not load {path}")


def load_textencoder(ckpt_path: str, dev: str):
    enc = DistilbertActorAgnosticEncoder(
        modelpath="distilbert-base-uncased", finetune=False, vae=True,
        latent_dim=256, ff_size=1024, num_layers=4, num_heads=4)
    sd = _tolerant_load(ckpt_path)
    sd = sd.get("state_dict", sd)
    sub = {k[len("textencoder."):]: v for k, v in sd.items()
           if k.startswith("textencoder.")}
    missing, unexpected = enc.load_state_dict(sub, strict=False)
    # DistilBERT backbone weights are frozen/pretrained and may be absent
    # from the checkpoint when finetune=False; anything else missing is a bug.
    non_bert_missing = [k for k in missing if not k.startswith("text_model")]
    assert not non_bert_missing, f"missing non-backbone keys: {non_bert_missing[:5]}"
    print(f"  loaded {len(sub)} tensors "
          f"({len(missing)} backbone-missing, {len(unexpected)} unexpected)")
    enc.eval().to(dev)
    return enc


def encode(enc, prompts, dev) -> np.ndarray:
    with torch.no_grad():
        dist = enc(prompts)
    return dist.loc.cpu().numpy()   # (N, 256)


def main():
    p = ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    grids = json.load(open(args.grid))

    for name, ckpt in CKPTS.items():
        print(f"=== {name} ({ckpt}) ===")
        enc = load_textencoder(ckpt, dev)
        for g in grids:
            t = g["template"]
            out = Path(args.out) / name / str(t)
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            Z_S = encode(enc, g["neutral"], dev)
            Z_T = np.stack([encode(enc, row, dev) for row in g["styled"]])

            np.save(out / "Z_S.npy", Z_S)
            np.save(out / "Z_T.npy", Z_T)
            (out / "meta.json").write_text(json.dumps(
                {"actions": g["actions"], "styles": g["styles"],
                 "template": t, "ckpt": ckpt}))
            print(f"  [template {t}] {time.time()-t0:.0f}s  "
                  f"Z_S {Z_S.shape}  Z_T {Z_T.shape}", flush=True)
        del enc
    print("ALL ENCODERS COMPLETE")


if __name__ == "__main__":
    main()
