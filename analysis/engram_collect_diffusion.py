"""
Engram covariance collection for the diffusion generators (MDM, MLD),
extending experiment 1 beyond T2M-GPT.

Hooks nn.Linear layers of the MOTION network only (text-encoder submodules
are excluded by name), collects mean input covariances per (style, action
parity) bucket across the full denoising loop, and saves covariances +
the hooked layers' weights to one cache file. The model-agnostic compute
stage (engram_compute.py --cache ... ) then produces the overlap numbers.

CFG is left on, as in real sampling; its unconditional component is nearly
identical across buckets and is removed by the centred-overlap analysis.

Usage (from /data/pmyap24/sac):
  python analysis/engram_collect_diffusion.py --model mld \
      --grid grid_v2_all_templates.json
  python analysis/engram_collect_diffusion.py --model mdm \
      --grid grid_v2_all_templates.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MAX_IN = 1024
EXCLUDE = ("clip", "text_encoder", "vae", "clip_model")   # non-motion submodules


class CovCollector:
    def __init__(self, model: nn.Module):
        self.layers: dict[str, nn.Linear] = {}
        self.buckets: dict[str, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self._dev_acc: dict[str, torch.Tensor] = {}
        self._dev_cnt: dict[str, int] = {}
        self.active: str | None = None
        for name, mod in model.named_modules():
            low = name.lower()
            if any(x in low for x in EXCLUDE):
                continue
            if isinstance(mod, nn.Linear) and mod.in_features <= MAX_IN:
                self.layers[name] = mod
                mod.register_forward_pre_hook(self._make_hook(name))

    def set_active(self, bucket: str | None):
        if self._dev_acc:
            b = self.buckets.setdefault(self.active, {})
            n = self.counts.setdefault(self.active, {})
            for name, c in self._dev_acc.items():
                cc = c.cpu()
                if name in b:
                    b[name] += cc
                    n[name] += self._dev_cnt[name]
                else:
                    b[name] = cc
                    n[name] = self._dev_cnt[name]
            self._dev_acc, self._dev_cnt = {}, {}
        self.active = bucket

    def _make_hook(self, name):
        def hook(_mod, inputs):
            if self.active is None:
                return
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            c = x.T @ x
            if name in self._dev_acc:
                self._dev_acc[name] += c
                self._dev_cnt[name] += x.shape[0]
            else:
                self._dev_acc[name] = c
                self._dev_cnt[name] = x.shape[0]
        return hook


# ---------------- model runners ----------------

def make_mdm_runner(dev):
    """Returns (collector_target_module, run(prompts) callable)."""
    repo = "/data/pmyap24/sac/motion-diffusion-model"
    sys.path.insert(0, repo)
    import os
    os.chdir(repo)
    from types import SimpleNamespace
    from argparse import Namespace
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from model.cfg_sampler import ClassifierFreeSampleModel

    ckpt = f"{repo}/save/humanml_enc_512_50steps/model000750000.pt"
    margs = Namespace(**json.load(open(Path(ckpt).parent / "args.json")))
    dist_util.setup_dist(-1)
    model, diffusion = create_model_and_diffusion(
        margs, SimpleNamespace(dataset=SimpleNamespace()))
    load_saved_model(model, ckpt, use_avg=False)
    model.to(dev)
    model.eval()
    sample_model = ClassifierFreeSampleModel(model)
    n_frames = 120

    def run(prompts):
        B = len(prompts)
        kw = {"y": {"text": list(prompts),
                    "mask": torch.ones(B, 1, 1, n_frames, dtype=torch.bool, device=dev),
                    "lengths": torch.tensor([n_frames] * B, device=dev),
                    "scale": torch.ones(B, device=dev) * 2.5}}
        with torch.no_grad():
            diffusion.p_sample_loop(sample_model, (B, model.njoints, model.nfeats, n_frames),
                                    clip_denoised=False, model_kwargs=kw, progress=False)
    return model, run


def make_mld_runner(dev):
    repo = "/data/pmyap24/sac/motion-latent-diffusion"
    sys.path.insert(0, repo)
    import os
    os.chdir(repo)
    from types import SimpleNamespace
    from omegaconf import OmegaConf, open_dict
    from mld.config import parse_args
    from mld.models.get_model import get_model

    sys.argv = ["x", "--cfg", "configs/config_mld_humanml3d.yaml",
                "--cfg_assets", "configs/assets.yaml"]
    cfg = parse_args()
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.DATASET.NFEATS = 263
        cfg.DATASET.NJOINTS = 22
    stub = SimpleNamespace(feats2joints=lambda x: x, renorm4t2m=lambda x: x, is_mm=False)
    model = get_model(cfg, stub)
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(dev)
    model.eval()

    def run(prompts):
        texts = list(prompts)
        lengths = [196] * len(texts)
        with torch.no_grad():
            if model.do_classifier_free_guidance:
                texts = [""] * len(prompts) + texts
            emb = model.text_encoder(texts)
            model._diffusion_reverse(emb, lengths)
    # hook the denoiser only (text_encoder excluded by name anyway)
    return model.denoiser, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mdm", "mld"])
    p.add_argument("--grid", required=True)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid_path = str(Path(args.grid).resolve())

    if args.model == "mdm":
        target, run = make_mdm_runner(dev)
    else:
        target, run = make_mld_runner(dev)
    coll = CovCollector(target)
    print(f"hooked {len(coll.layers)} linear layers on {args.model}", flush=True)

    grids = json.load(open(grid_path))
    styles = grids[0]["styles"]
    nA = len(grids[0]["actions"])

    t0 = time.time()
    for g in grids:
        for parity in (0, 1):
            idx = [a for a in range(nA) if a % 2 == parity]
            coll.set_active(f"neutral|{parity}")
            run([g["neutral"][a] for a in idx])
            for j, s in enumerate(styles):
                coll.set_active(f"{s}|{parity}")
                run([g["styled"][j][a] for a in idx])
            coll.set_active(None)
        print(f"template {g['template']} done ({time.time()-t0:.0f}s)", flush=True)
    coll.set_active(None)

    weights = {n: m.weight.detach().float().cpu() for n, m in coll.layers.items()}
    out = Path(f"/data/pmyap24/sac/results/engram_covs_{args.model}.pt")
    torch.save({"buckets": coll.buckets, "counts": coll.counts,
                "weights": weights}, out)
    print(f"saved covariances + weights to {out}", flush=True)


if __name__ == "__main__":
    main()
