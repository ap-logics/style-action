"""
Day-one MDM latent probe. Runs INSIDE the MDM repo root on the cluster.

Loads the checkpoint once, then for a single prompt:
  1. grabs the CLIP text embedding from model.encode_text()   -> (1, 512)
  2. runs the full 50-step sampling loop to get denoised x0   -> (1, 263, 1, T)
  3. mean-pools x0 over time                                  -> (263,)
Saves both vectors and prints shapes. This validates the extraction sites
used by eval/extract/mdm.py before running the full grid.

Usage (from MDM repo root, inside srun with a GPU):
  python mdm_probe.py --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
                      --text_prompt "a person is walking"
"""
import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util


def load_args_from_checkpoint(model_path: str) -> Namespace:
    args_path = Path(model_path).parent / "args.json"
    with open(args_path) as f:
        ckpt_args = json.load(f)
    args = Namespace(**ckpt_args)
    args.model_path = model_path
    return args


def main():
    p = ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--text_prompt", default="a person is walking")
    p.add_argument("--n_frames", type=int, default=120)
    p.add_argument("--guidance_param", type=float, default=2.5)
    p.add_argument("--out", default="probe_output")
    cli = p.parse_args()

    args = load_args_from_checkpoint(cli.model_path)
    dist_util.setup_dist(-1)
    device = dist_util.dev()

    print("Building model ...")
    from types import SimpleNamespace
    dummy_data = SimpleNamespace(dataset=SimpleNamespace())
    model, diffusion = create_model_and_diffusion(args, dummy_data)
    load_saved_model(model, cli.model_path, use_avg=False)
    model.to(device)   # MDM.to() returns None, do not chain
    model.eval()

    # ---- extraction site 1: CLIP text embedding ----
    with torch.no_grad():
        text_emb = model.encode_text([cli.text_prompt])
    print(f"[1] CLIP text embedding: {tuple(text_emb.shape)}")

    # ---- extraction site 2: denoised x0 via full sampling loop ----
    n_frames = cli.n_frames
    shape = (1, model.njoints, model.nfeats, n_frames)
    model_kwargs = {
        "y": {
            "text": [cli.text_prompt],
            "mask": torch.ones(1, 1, 1, n_frames, dtype=torch.bool, device=device),
            "lengths": torch.tensor([n_frames], device=device),
            "scale": torch.ones(1, device=device) * cli.guidance_param,
        }
    }

    from model.cfg_sampler import ClassifierFreeSampleModel
    sample_model = ClassifierFreeSampleModel(model) if cli.guidance_param != 1 \
        else model

    print(f"Sampling ({args.diffusion_steps} steps) ...")
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            sample_model, shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )
    print(f"[2] denoised x0: {tuple(sample.shape)}")

    x0 = sample[0].squeeze().cpu().numpy()      # (263, T) after squeeze
    latent = x0.mean(axis=-1)                    # pool over time -> (263,)
    print(f"[3] pooled latent: {latent.shape}")

    np.save(f"{cli.out}_text_emb.npy", text_emb.cpu().numpy())
    np.save(f"{cli.out}_x0.npy", x0)
    np.save(f"{cli.out}_latent.npy", latent)
    print(f"Saved {cli.out}_{{text_emb,x0,latent}}.npy — day-one finish line reached.")


if __name__ == "__main__":
    main()
