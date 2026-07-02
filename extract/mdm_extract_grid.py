"""
Full-grid MDM latent extraction. Self-contained: runs in the MDM repo root,
reads prompts from grid_all_templates.json (exported by prompts/grid.py so
cluster and laptop use byte-identical prompts).

For each of 3 paraphrase templates x 64 prompts (8 neutral + 7x8 styled):
  z = mean-pooled denoised x0 from the full 50-step sampling loop  (263,)
Also saves the CLIP text embedding for every prompt (512,) as a bonus
control. Prompts are batched 8 at a time (one action row per batch).

Output (per template t):
  out/{t}/Z_S.npy  (8, 263)     out/{t}/Z_T.npy  (7, 8, 263)
  out/{t}/E_S.npy  (8, 512)     out/{t}/E_T.npy  (7, 8, 512)
  out/{t}/meta.json

Usage:
  python mdm_extract_grid.py --model_path save/humanml_enc_512_50steps/model000750000.pt \
      --grid grid_all_templates.json --out /data/pmyap24/sac/results/mdm --seed 42
"""
import json
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel

N_FRAMES = 120
GUIDANCE = 2.5


def build(model_path: str):
    args = Namespace(**json.load(open(Path(model_path).parent / "args.json")))
    dist_util.setup_dist(-1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(
        args, SimpleNamespace(dataset=SimpleNamespace()))
    load_saved_model(model, model_path, use_avg=False)
    model.to(device)   # MDM.to() returns None — do not chain
    model.eval()
    return model, diffusion, device


def encode_batch(model, diffusion, device, prompts: list[str],
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pooled x0 latents (B, 263), CLIP text embeddings (B, 512))."""
    B = len(prompts)
    torch.manual_seed(seed)   # fixed noise across conditions: only text varies

    with torch.no_grad():
        text_emb = model.encode_text(prompts).squeeze(0)  # (B, 512)

    sample_model = ClassifierFreeSampleModel(model)
    shape = (B, model.njoints, model.nfeats, N_FRAMES)
    model_kwargs = {"y": {
        "text": prompts,
        "mask": torch.ones(B, 1, 1, N_FRAMES, dtype=torch.bool, device=device),
        "lengths": torch.tensor([N_FRAMES] * B, device=device),
        "scale": torch.ones(B, device=device) * GUIDANCE,
    }}
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            sample_model, shape, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False,
        )   # (B, 263, 1, T)
    x0 = sample.squeeze(2).cpu().numpy()          # (B, 263, T)
    return x0.mean(axis=-1), text_emb.cpu().numpy()


def main():
    p = ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    cli = p.parse_args()

    grids = json.load(open(cli.grid))
    model, diffusion, device = build(cli.model_path)

    for g in grids:
        t = g["template"]
        out = Path(cli.out) / str(t)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        print(f"[template {t}] neutral row ...", flush=True)
        Z_S, E_S = encode_batch(model, diffusion, device, g["neutral"], cli.seed)

        Z_T, E_T = [], []
        for j, row in enumerate(g["styled"]):
            print(f"[template {t}] style '{g['styles'][j]}' ...", flush=True)
            z, e = encode_batch(model, diffusion, device, row, cli.seed)
            Z_T.append(z); E_T.append(e)

        np.save(out / "Z_S.npy", Z_S)
        np.save(out / "Z_T.npy", np.stack(Z_T))
        np.save(out / "E_S.npy", E_S)
        np.save(out / "E_T.npy", np.stack(E_T))
        (out / "meta.json").write_text(json.dumps(
            {"actions": g["actions"], "styles": g["styles"],
             "template": t, "n_frames": N_FRAMES, "guidance": GUIDANCE,
             "seed": cli.seed, "model_path": cli.model_path}))
        print(f"[template {t}] done in {time.time()-t0:.0f}s  "
              f"Z_S {Z_S.shape}  Z_T {np.stack(Z_T).shape}", flush=True)

    print("ALL TEMPLATES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
