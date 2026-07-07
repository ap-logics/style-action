"""
MLD (Motion Latent Diffusion) grid extraction. Run from the MLD repo root
with the framework's arg parser, e.g.

  python mld_extract_grid.py --cfg configs/config_mld_humanml3d.yaml \
      --cfg_assets configs/assets.yaml \
      --grid grid_v2_all_templates.json \
      --out /data/pmyap24/sac/results/mld_v2/seed42 --seed 42

Extraction site: the sampled diffusion latent z = _diffusion_reverse(text_emb),
shape (bsz, 1, 256) -> squeezed to (256,). This is the latent the diffusion
model reasons in, the latent-diffusion analogue of MDM's denoised x0 and
T2M-GPT's codebook latent. VAE decode is not needed for the coupling metrics.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from types import SimpleNamespace

from mld.config import parse_args
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def encode_prompts(model, prompts, device):
    """Return sampled latents (n, 256) for a list of prompts."""
    lengths = [196] * len(prompts)
    texts = list(prompts)
    with torch.no_grad():
        if model.do_classifier_free_guidance:
            uncond = [""] * len(texts)
            texts = uncond + texts
        text_emb = model.text_encoder(texts)
        z = model._diffusion_reverse(text_emb, lengths)   # (n, 1, 256) or (1, n, 256)
    z = z.detach().cpu().numpy()
    z = z.reshape(z.shape[0] if z.shape[0] == len(prompts) else z.shape[1], -1)
    return z


def main():
    # extra args parsed manually, then handed to MLD's parser via argv
    argv = sys.argv
    grid_path = argv[argv.index("--grid") + 1]; argv.pop(argv.index("--grid") + 1); argv.remove("--grid")
    out_path = argv[argv.index("--out") + 1]; argv.pop(argv.index("--out") + 1); argv.remove("--out")
    seed = 42
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1]); argv.pop(argv.index("--seed") + 1); argv.remove("--seed")

    cfg = parse_args()
    logger = create_logger(cfg, phase="demo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get_datasets normally injects these; we set them for HumanML3D directly
    from omegaconf import OmegaConf, open_dict
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.DATASET.NFEATS = 263
        cfg.DATASET.NJOINTS = 22

    # stub datamodule: MLD only touches datamodule.feats2joints, on the decode
    # path we never take (we extract the diffusion latent, not decoded motion).
    stub = SimpleNamespace(feats2joints=lambda x: x, renorm4t2m=lambda x: x,
                           is_mm=False)
    model = get_model(cfg, stub)
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device).eval()

    grids = json.load(open(grid_path))
    for g in grids:
        t = g["template"]
        out = Path(out_path) / str(t)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        torch.manual_seed(seed)
        Z_S = encode_prompts(model, g["neutral"], device)
        Z_T = []
        for j, row in enumerate(g["styled"]):
            torch.manual_seed(seed)
            Z_T.append(encode_prompts(model, row, device))
            print(f"[template {t}] style '{g['styles'][j]}' done", flush=True)
        Z_T = np.stack(Z_T)

        np.save(out / "Z_S.npy", Z_S)
        np.save(out / "Z_T.npy", Z_T)
        (out / "meta.json").write_text(json.dumps(
            {"actions": g["actions"], "styles": g["styles"], "template": t,
             "seed": seed, "model": "mld"}))
        print(f"[template {t}] done in {time.time()-t0:.0f}s  "
              f"Z_S {Z_S.shape}  Z_T {Z_T.shape}", flush=True)
    print("ALL TEMPLATES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
