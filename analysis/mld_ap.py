"""
Classifier-free Action Preservation for MLD: decode the saved diffusion
latents through MLD's VAE decoder to motion features, mean-pool, build one
prototype per action from the neutral decodes, and classify each styled
decode by nearest prototype (cosine). Run from the MLD repo root:

  python mld_ap.py --cfg configs/config_mld_humanml3d.yaml \
      --cfg_assets configs/assets.yaml \
      --latents $SAC_ROOT/results_v2/mld_v2/seed42/0 \
      --out $SAC_ROOT/results/ap_mld_seed42.json
"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from mld.config import parse_args
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def decode_batch(model, Z, device, chunk=24):
    feats = []
    for i in range(0, Z.shape[0], chunk):
        z = torch.tensor(Z[i:i+chunk], dtype=torch.float32, device=device)
        lengths = [196] * z.shape[0]
        with torch.no_grad():
            for shape in [(1, z.shape[0], -1), (z.shape[0], 1, -1)]:
                try:
                    out = model.vae.decode(z.reshape(*shape), lengths)
                    break
                except Exception as e:
                    err = e
            else:
                raise err
        feats.append(out.detach().cpu().numpy().mean(axis=1))  # pool time
    return np.concatenate(feats)


def main():
    argv = sys.argv
    lat_dir = Path(argv[argv.index("--latents") + 1]); argv.pop(argv.index("--latents") + 1); argv.remove("--latents")
    out_path = Path(argv[argv.index("--out") + 1]); argv.pop(argv.index("--out") + 1); argv.remove("--out")

    cfg = parse_args()
    create_logger(cfg, phase="demo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from omegaconf import OmegaConf, open_dict
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.DATASET.NFEATS = 263
        cfg.DATASET.NJOINTS = 22
    stub = SimpleNamespace(feats2joints=lambda x: x, renorm4t2m=lambda x: x, is_mm=False)
    model = get_model(cfg, stub)
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    Z_S = np.load(lat_dir / "Z_S.npy")                    # (A, 256)
    Z_T = np.load(lat_dir / "Z_T.npy")                    # (S, A, 256)
    meta = json.load(open(lat_dir / "meta.json"))
    S, A = Z_T.shape[0], Z_T.shape[1]

    F_S = decode_batch(model, Z_S, device)                # (A, 263)
    proto = F_S / (np.linalg.norm(F_S, axis=1, keepdims=True) + 1e-8)
    per_style, per_action = {}, {a: 0 for a in range(A)}
    correct = 0
    correct_l2 = 0
    for j in range(S):
        F = decode_batch(model, Z_T[j], device)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        pred = (Fn @ proto.T).argmax(1)
        pred_l2 = np.linalg.norm(F[:, None] - F_S[None], axis=2).argmin(1)
        correct_l2 += int((pred_l2 == np.arange(A)).sum())
        hit = (pred == np.arange(A))
        F = Fn
        per_style[meta["styles"][j]] = round(float(hit.mean()), 4)
        for a in np.where(~hit)[0]:
            per_action[int(a)] += 1
        correct += int(hit.sum())
        print(f"style {meta['styles'][j]}: AP {hit.mean():.3f}", flush=True)

    result = {
        "ap_overall": round(correct / (S * A), 4),
        "ap_overall_l2": round(correct_l2 / (S * A), 4),
        "chance": round(1 / A, 4),
        "per_style": per_style,
        "misses_per_action": {meta["actions"][a]: n for a, n in per_action.items() if n},
        "n_styles": S, "n_actions": A, "seed": meta.get("seed"),
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(f"AP overall {result['ap_overall']} (chance {result['chance']})")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
