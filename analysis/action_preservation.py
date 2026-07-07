"""
Classifier-free Action Preservation in decoded motion space (T2M-GPT).

Rather than trust a separate HumanML3D action classifier (which carries its
own biases, and may share training data with the model under test), we ask
a self-contained question: when a styled prompt is generated and decoded to
motion, does the resulting motion still read as the intended action?

We decode every neutral and styled prompt to motion features (mean-pooled
over time), build one prototype per action from the neutral motions, and
classify each styled motion by nearest prototype. AP is the fraction of
styled motions assigned to their intended action.

This is the behavioural complement to the latent BER: BER asks whether the
styled *latent* left its action basin, AP asks whether the styled *motion*
still looks like the action.

Usage:
  python analysis/action_preservation.py --t2mgpt_root /path/to/T2M-GPT
"""
from __future__ import annotations
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3)


def build_vqvae(repo: Path):
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self
    sys.path.insert(0, str(repo))
    import models.vqvae as vqvae
    args = Namespace(**HP, quantizer="ema_reset", mu=0.99, dataname="t2m")
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                           args.down_t, args.stride_t, args.width, args.depth,
                           args.dilation_growth_rate)
    net.load_state_dict(torch.load(repo / "pretrained/VQVAE/net_last.pth",
                        map_location="cpu", weights_only=False)["net"], strict=True)
    net.eval()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2mgpt_root", required=True)
    p.add_argument("--results", default="t2mgpt")
    p.add_argument("--template", default="0")
    args = p.parse_args()

    net = build_vqvae(Path(args.t2mgpt_root))
    base = ROOT / "results" / args.results / args.template
    tokens = json.loads((base / "tokens.json").read_text())
    meta = json.loads((base / "meta.json").read_text())
    n_styles = len(meta["styles"]); n_actions = len(meta["actions"])

    def decode(tok):
        idx = torch.tensor(tok, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            codes = net.vqvae.quantizer.dequantize(idx)          # (1,T',512)
            motion = net.vqvae.decoder(codes.permute(0, 2, 1))   # (1,263,T)
        return motion[0].mean(dim=-1).numpy()

    proto = np.stack([decode(tokens["neutral"][a]) for a in range(n_actions)])

    def nearest(m):
        return int(np.linalg.norm(proto - m[None], axis=1).argmin())

    preserved = np.zeros((n_styles, n_actions), dtype=bool)
    for j in range(n_styles):
        for a in range(n_actions):
            preserved[j, a] = nearest(decode(tokens["styled"][j][a])) == a

    ap = float(preserved.mean())
    per_action = {meta["actions"][a].replace("a person is ", ""):
                  round(float(preserved[:, a].mean()), 3) for a in range(n_actions)}
    print(f"{args.results}: motion-space AP = {ap:.3f} "
          f"(chance = {1/n_actions:.3f})")
    worst = sorted(per_action.items(), key=lambda x: x[1])[:3]
    print(f"  lowest-AP actions: {worst}")
    (ROOT / "results" / f"action_preservation_{args.results}.json").write_text(
        json.dumps({"ap": round(ap, 4), "per_action": per_action,
                    "chance": round(1 / n_actions, 4)}, indent=1))
    print(f"saved results/action_preservation_{args.results}.json")


if __name__ == "__main__":
    main()
