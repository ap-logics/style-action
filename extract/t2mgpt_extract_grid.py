"""
Full-grid T2M-GPT latent extraction. Self-contained: runs in the T2M-GPT
repo root, reads prompts from grid_all_templates.json.

Latent = mean-pooled VQ codebook vectors of the generated token sequence
(512,). Token sequences are also saved so the causal transfer test can
decode edited latents later without re-running generation.

Output per template t:
  out/{t}/Z_S.npy (8, 512)   out/{t}/Z_T.npy (7, 8, 512)
  out/{t}/tokens.json        raw token index lists (ragged)
  out/{t}/meta.json

Usage:
  python t2mgpt_extract_grid.py --grid grid_all_templates.json \
      --out /data/pmyap24/sac/results/t2mgpt --seed 42
"""
import json
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
import clip

import models.vqvae as vqvae
import models.t2m_trans as trans

HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3,
          embed_dim_gpt=1024, clip_dim=512, block_size=51, num_layers=9,
          n_head_gpt=16, drop_out_rate=0.1, ff_rate=4)


def build(vq_ckpt: str, trans_ckpt: str, dev: str):
    args = Namespace(**HP, quantizer="ema_reset", mu=0.99, dataname="t2m")
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                           args.output_emb_width, args.down_t, args.stride_t,
                           args.width, args.depth, args.dilation_growth_rate)
    net.load_state_dict(torch.load(vq_ckpt, map_location="cpu")["net"], strict=True)
    net.eval().to(dev)

    tr = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim, block_size=args.block_size,
        num_layers=args.num_layers, n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
    tr.load_state_dict(torch.load(trans_ckpt, map_location="cpu")["trans"], strict=True)
    tr.eval().to(dev)

    clip_model, _ = clip.load("ViT-B/32", device=dev, jit=False)
    clip_model.eval()
    return net, tr, clip_model


def encode_prompt(net, tr, clip_model, prompt: str, dev: str,
                  seed: int, categorical: bool = False) -> tuple[np.ndarray, list]:
    torch.manual_seed(seed)
    with torch.no_grad():
        tokens = clip.tokenize([prompt], truncate=True).to(dev)
        feat = clip_model.encode_text(tokens).float()
        # categorical sampling varies with seed; greedy is deterministic
        idx = tr.sample(feat, if_categorial=categorical)
        codes = net.vqvae.quantizer.dequantize(idx)     # (1, T', 512)
        z = codes[0].mean(dim=0).cpu().numpy()
    return z, idx[0].cpu().tolist()


def main():
    p = ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vq_ckpt", default="pretrained/VQVAE/net_last.pth")
    p.add_argument("--trans_ckpt",
                   default="pretrained/VQTransformer_corruption05/net_best_fid.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--categorical", action="store_true")
    cli = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    grids = json.load(open(cli.grid))
    net, tr, clip_model = build(cli.vq_ckpt, cli.trans_ckpt, dev)

    for g in grids:
        t = g["template"]
        out = Path(cli.out) / str(t)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tokens_log = {"neutral": [], "styled": []}

        Z_S = []
        for prompt in g["neutral"]:
            z, tok = encode_prompt(net, tr, clip_model, prompt, dev, cli.seed, cli.categorical)
            Z_S.append(z); tokens_log["neutral"].append(tok)
        Z_S = np.stack(Z_S)
        print(f"[template {t}] neutral done", flush=True)

        Z_T = []
        for j, row in enumerate(g["styled"]):
            zs, toks = [], []
            for prompt in row:
                z, tok = encode_prompt(net, tr, clip_model, prompt, dev, cli.seed, cli.categorical)
                zs.append(z); toks.append(tok)
            Z_T.append(np.stack(zs)); tokens_log["styled"].append(toks)
            print(f"[template {t}] style '{g['styles'][j]}' done", flush=True)

        np.save(out / "Z_S.npy", Z_S)
        np.save(out / "Z_T.npy", np.stack(Z_T))
        (out / "tokens.json").write_text(json.dumps(tokens_log))
        (out / "meta.json").write_text(json.dumps(
            {"actions": g["actions"], "styles": g["styles"], "template": t,
             "seed": cli.seed, "sampling": "categorical" if cli.categorical else "greedy",
             "vq_ckpt": cli.vq_ckpt, "trans_ckpt": cli.trans_ckpt}))
        print(f"[template {t}] done in {time.time()-t0:.0f}s  "
              f"Z_S {Z_S.shape}  Z_T {np.stack(Z_T).shape}", flush=True)

    print("ALL TEMPLATES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
