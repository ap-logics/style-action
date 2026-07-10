"""
Experiment 2 (Engram ablation): is the parameter-space style trace causally
responsible for the model's response to the modifier?

For a target style j, compute its engram per layer from the cached
covariances, E_j = W C_j pinv(C_total), and edit W <- W - alpha (n/N) E_j
(count-ratio scaling per Kwon et al.). Then re-generate the grid for the
edited model and measure, against the unedited baseline:

  suppression   ||delta_j|| after / before      (target style's displacement;
                                                 <1 means styling weakened)
  specificity   same ratio for OTHER styles     (should stay ~1)
  neutral drift ||z0_after - z0_before|| / ||delta_j before||
                                                 (neutral behaviour preserved)

If suppression tracks the style's trace strength from experiment 1
(strong-trace styles ablate cleanly, weak-trace styles resist), the trace
is causally real, not a correlational artifact.

Usage (cluster, /data/pmyap24/sac):
  python analysis/engram_ablate.py --t2mgpt_root T2M-GPT \
      --cache results/engram_covs_t2mgpt.pt --grid grid_v2_all_templates.json \
      --targets gently carefully tiredly --alphas 0.5 1.0
"""
from __future__ import annotations
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3,
          embed_dim_gpt=1024, clip_dim=512, block_size=51, num_layers=9,
          n_head_gpt=16, drop_out_rate=0.1, ff_rate=4)


def build(repo: Path, dev: str):
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self
    sys.path.insert(0, str(repo))
    import models.vqvae as vqvae
    import models.t2m_trans as trans
    import clip
    args = Namespace(**HP, quantizer="ema_reset", mu=0.99, dataname="t2m")
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                           args.down_t, args.stride_t, args.width, args.depth,
                           args.dilation_growth_rate)
    net.load_state_dict(torch.load(repo / "pretrained/VQVAE/net_last.pth",
                        map_location="cpu", weights_only=False)["net"], strict=True)
    net.eval().to(dev)
    tr = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt, clip_dim=args.clip_dim,
        block_size=args.block_size, num_layers=args.num_layers,
        n_head=args.n_head_gpt, drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
    tr.load_state_dict(torch.load(repo / "pretrained/VQTransformer_corruption05/net_best_fid.pth",
                       map_location="cpu", weights_only=False)["trans"], strict=True)
    tr.eval().to(dev)
    clip_model, _ = clip.load("ViT-B/32", device=dev, jit=False)
    clip_model.eval()
    return net, tr, clip_model, clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2mgpt_root", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--targets", nargs="+", default=["gently", "carefully", "tiredly"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0])
    p.add_argument("--probe_styles", nargs="+",
                   default=["gently", "carefully", "tiredly", "angrily"])
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net, tr, clip_model, clip = build(Path(args.t2mgpt_root), dev)
    data = torch.load(args.cache, map_location="cpu", weights_only=False)
    buckets, counts = data["buckets"], data["counts"]

    g = json.load(open(args.grid))[0]                     # template 0
    styles = g["styles"]
    nA = len(g["actions"])
    probe_idx = {s: styles.index(s) for s in args.probe_styles}

    layers = {n: m for n, m in tr.named_modules()
              if isinstance(m, nn.Linear) and m.in_features <= 1024}
    originals = {n: m.weight.detach().clone() for n, m in layers.items()}

    def mean_cov(bks, name):
        tot, cnt = None, 0
        for bk in bks:
            c = buckets[bk][name].float()
            tot = c if tot is None else tot + c
            cnt += counts[bk][name]
        return tot / cnt, cnt

    all_bks = [f"{s}|{q}" for s in styles + ["neutral"] for q in (0, 1)]

    def encode(prompt):
        with torch.no_grad():
            toks = clip.tokenize([prompt], truncate=True).to(dev)
            feat = clip_model.encode_text(toks).float()
            try:
                idx = tr.sample(feat, if_categorial=False)
            except UnboundLocalError:
                return None                    # model emitted end-token first: collapse
            codes = net.vqvae.quantizer.dequantize(idx)
            return codes[0].mean(dim=0).cpu().numpy()

    def grid_latents():
        collapsed = 0
        def enc(prompt):
            nonlocal collapsed
            z = encode(prompt)
            if z is None:
                collapsed += 1
                return np.zeros(512, dtype=np.float32)
            return z
        Z0 = np.stack([enc(g["neutral"][a]) for a in range(nA)])
        ZT = {s: np.stack([enc(g["styled"][probe_idx[s]][a]) for a in range(nA)])
              for s in args.probe_styles}
        return Z0, ZT, collapsed

    print("baseline generation ...", flush=True)
    Z0_base, ZT_base, _ = grid_latents()
    d_base = {s: ZT_base[s] - Z0_base for s in args.probe_styles}

    results = {}
    for target in args.targets:
        # engram of the target style (both parities)
        for alpha in args.alphas:
            print(f"editing: ablate '{target}' alpha={alpha}", flush=True)
            with torch.no_grad():
                for name, m in layers.items():
                    C_t, n_t = mean_cov([f"{target}|0", f"{target}|1"], name)
                    C_tot, N_t = mean_cov(all_bks, name)
                    C_tot = 0.5 * (C_tot + C_tot.T)
                    rtol = C_tot.shape[-1] * torch.finfo(torch.float32).eps
                    pinv_tot = torch.linalg.pinv(C_tot, rtol=rtol, hermitian=True)
                    E = originals[name].cpu().float() @ (0.5 * (C_t + C_t.T)) @ pinv_tot
                    scale = alpha * (n_t / N_t)
                    m.weight.copy_((originals[name].cpu().float() - scale * E).to(dev))

            Z0_e, ZT_e, n_collapsed = grid_latents()
            entry = {"collapsed_generations": n_collapsed}
            for s in args.probe_styles:
                d_e = ZT_e[s] - Z0_e
                ratio = float(np.linalg.norm(d_e, axis=1).mean() /
                              np.linalg.norm(d_base[s], axis=1).mean())
                entry[s] = round(ratio, 4)
            drift = float(np.linalg.norm(Z0_e - Z0_base, axis=1).mean() /
                          np.linalg.norm(d_base[target], axis=1).mean())
            entry["neutral_drift"] = round(drift, 4)
            results[f"{target}@{alpha}"] = entry
            tgt = entry[target]
            others = np.mean([entry[s] for s in args.probe_styles if s != target])
            print(f"  {target}@{alpha}: target ratio {tgt:.3f}, "
                  f"other styles {others:.3f}, neutral drift {entry['neutral_drift']:.3f}",
                  flush=True)

            with torch.no_grad():                          # restore
                for name, m in layers.items():
                    m.weight.copy_(originals[name].to(dev))

    out = Path("/data/pmyap24/sac/results/engram_ablation_t2mgpt.json") \
        if Path("/data/pmyap24/sac").exists() \
        else Path(__file__).resolve().parents[1] / "results" / "engram_ablation_t2mgpt.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
