"""
Compute stage of engram_overlap, from cached covariances (no sampling).
Loads engram_covs_t2mgpt.pt + T2M-GPT weights, streams per layer.
Fixes the fp16 SVD failure: covariances are symmetrized and pinv uses the
hermitian (eigh) path, which is the numerically correct choice for PSD
matrices.
"""
from __future__ import annotations
import argparse, collections, json, sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3,
          embed_dim_gpt=1024, clip_dim=512, block_size=51, num_layers=9,
          n_head_gpt=16, drop_out_rate=0.1, ff_rate=4)

p = argparse.ArgumentParser()
p.add_argument("--t2mgpt_root", required=True)
p.add_argument("--cache", required=True)
args = p.parse_args()

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self
sys.path.insert(0, args.t2mgpt_root)
import models.t2m_trans as trans
ns = Namespace(**HP)
tr = trans.Text2Motion_Transformer(
    num_vq=ns.nb_code, embed_dim=ns.embed_dim_gpt, clip_dim=ns.clip_dim,
    block_size=ns.block_size, num_layers=ns.num_layers, n_head=ns.n_head_gpt,
    drop_out_rate=ns.drop_out_rate, fc_rate=ns.ff_rate)
tr.load_state_dict(torch.load(Path(args.t2mgpt_root) / "pretrained/VQTransformer_corruption05/net_best_fid.pth",
                   map_location="cpu", weights_only=False)["trans"], strict=True)
tr.eval()
import torch.nn as nn
weights = {n: m.weight.detach().float() for n, m in tr.named_modules()
           if isinstance(m, nn.Linear) and m.in_features <= 1024}

data = torch.load(args.cache, map_location="cpu", weights_only=False)
buckets, counts = data["buckets"], data["counts"]

styles = sorted({b.split("|")[0] for b in buckets} - {"neutral"})
groups = styles + ["neutral"]
keys = [(s, q) for s in groups for q in (0, 1)]
layer_names = [n for n in weights if all(n in buckets[f"{s}|{q}"] for s, q in keys)]
print(f"{len(styles)} styles, {len(layer_names)} layers")

def mean_cov(bks, name):
    tot, cnt = None, 0
    for bk in bks:
        c = buckets[bk][name].float()
        tot = c if tot is None else tot + c
        cnt += counts[bk][name]
    return tot / cnt

all_bks = [f"{s}|{q}" for s, q in keys]
num = collections.defaultdict(float)
sq = collections.defaultdict(float)

for li, name in enumerate(layer_names):
    C_tot = mean_cov(all_bks, name)
    C_tot = 0.5 * (C_tot + C_tot.T)                      # symmetrize (fp16 storage)
    rtol = C_tot.shape[-1] * torch.finfo(torch.float32).eps
    pinv_tot = torch.linalg.pinv(C_tot, rtol=rtol, hermitian=True)
    W = weights[name]
    E = {}
    for s, q in keys:
        C_t = mean_cov([f"{s}|{q}"], name)
        C_t = 0.5 * (C_t + C_t.T)
        E[(s, q)] = W @ C_t @ pinv_tot
    for q in (0, 1):
        mean = sum(E[(s, q)] for s in groups) / len(groups)
        for s in groups:
            E[(s, q)] = E[(s, q)] - mean
    for s, q in keys:
        sq[(s, q)] += float((E[(s, q)] ** 2).sum())
    for i, si in enumerate(groups):
        for sj in groups[i:]:
            num[(si, sj)] += float((E[(si, 0)] * E[(sj, 1)]).sum())
            if si != sj:
                num[(sj, si)] += float((E[(sj, 0)] * E[(si, 1)]).sum())
    if (li + 1) % 20 == 0:
        print(f"  layers {li+1}/{len(layer_names)}", flush=True)

def ccos(si, sj):
    return num[(si, sj)] / (np.sqrt(sq[(si, 0)]) * np.sqrt(sq[(sj, 1)]) + 1e-12)

within = {s: round(ccos(s, s), 4) for s in groups}
for s in groups:
    print(f"{s:<14} centered split-half {within[s]:+.4f}", flush=True)
between = [0.5 * (ccos(si, sj) + ccos(sj, si))
           for i, si in enumerate(styles) for sj in styles[i + 1:]]
vs_neutral = [0.5 * (ccos(s, "neutral") + ccos("neutral", s)) for s in styles]

summary = {
    "within_style_mean": round(float(np.mean([within[s] for s in styles])), 4),
    "between_style_mean": round(float(np.mean(between)), 4),
    "between_style_sd": round(float(np.std(between)), 4),
    "style_vs_neutral_mean": round(float(np.mean(vs_neutral)), 4),
    "within_per_style": within, "n_layers": len(layer_names),
}
print(f"\nwithin-style (centered split-half):  {summary['within_style_mean']:+.3f}")
print(f"between-style (centered, cross-half): {summary['between_style_mean']:+.3f} ± {summary['between_style_sd']:.3f}")
print(f"style vs neutral (centered):          {summary['style_vs_neutral_mean']:+.3f}")
out = Path("/data/pmyap24/sac/results/engram_overlap_t2mgpt.json") \
    if Path("/data/pmyap24/sac").exists() else ROOT / "results" / "engram_overlap_t2mgpt.json"
out.write_text(json.dumps(summary, indent=1))
print(f"saved {out}")
