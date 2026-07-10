"""
Experiment 1 (Engram localisation): is the style axis present in PARAMETER
space even though it is absent in representation space?

Following Kwon et al. (ICML 2026), the engram of a prompt set is the weight
slice attributable to it:  E = W . C_target . pinv(C_total), with C the mean
input covariance mean(x^T x) collected by forward pre-hooks.

We collect covariances during T2M-GPT's autoregressive sampling for every
(style, action) cell of the grid, then ask the parameter-space analogue of
our consistency question:

  within-style overlap   cos( E_j^{even actions} , E_j^{odd actions} )
      does modifier j leave the same weight signature whichever actions
      carried it?  (split-half, so overlap is not trivially self-similarity)
  between-style overlap  cos( E_j , E_k ),  j != k
      baseline: how similar are engrams of different modifiers?

If within >> between, style has a shared, action-general trace in the
weights -- the axis exists in parameter space and the representation map
scrambles it. If within ~ between, entanglement goes all the way down.

Usage:
  python analysis/engram_overlap.py --t2mgpt_root /path/to/T2M-GPT \
      --grid /path/to/grid_v2_all_templates.json [--n_actions 24]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3,
          embed_dim_gpt=1024, clip_dim=512, block_size=51, num_layers=9,
          n_head_gpt=16, drop_out_rate=0.1, ff_rate=4)
MAX_IN = 1024   # hook only Linears with in_features <= this (memory bound)


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


class CovCollector:
    """Accumulates mean input covariance per hooked layer into named buckets.

    Accumulation happens on the model device (fast); the running bucket is
    flushed to CPU RAM only when set_active() switches buckets, so the
    GPU->CPU transfer cost is once per prompt, not once per forward step.
    """

    def __init__(self, model: nn.Module):
        self.layers: dict[str, nn.Linear] = {}
        self.buckets: dict[str, dict[str, torch.Tensor]] = {}   # CPU store
        self.counts: dict[str, dict[str, int]] = {}
        self._dev_acc: dict[str, torch.Tensor] = {}             # device accumulators
        self._dev_cnt: dict[str, int] = {}
        self.active: str | None = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.in_features <= MAX_IN:
                self.layers[name] = mod
                mod.register_forward_pre_hook(self._make_hook(name))

    def set_active(self, bucket: str | None):
        if self._dev_acc:                                        # flush previous
            b = self.buckets.setdefault(self.active, {})
            n = self.counts.setdefault(self.active, {})
            for name, c in self._dev_acc.items():
                cc = c.cpu().half()          # fp16 store halves bucket RAM
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
            c = x.T @ x                                          # on device
            if name in self._dev_acc:
                self._dev_acc[name] += c
                self._dev_cnt[name] += x.shape[0]
            else:
                self._dev_acc[name] = c
                self._dev_cnt[name] = x.shape[0]
        return hook

    def mean_cov(self, buckets: list[str], name: str) -> tuple[torch.Tensor, int]:
        tot, cnt = None, 0
        for bk in buckets:
            if bk in self.buckets and name in self.buckets[bk]:
                c = self.buckets[bk][name].float()
                tot = c if tot is None else tot + c
                cnt += self.counts[bk][name]
        return tot / cnt, cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2mgpt_root", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--template", type=int, default=0)
    p.add_argument("--n_actions", type=int, default=24)
    p.add_argument("--n_styles", type=int, default=13)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net, tr, clip_model, clip = build(Path(args.t2mgpt_root), dev)
    coll = CovCollector(tr)
    print(f"hooked {len(coll.layers)} linear layers (in_features <= {MAX_IN})")

    grids = json.load(open(args.grid))
    templates = grids if args.template < 0 else [grids[args.template]]
    styles = templates[0]["styles"][:args.n_styles]
    nA = args.n_actions

    def run_prompt(prompt: str, bucket: str):
        coll.set_active(bucket)
        with torch.no_grad():
            toks = clip.tokenize([prompt], truncate=True).to(dev)
            feat = clip_model.encode_text(toks).float()
            tr.sample(feat, if_categorial=False)
        coll.set_active(None)

    t0 = time.time()
    for g in templates:
        for a in range(nA):
            run_prompt(g["neutral"][a], f"neutral|{a % 2}")
        for j, s in enumerate(styles):
            for a in range(nA):
                run_prompt(g["styled"][j][a], f"{s}|{a % 2}")
        print(f"collected template {g['template']}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- cache covariances so reruns skip the sampling stage ----
    cache = Path("/data/pmyap24/sac/results/engram_covs_t2mgpt.pt") \
        if Path("/data/pmyap24/sac").exists() else ROOT / "results" / "engram_covs_t2mgpt.pt"
    torch.save({"buckets": coll.buckets, "counts": coll.counts}, cache)
    print(f"covariances cached to {cache}", flush=True)

    # ---- engrams, streamed per layer (memory-safe) ----
    # cos(E_a, E_b) over the concatenation of all layers decomposes into
    # per-layer dot products, so we never hold more than one layer's engrams.
    all_buckets = [f"{s}|{p}" for s in styles + ["neutral"] for p in (0, 1)]
    layer_names = list(coll.layers.keys())
    groups = styles + ["neutral"]
    keys = [(s, p) for s in groups for p in (0, 1)]

    import collections
    num = collections.defaultdict(float)      # pair -> running dot
    sq = collections.defaultdict(float)       # key -> running norm^2

    for li, name in enumerate(layer_names):
        C_tot, _ = coll.mean_cov(all_buckets, name)
        rtol = C_tot.shape[-1] * torch.finfo(torch.float32).eps
        pinv_tot = torch.linalg.pinv(C_tot, rtol=rtol)
        W = coll.layers[name].weight.detach().float().cpu()

        E = {}
        for s, p in keys:
            C_t, _ = coll.mean_cov([f"{s}|{p}"], name)
            E[(s, p)] = W @ C_t @ pinv_tot
        # centered deviations within each parity
        for p in (0, 1):
            mean = sum(E[(s, p)] for s in groups) / len(groups)
            for s in groups:
                E[(s, p)] = E[(s, p)] - mean
        for s, p in keys:
            sq[(s, p)] += float((E[(s, p)] ** 2).sum())
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

    between = []
    for i, si in enumerate(styles):
        for sj in styles[i + 1:]:
            between.append(0.5 * (ccos(si, sj) + ccos(sj, si)))
    vs_neutral = [0.5 * (ccos(s, "neutral") + ccos("neutral", s)) for s in styles]

    summary = {
        "within_style_mean": round(float(np.mean([within[s] for s in styles])), 4),
        "between_style_mean": round(float(np.mean(between)), 4),
        "between_style_sd": round(float(np.std(between)), 4),
        "style_vs_neutral_mean": round(float(np.mean(vs_neutral)), 4),
        "within_per_style": within,
        "n_layers": len(layer_names), "n_actions": nA, "template": args.template,
    }
    print(f"\nwithin-style (centered split-half):  {summary['within_style_mean']:+.3f}")
    print(f"between-style (centered, cross-half): {summary['between_style_mean']:+.3f} "
          f"± {summary['between_style_sd']:.3f}")
    print(f"style vs neutral (centered):          {summary['style_vs_neutral_mean']:+.3f}")
    (ROOT / "results" / "engram_overlap_t2mgpt.json").write_text(
        json.dumps(summary, indent=1))
    print("saved results/engram_overlap_t2mgpt.json")


if __name__ == "__main__":
    main()
