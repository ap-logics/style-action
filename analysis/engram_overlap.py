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
    """Accumulates mean input covariance per hooked layer into named buckets."""

    def __init__(self, model: nn.Module):
        self.layers: dict[str, nn.Linear] = {}
        self.buckets: dict[str, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.active: str | None = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.in_features <= MAX_IN:
                self.layers[name] = mod
                mod.register_forward_pre_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(_mod, inputs):
            if self.active is None:
                return
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()      # (tokens, D)
            c = (x.T @ x).cpu()                          # accumulate on CPU RAM
            b = self.buckets.setdefault(self.active, {})
            n = self.counts.setdefault(self.active, {})
            if name in b:
                b[name] += c
                n[name] += x.shape[0]
            else:
                b[name] = c
                n[name] = x.shape[0]
        return hook

    def mean_cov(self, buckets: list[str], name: str) -> tuple[torch.Tensor, int]:
        tot, cnt = None, 0
        for bk in buckets:
            if bk in self.buckets and name in self.buckets[bk]:
                tot = self.buckets[bk][name] if tot is None else tot + self.buckets[bk][name]
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
        coll.active = bucket
        with torch.no_grad():
            toks = clip.tokenize([prompt], truncate=True).to(dev)
            feat = clip_model.encode_text(toks).float()
            tr.sample(feat, if_categorial=False)
        coll.active = None

    t0 = time.time()
    for g in templates:
        for a in range(nA):
            run_prompt(g["neutral"][a], f"neutral|{a % 2}")
        for j, s in enumerate(styles):
            for a in range(nA):
                run_prompt(g["styled"][j][a], f"{s}|{a % 2}")
        print(f"collected template {g['template']}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- engrams ----
    all_buckets = [f"{s}|{p}" for s in styles + ["neutral"] for p in (0, 1)]
    layer_names = list(coll.layers.keys())

    def engram(buckets: list[str]) -> dict[str, torch.Tensor]:
        out = {}
        for name in layer_names:
            C_t, _ = coll.mean_cov(buckets, name)
            W = coll.layers[name].weight.detach().float().cpu()
            out[name] = W @ C_t @ pinv_total[name]
        return out

    print("computing pinv of total covariance per layer ...", flush=True)
    pinv_total = {}
    for name in layer_names:
        C_tot, _ = coll.mean_cov(all_buckets, name)
        rtol = C_tot.shape[-1] * torch.finfo(torch.float32).eps
        pinv_total[name] = torch.linalg.pinv(C_tot, rtol=rtol)

    def cos(Ea, Eb):
        num = sum((Ea[n] * Eb[n]).sum() for n in layer_names)
        da = sum((Ea[n] ** 2).sum() for n in layer_names).sqrt()
        db = sum((Eb[n] ** 2).sum() for n in layer_names).sqrt()
        return float(num / (da * db + 1e-12))

    # engrams per (style, parity) and full; centered deviations for overlap.
    # Raw engrams share a large W-shaped component and, because disjoint
    # targets' engrams sum to ~W, their deviations anticorrelate by
    # construction. Centering by the grand mean within each parity removes
    # both artifacts symmetrically for within- and between-style comparisons.
    groups = styles + ["neutral"]
    E_half = {s: {p: engram([f"{s}|{p}"]) for p in (0, 1)} for s in groups}

    def centered(E_dict):
        mean = {n: sum(E_dict[s][n] for s in groups) / len(groups)
                for n in layer_names}
        return {s: {n: E_dict[s][n] - mean[n] for n in layer_names}
                for s in groups}

    D_even = centered({s: E_half[s][0] for s in groups})
    D_odd = centered({s: E_half[s][1] for s in groups})

    within, raw_within = {}, {}
    for s in groups:
        within[s] = cos(D_even[s], D_odd[s])
        raw_within[s] = cos(E_half[s][0], E_half[s][1])
        print(f"{s:<14} centered split-half {within[s]:+.4f}   "
              f"raw {raw_within[s]:+.4f}", flush=True)

    # between-style: centered, cross-parity (j even vs k odd) so the
    # comparison carries the same split-half noise as the within metric
    between = []
    for i, si in enumerate(styles):
        for sj in styles[i + 1:]:
            between.append(0.5 * (cos(D_even[si], D_odd[sj]) +
                                  cos(D_even[sj], D_odd[si])))
    vs_neutral = [0.5 * (cos(D_even[s], D_odd["neutral"]) +
                         cos(D_even["neutral"], D_odd[s])) for s in styles]

    summary = {
        "within_style_mean": round(float(np.mean([within[s] for s in styles])), 4),
        "between_style_mean": round(float(np.mean(between)), 4),
        "between_style_sd": round(float(np.std(between)), 4),
        "style_vs_neutral_mean": round(float(np.mean(vs_neutral)), 4),
        "within_per_style": {s: round(v, 4) for s, v in within.items()},
        "raw_within_per_style": {s: round(v, 4) for s, v in raw_within.items()},
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
