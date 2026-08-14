"""
Action Preservation for T2M-GPT, aggregated over seeds.

This regenerates the number reported in the paper (AP = 32.3 +/- 1.5% against a
4.2% chance floor). The existing analysis/action_preservation.py scores a single
(seed, template) run and writes no aggregate, which is why the reported mean and
deviation had no backing file; this script runs the five seeds at the primary
template and writes both the per-seed values and the aggregate.

Method, unchanged from action_preservation.py: decode every neutral and styled
token sequence through T2M-GPT's own VQ decoder, mean-pool over time, build one
prototype per action from the neutral decodes, and assign each styled decode to
its nearest prototype. No external action classifier is involved. Cosine is the
primary rule, matching analysis/mld_ap.py; the Euclidean rule is reported
alongside it because the appendix claims the two agree to within three points.

Needs the T2M-GPT checkpoint, which is not in this repo:
    <t2mgpt_root>/models/vqvae.py
    <t2mgpt_root>/pretrained/VQVAE/net_last.pth

Usage:
    python analysis/ap_t2mgpt_seeds.py --t2mgpt_root /path/to/T2M-GPT

Writes results/action_preservation_t2mgpt.json.
"""
from __future__ import annotations
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SEEDS = [42, 43, 44, 45, 46]
PRIMARY_TEMPLATE = 0          # adverb-final, the template Table 1 reports
HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3)


def build_vqvae(repo: Path):
    """Identical construction to analysis/action_preservation.py."""
    import torch
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


def make_decoder(net):
    """tok (list[int]) -> mean-pooled motion feature vector.

    The index tensor is placed on the same device as the codebook. The original
    action_preservation.py built it on CPU unconditionally, which is fine on a
    CPU-only host (where the .cuda() monkeypatch above is active) but crashes on
    a GPU node with "Expected all tensors to be on the same device" -- which is
    exactly how the first 24-action run died.
    """
    import torch
    dev = net.vqvae.quantizer.codebook.device

    def decode(tok):
        idx = torch.tensor(tok, dtype=torch.long, device=dev).unsqueeze(0)
        with torch.no_grad():
            codes = net.vqvae.quantizer.dequantize(idx)          # (1, T', 512)
            motion = net.vqvae.decoder(codes.permute(0, 2, 1))   # (1, 263, T)
        return motion[0].mean(dim=-1).float().cpu().numpy()
    return decode


def score_run(run_dir: Path, decode) -> dict:
    """AP for one (seed, template) directory, under both assignment rules."""
    tokens = json.loads((run_dir / "tokens.json").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    n_actions, n_styles = len(meta["actions"]), len(meta["styles"])

    proto = np.stack([decode(tokens["neutral"][a]) for a in range(n_actions)])
    proto_n = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    hit_cos = np.zeros((n_styles, n_actions), dtype=bool)
    hit_euc = np.zeros((n_styles, n_actions), dtype=bool)
    for j in range(n_styles):
        for a in range(n_actions):
            m = decode(tokens["styled"][j][a])
            hit_euc[j, a] = int(np.linalg.norm(proto - m[None], axis=1).argmin()) == a
            mn = m / (np.linalg.norm(m) + 1e-8)
            hit_cos[j, a] = int((proto_n @ mn).argmax()) == a

    strip = lambda s: s.replace("a person is ", "")
    return {
        "ap_cos": round(float(hit_cos.mean()), 4),
        "ap_euclidean": round(float(hit_euc.mean()), 4),
        "chance": round(1.0 / n_actions, 4),
        "per_style": {meta["styles"][j]: round(float(hit_cos[j].mean()), 4)
                      for j in range(n_styles)},
        "misses_per_action": {strip(meta["actions"][a]):
                              int((~hit_cos[:, a]).sum()) for a in range(n_actions)},
        "n_styles": n_styles,
        "n_actions": n_actions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2mgpt_root", required=True)
    ap.add_argument("--results", default="t2mgpt_v2")
    ap.add_argument("--template", type=int, default=PRIMARY_TEMPLATE)
    args = ap.parse_args()

    decode = make_decoder(build_vqvae(Path(args.t2mgpt_root)))

    per_seed = {}
    for s in SEEDS:
        run = RES / args.results / f"seed{s}" / str(args.template)
        if not (run / "tokens.json").exists():
            print(f"  seed{s}: SKIP (no tokens.json at {run})")
            continue
        rec = score_run(run, decode)
        rec["seed"] = s
        per_seed[f"seed{s}"] = rec
        print(f"  seed{s}: AP(cos) {rec['ap_cos']:.4f}  "
              f"AP(euclid) {rec['ap_euclidean']:.4f}  chance {rec['chance']:.4f}")

    if not per_seed:
        sys.exit("no runs scored; check --results / --template")

    cos = np.array([r["ap_cos"] for r in per_seed.values()])
    euc = np.array([r["ap_euclidean"] for r in per_seed.values()])
    sd = lambda v: round(float(v.std(ddof=1)), 4) if len(v) > 1 else None
    out = {
        "summary": {
            "ap_cos_mean": round(float(cos.mean()), 4), "ap_cos_sd": sd(cos),
            "ap_euclidean_mean": round(float(euc.mean()), 4), "ap_euclidean_sd": sd(euc),
            "chance": next(iter(per_seed.values()))["chance"],
            "n_seeds": len(per_seed), "template": args.template,
            "rule": "cosine nearest-prototype (primary); euclidean reported as a check",
        },
        "per_seed": per_seed,
    }
    (RES / "action_preservation_t2mgpt.json").write_text(json.dumps(out, indent=1))
    m, d = out["summary"]["ap_cos_mean"], out["summary"]["ap_cos_sd"]
    print(f"\nAP = {100*m:.1f} +/- {100*(d or 0):.1f}%  "
          f"(chance {100*out['summary']['chance']:.1f}%), n={len(per_seed)} seeds")
    print("saved results/action_preservation_t2mgpt.json")


if __name__ == "__main__":
    main()
