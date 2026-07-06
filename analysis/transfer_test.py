"""
Causal style-transfer test in DECODED MOTION SPACE (T2M-GPT, CPU-friendly).

The linear probe showed the supervised style estimate does not transfer in
latent space. This test closes the loop to behaviour: apply the transplanted
style vector to a held-out action's neutral CODE SEQUENCE, decode it through
the VQ-VAE decoder, and ask what motion comes out.

For style j and held-out action b:
  delta_hat_j = mean_{a != b} [Z_T[j,a] - Z_S[a]]        (pooled latents)
  edited codes = dequantize(neutral tokens of b) + delta_hat_j   (broadcast)
  motion_edit  = decoder(edited codes)

Measured in motion-descriptor space (motion features mean-pooled over time):
  action retention   nearest neutral motion to motion_edit is still b
  style alignment    cos( motion_edit - motion_neutral_b ,
                          motion_styled_b - motion_neutral_b )
      does the edit move the MOTION toward the true styled motion?

Oracle rows use delta_j(b) itself (same action, no transplant) as the
upper bound of what latent-space editing could achieve.

Usage:
  python analysis/transfer_test.py --t2mgpt_root /path/to/T2M-GPT
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
        # T2M-GPT hardcodes .cuda() in quantize_cnn; make it a no-op on CPU
        torch.Tensor.cuda = lambda self, *a, **k: self
    sys.path.insert(0, str(repo))
    import models.vqvae as vqvae
    args = Namespace(**HP, quantizer="ema_reset", mu=0.99, dataname="t2m")
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                           args.output_emb_width, args.down_t, args.stride_t,
                           args.width, args.depth, args.dilation_growth_rate)
    sd = torch.load(repo / "pretrained/VQVAE/net_last.pth",
                    map_location="cpu", weights_only=False)["net"]
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def decode_codes(net, codes: torch.Tensor) -> np.ndarray:
    """codes (T', 512) -> motion descriptor (263,) mean-pooled over frames."""
    with torch.no_grad():
        x = codes.unsqueeze(0).permute(0, 2, 1)        # (1, 512, T')
        motion = net.vqvae.decoder(x)                   # (1, 263, T)
    return motion[0].mean(dim=-1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2mgpt_root", required=True)
    p.add_argument("--results", default="t2mgpt")
    p.add_argument("--template", default="0")
    args = p.parse_args()

    repo = Path(args.t2mgpt_root)
    net = build_vqvae(repo)

    base = ROOT / "results" / args.results / args.template
    Z_S = np.load(base / "Z_S.npy")                     # (8, 512)
    Z_T = np.load(base / "Z_T.npy")                     # (7, 8, 512)
    tokens = json.loads((base / "tokens.json").read_text())
    meta = json.loads((base / "meta.json").read_text())
    styles, actions = meta["styles"], meta["actions"]
    n_styles, n_actions = len(styles), len(actions)
    deltas = Z_T - Z_S[None]

    def codes_of(tok_list):
        idx = torch.tensor(tok_list, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            return net.vqvae.quantizer.dequantize(idx)[0]   # (T', 512)

    # reference motions
    m_neutral = np.stack([decode_codes(net, codes_of(tokens["neutral"][a]))
                          for a in range(n_actions)])
    m_styled = np.stack([[decode_codes(net, codes_of(tokens["styled"][j][a]))
                          for a in range(n_actions)]
                         for j in range(n_styles)])

    def nearest_action(desc):
        return int(np.linalg.norm(m_neutral - desc[None], axis=1).argmin())

    def run(delta_source: str):
        retain = np.zeros((n_styles, n_actions), dtype=bool)
        align = np.zeros((n_styles, n_actions))
        for j in range(n_styles):
            for b in range(n_actions):
                if delta_source == "transplant":
                    train = [a for a in range(n_actions) if a != b]
                    d_hat = deltas[j, train].mean(axis=0)
                else:                                   # oracle: own delta
                    d_hat = deltas[j, b]
                codes = codes_of(tokens["neutral"][b])
                edited = codes + torch.tensor(d_hat, dtype=codes.dtype)
                m_edit = decode_codes(net, edited)

                retain[j, b] = nearest_action(m_edit) == b
                v_edit = m_edit - m_neutral[b]
                v_true = m_styled[j, b] - m_neutral[b]
                align[j, b] = (v_edit @ v_true) / (
                    np.linalg.norm(v_edit) * np.linalg.norm(v_true) + 1e-8)
        return retain, align

    out = {}
    for source in ["transplant", "oracle"]:
        retain, align = run(source)
        out[source] = {
            "action_retention": round(float(retain.mean()), 4),
            "style_alignment_mean": round(float(align.mean()), 4),
            "style_alignment_sd": round(float(align.std()), 4),
        }
        print(f"{source:<11} action retention {retain.mean():.3f}   "
              f"style alignment {align.mean():.3f} ± {align.std():.3f}")

    (ROOT / "results" / f"transfer_test_{args.results}.json").write_text(
        json.dumps(out, indent=1))
    print(f"saved results/transfer_test_{args.results}.json")


if __name__ == "__main__":
    main()
