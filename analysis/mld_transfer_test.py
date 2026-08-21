"""
Decoded style-transplant test for MLD, the latent-diffusion generator.

Section 4.2 runs this test on T2M-GPT only, which already has consistency
0.004 and the second-worst CKA, so failure there is the expected outcome. MLD
is the system the paper's argument rests on: relationally as clean as the
input, directionally at generator level. Showing that a representation a
similarity diagnostic would certify still cannot be steered is the point.

Conditions, all decoded through MLD's VAE and compared in motion space:

  oracle      z_0(b) + delta_j(b), the action's OWN style vector. For MLD the
              extraction site is the sampled diffusion latent rather than a
              time-pooled descriptor, so this reconstructs the styled latent
              exactly and alignment is 1 by construction. It is reported only
              as a check that the decode path is deterministic and wired up;
              it is NOT an informative positive control here, unlike the
              T2M-GPT case where the edit is broadcast over a pooled sequence.

  cross_seed  z_0(b) + delta_j(b) taken from a DIFFERENT seed. Same provenance,
              same action, but a different draw, so this measures how much of
              a style vector survives sampling noise. It is the honest upper
              bound the transplant should be read against.

  transplant  z_0(b) + mean_{a != b} delta_j(a), the style vector estimated
              from every other action. This is the quantity a steering vector
              would use.

Measured per (style, held-out action):
  action retention   nearest neutral decode to the edited decode is still b
  style alignment    cos( decode(edit) - decode(z_0(b)),
                          decode(z_j(b)) - decode(z_0(b)) )

Run from the MLD repo root:

  python mld_transfer_test.py --cfg configs/config_mld_humanml3d.yaml \
      --cfg_assets configs/assets.yaml \
      --latents $SAC_ROOT/results_v2/mld_v2 \
      --template 0 --seeds 42,43,44,45,46 \
      --out $SAC_ROOT/results/transfer_test_mld.json
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mld.config import parse_args
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def decode_batch(model, Z, device, chunk=24):
    """(n, 256) latents -> (n, 263) motion descriptors, mean-pooled over time."""
    feats = []
    for i in range(0, Z.shape[0], chunk):
        z = torch.tensor(Z[i:i + chunk], dtype=torch.float32, device=device)
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
        feats.append(out.detach().cpu().numpy().mean(axis=1))
    return np.concatenate(feats)


def pop_arg(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    argv.pop(i + 1)
    argv.remove(flag)
    return val


def main():
    argv = sys.argv
    lat_root = Path(pop_arg(argv, "--latents"))
    template = pop_arg(argv, "--template", "0")
    seeds = [s.strip() for s in pop_arg(argv, "--seeds", "42").split(",")]
    out_path = Path(pop_arg(argv, "--out"))

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

    # ---- load every seed's latents for this template
    Zs, Zt = {}, {}
    for s in seeds:
        d = lat_root / f"seed{s}" / template
        Zs[s] = np.load(d / "Z_S.npy")
        Zt[s] = np.load(d / "Z_T.npy")
    meta = json.load(open(lat_root / f"seed{seeds[0]}" / template / "meta.json"))
    styles, actions = meta["styles"], meta["actions"]
    S, A = len(styles), len(actions)
    print(f"{len(seeds)} seeds, {S} styles, {A} actions, latent dim {Zs[seeds[0]].shape[1]}",
          flush=True)

    per_seed = {}
    for si, s in enumerate(seeds):
        Z_S, Z_T = Zs[s], Zt[s]
        deltas = Z_T - Z_S[None]                       # (S, A, 256)
        other = seeds[(si + 1) % len(seeds)]           # a different draw
        deltas_other = Zt[other] - Zs[other][None]

        F_S = decode_batch(model, Z_S, device)         # (A, 263)
        F_T = np.stack([decode_batch(model, Z_T[j], device) for j in range(S)])

        def nearest(desc):
            return int(np.linalg.norm(F_S - desc[None], axis=1).argmin())

        cond_edits = {"oracle": [], "cross_seed": [], "transplant": [],
                      "transplant_scaled": []}
        for j in range(S):
            dsum = deltas[j].sum(axis=0)
            for b in range(A):
                cond_edits["oracle"].append(Z_S[b] + deltas[j, b])
                cond_edits["cross_seed"].append(Z_S[b] + deltas_other[j, b])
                loo = (dsum - deltas[j, b]) / (A - 1)
                cond_edits["transplant"].append(Z_S[b] + loo)
                # Section 3 treats magnitude as a free parameter set at
                # application time, so the fair test takes the transplanted
                # DIRECTION and the true magnitude. Without this the transplant
                # is penalised for being a mean of 23 partly cancelling vectors.
                scaled = loo / (np.linalg.norm(loo) + 1e-12) * np.linalg.norm(deltas[j, b])
                cond_edits["transplant_scaled"].append(Z_S[b] + scaled)

        res = {}
        for cond, edits in cond_edits.items():
            F_E = decode_batch(model, np.stack(edits), device)   # (S*A, 263)
            retain, align, align_other = [], [], []
            k = 0
            for j in range(S):
                for b in range(A):
                    m = F_E[k]; k += 1
                    retain.append(nearest(m) == b)
                    v_e = m - F_S[b]
                    ne = np.linalg.norm(v_e) + 1e-8
                    # alignment to the style actually requested
                    v_t = F_T[j, b] - F_S[b]
                    align.append(float(v_e @ v_t / (ne * np.linalg.norm(v_t) + 1e-8)))
                    # SPECIFICITY: alignment to every OTHER style at the same
                    # action. A generic "styled motion" push scores just as
                    # well here, so only the gap is evidence of style transfer.
                    o = []
                    for kk in range(S):
                        if kk == j:
                            continue
                        v_o = F_T[kk, b] - F_S[b]
                        o.append(float(v_e @ v_o / (ne * np.linalg.norm(v_o) + 1e-8)))
                    align_other.append(float(np.mean(o)))
            gap = float(np.mean(np.array(align) - np.array(align_other)))
            res[cond] = dict(action_retention=round(float(np.mean(retain)), 4),
                             style_alignment_mean=round(float(np.mean(align)), 4),
                             style_alignment_sd=round(float(np.std(align)), 4),
                             alignment_other_styles=round(float(np.mean(align_other)), 4),
                             specificity_gap=round(gap, 4))
            print(f"  seed {s} {cond:<11} retention {np.mean(retain):.3f}  "
                  f"align(own) {np.mean(align):.3f}  align(other) "
                  f"{np.mean(align_other):.3f}  gap {gap:+.3f}", flush=True)
        per_seed[s] = res

    summary = {}
    for cond in ["oracle", "cross_seed", "transplant", "transplant_scaled"]:
        m = [per_seed[s][cond]["style_alignment_mean"] for s in seeds]
        o = [per_seed[s][cond]["alignment_other_styles"] for s in seeds]
        g = [per_seed[s][cond]["specificity_gap"] for s in seeds]
        r = [per_seed[s][cond]["action_retention"] for s in seeds]
        summary[cond] = dict(
            style_alignment_mean=round(float(np.mean(m)), 4),
            style_alignment_sd_across_seeds=round(float(np.std(m)), 4),
            alignment_other_styles=round(float(np.mean(o)), 4),
            specificity_gap=round(float(np.mean(g)), 4),
            specificity_gap_sd_across_seeds=round(float(np.std(g)), 4),
            action_retention_mean=round(float(np.mean(r)), 4))
    out = dict(model="mld", template=template, seeds=seeds,
               n_styles=S, n_actions=A, summary=summary, per_seed=per_seed)
    out_path.write_text(json.dumps(out, indent=1))
    print("\nSUMMARY")
    for cond, v in summary.items():
        print(f"  {cond:<11} own {v['style_alignment_mean']:.3f}  other {v['alignment_other_styles']:.3f}  "
              f"gap {v['specificity_gap']:+.3f} (sd {v['specificity_gap_sd_across_seeds']:.3f})  "
              f"retention {v['action_retention_mean']:.3f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
