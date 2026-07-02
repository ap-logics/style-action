"""
DiagnosticPipeline — orchestrates the full style-action coupling evaluation.

Kernel layout (matches the paper's Formalisation section):
  K_S      : (8, 8)     cosine similarities among the 8 neutral action embeddings
  K_T[j]   : (7, 8, 8)  same actions conditioned on style j — one kernel per style
  CKA/GED are computed per style, reported as per-style profiles + means.

Usage:
  # Stage 1: extract latents and build kernel matrices
  python pipeline.py --model mdm --stage extract
  python pipeline.py --model clip --stage extract   # text-encoder control

  # Stage 2: generate motions for AP (run as SLURM array, prompt_idx 0-63)
  python pipeline.py --model mdm --stage generate --prompt_idx 3

  # Stage 3: score everything (AP, BER, CKA, GED, nulls)
  python pipeline.py --model mdm --stage score

  # Debug: 2 styles only, CPU-friendly
  python pipeline.py --model clip --stage extract --debug

Failure-mode taxonomy reported in stage 3:
  Type A — Geometric coupling, AP preserved:
      BER high for the action, but AP stays high.
      Decoder compensates or classifier insensitive.
  Type B — Geometric coupling, AP drops:
      BER high for the action, and AP falls.
      Coupling propagates all the way to observable action identity.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# repo-local imports
sys.path.insert(0, str(Path(__file__).parent / "prompts"))
sys.path.insert(0, str(Path(__file__).parent / "metrics"))
sys.path.insert(0, str(Path(__file__).parent / "extract"))
sys.path.insert(0, str(Path(__file__).parent / "classifier"))

from grid import build_grid, flat_prompts, ACTIONS, STYLES
from cka import linear_cka, null_cka
from ged import graph_edit_distance, null_ged
from tau_select import tau_selection
from ap import action_preservation
from base import LatentExtractor


@dataclass
class DiagnosticReport:
    model: str
    tau: float
    cka_mean: float
    ged_mean: float
    ap: float
    ber: float
    n_actions: int
    n_styles: int
    tau_score_curve: dict = field(default_factory=dict)
    # per-style profiles (style_name -> value)
    per_style_cka:   dict = field(default_factory=dict)
    per_style_cka_p: dict = field(default_factory=dict)
    per_style_ged:   dict = field(default_factory=dict)
    per_style_ged_p: dict = field(default_factory=dict)
    # per-action breakdown (action_name -> value)
    per_action_ap:  dict = field(default_factory=dict)
    per_action_ber: dict = field(default_factory=dict)
    # failure-mode taxonomy
    type_a_actions: list = field(default_factory=list)   # coupling, AP preserved
    type_b_actions: list = field(default_factory=list)   # coupling, AP drops


def _results_dir(model: str) -> Path:
    d = Path(__file__).parent / "results" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_extractor(model: str) -> LatentExtractor:
    if model == "mdm":
        from mdm import MDMExtractor
        return MDMExtractor()
    elif model == "t2mgpt":
        from t2mgpt import T2MGPTExtractor
        return T2MGPTExtractor()
    elif model == "clip":
        from clip_control import CLIPControlExtractor
        return CLIPControlExtractor()
    else:
        raise ValueError(f"Unknown model: {model}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: extract
# ─────────────────────────────────────────────────────────────────────────────

def stage_extract(model: str, debug: bool = False):
    neutral, styled, actions, styles = build_grid()

    if debug:
        styled = styled[:2]
        styles = styles[:2]
        print(f"[debug] running 2 styles: {styles}")

    extractor = _load_extractor(model)

    print(f"Encoding {len(neutral)} neutral prompts …")
    Z_S = extractor.encode(neutral)                    # (8, d)
    K_S = LatentExtractor.cosine_kernel(Z_S)           # (8, 8)

    Z_T, K_T = [], []
    for j, style_row in enumerate(styled):
        print(f"Encoding style '{styles[j]}' ({len(style_row)} prompts) …")
        z = extractor.encode(style_row)                # (8, d)
        Z_T.append(z)
        K_T.append(LatentExtractor.cosine_kernel(z))
    Z_T = np.stack(Z_T)                                # (n_styles, 8, d)
    K_T = np.stack(K_T)                                # (n_styles, 8, 8)

    out = _results_dir(model)
    suffix = "_debug" if debug else ""
    np.save(out / f"Z_S{suffix}.npy", Z_S)
    np.save(out / f"Z_T{suffix}.npy", Z_T)
    np.save(out / f"K_S{suffix}.npy", K_S)
    np.save(out / f"K_T{suffix}.npy", K_T)
    (out / f"meta{suffix}.json").write_text(json.dumps(
        {"actions": actions, "styles": styles}
    ))
    print(f"Saved to {out}/  Z_S {Z_S.shape}, Z_T {Z_T.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: generate (one prompt index over the flat 64-prompt list)
# ─────────────────────────────────────────────────────────────────────────────

def stage_generate(model: str, prompt_idx: int, n_seeds: int = 5):
    prompts, labels = flat_prompts()
    prompt = prompts[prompt_idx]
    label = labels[prompt_idx]

    extractor = _load_extractor(model)
    motions = extractor.generate_from_prompt(prompt, n_seeds=n_seeds)

    out = _results_dir(model) / "motions"
    out.mkdir(exist_ok=True)
    np.save(out / f"prompt_{prompt_idx:03d}.npy", motions)
    (out / f"prompt_{prompt_idx:03d}_meta.json").write_text(
        json.dumps({"prompt": prompt, "label": label, "prompt_idx": prompt_idx,
                    "is_neutral": prompt_idx < len(ACTIONS)})
    )
    print(f"[{prompt_idx:03d}] '{prompt}' → {motions.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: score
# ─────────────────────────────────────────────────────────────────────────────

def _basin_escape(Z_S: np.ndarray, Z_T: np.ndarray) -> tuple[float, dict]:
    """
    BER with one neutral embedding per action: a styled embedding z_j(a) has
    escaped when its nearest neutral embedding (cosine) is some action b != a.
    Returns (global_ber, per_action_ber keyed by action name).
    """
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    n_styles, n_actions, _ = Z_T.shape
    escaped = np.zeros((n_styles, n_actions), dtype=bool)
    for j in range(n_styles):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        sims = Zt @ Zs.T                       # (8, 8): styled x neutral
        escaped[j] = sims.argmax(axis=1) != np.arange(n_actions)
    per_action = {
        ACTIONS[a]: round(float(escaped[:, a].mean()), 4)
        for a in range(n_actions)
    }
    return float(escaped.mean()), per_action


def _failure_taxonomy(
    per_action_ber: dict,
    per_action_ap: dict,
    ber_threshold: float = 0.3,
    ap_threshold: float = 0.6,
) -> tuple[list, list]:
    """
    Type A: geometric coupling (BER high) but AP preserved.
    Type B: geometric coupling AND AP drops — observable failure.
    """
    type_a, type_b = [], []
    for action, ber in per_action_ber.items():
        if ber <= ber_threshold:
            continue
        if action in per_action_ap and per_action_ap[action] < ap_threshold:
            type_b.append(action)
        else:
            type_a.append(action)
    return type_a, type_b


def stage_score(model: str, n_permutations: int = 1000):
    out = _results_dir(model)

    K_S = np.load(out / "K_S.npy")            # (8, 8)
    K_T = np.load(out / "K_T.npy")            # (n_styles, 8, 8)
    Z_S = np.load(out / "Z_S.npy")
    Z_T = np.load(out / "Z_T.npy")
    meta = json.loads((out / "meta.json").read_text())
    styles = meta["styles"]

    # tau selection on the neutral graph
    best_tau, tau_curve = tau_selection(K_S)
    print(f"Selected tau={best_tau} (score curve: {tau_curve})")

    # per-style CKA + GED with permutation nulls
    per_cka, per_cka_p, per_ged, per_ged_p = {}, {}, {}, {}
    for j, style in enumerate(styles):
        c, cp, _ = null_cka(K_S, K_T[j], n_permutations=n_permutations)
        g, gp, _ = null_ged(K_S, K_T[j], tau=best_tau, n_permutations=n_permutations)
        per_cka[style],   per_cka_p[style] = round(c, 4), round(cp, 4)
        per_ged[style],   per_ged_p[style] = round(g, 4), round(gp, 4)
        print(f"  {style:<12} CKA={c:.4f} (p={cp:.3f})  GED={g:.4f} (p={gp:.3f})")

    cka_mean = float(np.mean(list(per_cka.values())))
    ged_mean = float(np.mean(list(per_ged.values())))

    # basin escape
    ber_score, per_action_ber = _basin_escape(Z_S, Z_T)
    print(f"\nmean CKA={cka_mean:.4f}  mean GED={ged_mean:.4f}  BER={ber_score:.4f}")

    # AP (needs motions + classifier)
    ap_score = float("nan")
    per_action_ap: dict = {}
    motions_dir = out / "motions"

    if motions_dir.exists():
        from humanml3d import HumanML3DClassifier
        clf = HumanML3DClassifier()

        action_motions: dict[int, list] = {ai: [] for ai in range(len(ACTIONS))}
        all_motions, all_labels = [], []

        for meta_path in sorted(motions_dir.glob("*_meta.json")):
            m = json.loads(meta_path.read_text())
            if m.get("is_neutral"):
                continue   # AP is measured on styled outputs only
            motion_path = motions_dir / f"prompt_{m['prompt_idx']:03d}.npy"
            if not motion_path.exists():
                continue
            for motion in np.load(motion_path):
                all_motions.append(motion)
                all_labels.append(m["label"])
                action_motions[m["label"]].append(motion)

        if all_motions:
            ap_score, _ = action_preservation(np.stack(all_motions), all_labels, clf)
            print(f"AP={ap_score:.4f}")
            for ai, action_name in enumerate(ACTIONS):
                if action_motions[ai]:
                    arr = np.stack(action_motions[ai])
                    pa_ap, _ = action_preservation(arr, [ai] * len(arr), clf)
                    per_action_ap[action_name] = round(float(pa_ap), 4)
    else:
        print("No motions found — skipping AP (run stage=generate first)")

    type_a, type_b = _failure_taxonomy(per_action_ber, per_action_ap)

    print("\nPer-action breakdown:")
    for a in ACTIONS:
        ap_str = f"{per_action_ap[a]:.3f}" if a in per_action_ap else "n/a"
        tag = " [Type B]" if a in type_b else (" [Type A]" if a in type_a else "")
        print(f"  {a:<32} BER={per_action_ber[a]:.3f}  AP={ap_str}{tag}")

    report = DiagnosticReport(
        model=model,
        tau=best_tau,
        cka_mean=round(cka_mean, 4),
        ged_mean=round(ged_mean, 4),
        ap=ap_score,
        ber=round(ber_score, 4),
        n_actions=len(ACTIONS),
        n_styles=len(styles),
        tau_score_curve=tau_curve,
        per_style_cka=per_cka,
        per_style_cka_p=per_cka_p,
        per_style_ged=per_ged,
        per_style_ged_p=per_ged_p,
        per_action_ap=per_action_ap,
        per_action_ber=per_action_ber,
        type_a_actions=type_a,
        type_b_actions=type_b,
    )
    report_path = out / "report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2))
    print(f"\nReport saved to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Style-Action Coupling Diagnostic Pipeline")
    p.add_argument("--model",      required=True, choices=["mdm", "t2mgpt", "clip"])
    p.add_argument("--stage",      required=True, choices=["extract", "generate", "score"])
    p.add_argument("--prompt_idx", type=int, default=0, help="Prompt index (stage=generate)")
    p.add_argument("--n_seeds",    type=int, default=5)
    p.add_argument("--n_perms",    type=int, default=1000)
    p.add_argument("--debug",      action="store_true", help="2 styles only, CPU-friendly")
    args = p.parse_args()

    if args.stage == "extract":
        stage_extract(args.model, debug=args.debug)
    elif args.stage == "generate":
        stage_generate(args.model, args.prompt_idx, n_seeds=args.n_seeds)
    elif args.stage == "score":
        stage_score(args.model, n_permutations=args.n_perms)


if __name__ == "__main__":
    main()
