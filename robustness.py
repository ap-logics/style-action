"""
Paraphrase robustness check: run the per-style CKA analysis under all
paraphrase templates and report mean ± std over templates.

If conclusions hold across templates, the coupling measurement reflects
the model rather than the exact prompt string. Runs on any extractor;
for CLIP this is laptop-friendly.

Usage:
  python robustness.py --model clip
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "prompts"))
sys.path.insert(0, str(Path(__file__).parent / "metrics"))
sys.path.insert(0, str(Path(__file__).parent / "extract"))

from grid import build_grid, TEMPLATES
from cka import linear_cka
from tau_select import tau_selection
from ged import graph_edit_distance
from base import LatentExtractor
from pipeline import _load_extractor, _results_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mdm", "t2mgpt", "clip"])
    args = p.parse_args()

    extractor = _load_extractor(args.model)
    n_templates = len(TEMPLATES)

    # cka[t][style], ged[t][style]
    all_cka: list[dict] = []
    all_ged: list[dict] = []

    for t in range(n_templates):
        neutral, styled, _, styles = build_grid(template=t)
        print(f"\nTemplate {t+1}: e.g. {styled[0][0]!r}")

        Z_S = extractor.encode(neutral)
        K_S = LatentExtractor.cosine_kernel(Z_S)
        tau, _ = tau_selection(K_S)

        cka_t, ged_t = {}, {}
        for j, style in enumerate(styles):
            Z = extractor.encode(styled[j])
            K_T = LatentExtractor.cosine_kernel(Z)
            cka_t[style] = round(float(linear_cka(K_S, K_T)), 4)
            ged_t[style] = round(float(graph_edit_distance(K_S, K_T, tau)), 4)
            print(f"  {style:<12} CKA={cka_t[style]:.4f}  GED={ged_t[style]:.4f}")
        all_cka.append(cka_t)
        all_ged.append(ged_t)

    styles = list(all_cka[0].keys())
    print(f"\n{'style':<12} {'CKA mean±std':<18} {'GED mean±std':<18}")
    summary = {}
    for s in styles:
        c = np.array([all_cka[t][s] for t in range(n_templates)])
        g = np.array([all_ged[t][s] for t in range(n_templates)])
        summary[s] = {
            "cka_mean": round(float(c.mean()), 4), "cka_std": round(float(c.std()), 4),
            "ged_mean": round(float(g.mean()), 4), "ged_std": round(float(g.std()), 4),
        }
        print(f"{s:<12} {c.mean():.4f} ± {c.std():.4f}    {g.mean():.4f} ± {g.std():.4f}")

    grand_c = np.array([[all_cka[t][s] for s in styles] for t in range(n_templates)])
    print(f"\nGrand mean CKA over templates: "
          f"{grand_c.mean(axis=1).round(4).tolist()} (per template)")

    out = _results_dir(args.model)
    (out / "robustness.json").write_text(json.dumps(
        {"per_template_cka": all_cka, "per_template_ged": all_ged,
         "summary": summary}, indent=2))
    print(f"Saved to {out}/robustness.json")


if __name__ == "__main__":
    main()
