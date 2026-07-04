"""
Aggregate multi-seed grid extractions into seed-level means and deviations.

Input layout (produced by the v2 sbatch jobs):
  results/{model}/seed{s}/{t}/Z_S.npy, Z_T.npy, meta.json
e.g. results/mdm_v2/seed42/0/..., results/t2mgpt_v2/seed43/1/...
(t2mgpt_v2 also has a greedy/ reference dir, scored separately if present.)

For each seed x template: mean CKA, mean GED (at the per-run selected tau),
BER, mean consistency. Reported per metric as mean +/- sd over seeds
(primary template and pooled), written to results/{model}/seed_summary.json.

Usage:
  python score_seeds.py --model mdm_v2
  python score_seeds.py --model t2mgpt_v2
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "metrics"))
sys.path.insert(0, str(Path(__file__).parent / "extract"))

from cka import linear_cka
from ged import graph_edit_distance
from tau_select import tau_selection
from style_vectors import style_vectors, consistency
from base import LatentExtractor


def score_run(run_dir: Path) -> dict:
    """One seed x template directory -> scalar metrics."""
    Z_S = np.load(run_dir / "Z_S.npy")
    Z_T = np.load(run_dir / "Z_T.npy")
    K_S = LatentExtractor.cosine_kernel(Z_S)
    tau, _ = tau_selection(K_S)

    ckas, geds = [], []
    for z in Z_T:
        K_T = LatentExtractor.cosine_kernel(z)
        ckas.append(linear_cka(K_S, K_T))
        geds.append(graph_edit_distance(K_S, K_T, tau))

    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    esc = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        esc.append((Zt @ Zs.T).argmax(axis=1) != np.arange(Z_S.shape[0]))
    cons, _ = consistency(style_vectors(Z_S, Z_T))

    return {"cka": float(np.mean(ckas)), "ged": float(np.mean(geds)),
            "ber": float(np.stack(esc).mean()),
            "consistency": float(cons.mean()), "tau": tau,
            "n_actions": Z_S.shape[0], "n_styles": Z_T.shape[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()

    root = Path(__file__).parent / "results" / args.model
    seed_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and re.match(r"seed\d+$", d.name))
    if not seed_dirs:
        sys.exit(f"No seed dirs under {root} — pull the HPC results first.")

    all_runs: dict[str, dict[str, dict]] = {}   # seed -> template -> metrics
    for sd in seed_dirs:
        all_runs[sd.name] = {}
        for td in sorted(d for d in sd.iterdir()
                         if d.is_dir() and d.name.isdigit()):
            all_runs[sd.name][td.name] = score_run(td)
            m = all_runs[sd.name][td.name]
            print(f"{sd.name} t{td.name}: CKA {m['cka']:.3f}  "
                  f"GED {m['ged']:.3f}  BER {m['ber']:.3f}  "
                  f"cons {m['consistency']:.3f}")

    templates = sorted({t for runs in all_runs.values() for t in runs})
    summary = {}
    for metric in ["cka", "ged", "ber", "consistency"]:
        primary = [all_runs[s]["0"][metric] for s in all_runs if "0" in all_runs[s]]
        pooled = [all_runs[s][t][metric] for s in all_runs for t in all_runs[s]]
        summary[metric] = {
            "primary_template_mean": round(float(np.mean(primary)), 4),
            "primary_template_sd": round(float(np.std(primary)), 4),
            "pooled_mean": round(float(np.mean(pooled)), 4),
            "pooled_sd": round(float(np.std(pooled)), 4),
            "n_seeds": len(all_runs), "n_templates": len(templates),
        }

    greedy = root / "greedy"
    if greedy.exists():
        summary["greedy_reference"] = {
            t.name: {k: round(v, 4) for k, v in score_run(t).items()
                     if isinstance(v, float)}
            for t in sorted(greedy.iterdir()) if t.is_dir() and t.name.isdigit()}

    (root / "seed_summary.json").write_text(json.dumps(
        {"per_run": all_runs, "summary": summary}, indent=1))
    print(f"\n{args.model} (primary template, mean ± sd over "
          f"{len(all_runs)} seeds):")
    for metric in ["cka", "ged", "ber", "consistency"]:
        s = summary[metric]
        print(f"  {metric:<12} {s['primary_template_mean']:.3f} "
              f"± {s['primary_template_sd']:.3f}")
    print(f"saved {root}/seed_summary.json")


if __name__ == "__main__":
    main()
