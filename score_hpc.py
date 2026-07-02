"""
Score HPC grid-extraction results (multi-template layout).

Input layout (produced by extract/{mdm,t2mgpt}_extract_grid.py, scp'd home):
  results/{model}/{t}/Z_S.npy   (8, d)
  results/{model}/{t}/Z_T.npy   (7, 8, d)
  results/{model}/{t}/meta.json

For template 0 (the primary grid) this writes the same report.json that
pipeline.py produces, plus K_S/K_T/meta at the model root so make_figures.py
works unchanged. Cross-template numbers go to robustness.json. Style-vector
consistency is computed for every template.

Usage:
  python score_hpc.py --model mdm
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "metrics"))
sys.path.insert(0, str(Path(__file__).parent / "extract"))

from cka import linear_cka, null_cka
from ged import graph_edit_distance, null_ged
from tau_select import tau_selection
from style_vectors import style_vectors, null_consistency
from base import LatentExtractor


def score_template(Z_S, Z_T, styles, n_perms):
    K_S = LatentExtractor.cosine_kernel(Z_S)
    K_T = np.stack([LatentExtractor.cosine_kernel(z) for z in Z_T])
    tau, tau_curve = tau_selection(K_S)

    per_cka, per_cka_p, per_ged, per_ged_p = {}, {}, {}, {}
    for j, s in enumerate(styles):
        c, cp, _ = null_cka(K_S, K_T[j], n_permutations=n_perms)
        g, gp, _ = null_ged(K_S, K_T[j], tau=tau, n_permutations=n_perms)
        per_cka[s], per_cka_p[s] = round(c, 4), round(cp, 4)
        per_ged[s], per_ged_p[s] = round(g, 4), round(gp, 4)

    # BER: styled embedding nearer another action's neutral embedding
    Zs = Z_S / (np.linalg.norm(Z_S, axis=1, keepdims=True) + 1e-8)
    esc = []
    for j in range(Z_T.shape[0]):
        Zt = Z_T[j] / (np.linalg.norm(Z_T[j], axis=1, keepdims=True) + 1e-8)
        esc.append((Zt @ Zs.T).argmax(axis=1) != np.arange(Z_S.shape[0]))
    esc = np.stack(esc)

    cons, cons_p = null_consistency(Z_S, Z_T, n_permutations=n_perms)
    deltas = style_vectors(Z_S, Z_T)
    rel_mag = float(np.linalg.norm(deltas, axis=2).mean() /
                    np.mean([np.linalg.norm(Z_S[a] - Z_S[b])
                             for a in range(len(Z_S)) for b in range(a + 1, len(Z_S))]))

    return {
        "tau": tau, "tau_score_curve": tau_curve,
        "cka_mean": round(float(np.mean(list(per_cka.values()))), 4),
        "ged_mean": round(float(np.mean(list(per_ged.values()))), 4),
        "ber": round(float(esc.mean()), 4),
        "per_style_cka": per_cka, "per_style_cka_p": per_cka_p,
        "per_style_ged": per_ged, "per_style_ged_p": per_ged_p,
        "escape_matrix": esc.astype(int).tolist(),
        "consistency": {s: round(float(cons[j]), 4) for j, s in enumerate(styles)},
        "consistency_p": {s: round(float(cons_p[j]), 4) for j, s in enumerate(styles)},
        "consistency_mean": round(float(cons.mean()), 4),
        "delta_rel_magnitude": round(rel_mag, 4),
    }, K_S, K_T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n_perms", type=int, default=1000)
    args = p.parse_args()

    root = Path(__file__).parent / "results" / args.model
    tdirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])
    if not tdirs:
        sys.exit(f"No template dirs under {root} — scp the HPC results first.")

    all_reports = {}
    for td in tdirs:
        Z_S = np.load(td / "Z_S.npy")
        Z_T = np.load(td / "Z_T.npy")
        meta = json.loads((td / "meta.json").read_text())
        rep, K_S, K_T = score_template(Z_S, Z_T, meta["styles"], args.n_perms)
        all_reports[td.name] = rep
        print(f"template {td.name}: CKA {rep['cka_mean']:.3f}  "
              f"GED {rep['ged_mean']:.3f}  BER {rep['ber']:.3f}  "
              f"consistency {rep['consistency_mean']:.3f}")

        if td.name == "0":   # primary grid -> root artifacts for make_figures
            np.save(root / "K_S.npy", K_S)
            np.save(root / "K_T.npy", K_T)
            (root / "meta.json").write_text(json.dumps(
                {"actions": meta["actions"], "styles": meta["styles"]}))
            (root / "report.json").write_text(json.dumps(
                {"model": args.model, **rep}, indent=2))

    # cross-template summary in robustness.json format
    styles = json.loads((tdirs[0] / "meta.json").read_text())["styles"]
    summary = {}
    for s in styles:
        c = np.array([all_reports[t]["per_style_cka"][s] for t in all_reports])
        g = np.array([all_reports[t]["per_style_ged"][s] for t in all_reports])
        summary[s] = {"cka_mean": round(float(c.mean()), 4),
                      "cka_std": round(float(c.std()), 4),
                      "ged_mean": round(float(g.mean()), 4),
                      "ged_std": round(float(g.std()), 4)}
    (root / "robustness.json").write_text(json.dumps(
        {"per_template": all_reports, "summary": summary}, indent=2))
    print(f"\nWrote {root}/report.json and robustness.json")


if __name__ == "__main__":
    main()
