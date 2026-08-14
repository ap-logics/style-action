"""
Is style linearly decodable from the latent the decoder consumes?

Consistency asks whether the style displacements share a direction. This asks a
different question: can a linear classifier read the style off the styled
embedding at all? The two can come apart. Information can be linearly present
while conditioning fails to implement it as a shared translation.

Splits are leave-one-action-out, so the held-out action is never seen in
training and the classifier cannot succeed by memorising action identity. It
has to use whatever is style-general in the representation.

Two feature sets:
  raw   z_j(a)                 the representation as the decoder receives it
  delta z_j(a) - z_0(a)        the displacement, directly comparable to consistency

Chance is 1/13 = 0.077.

Usage:  python3 style_decodability.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]

SYSTEMS = [
    ("CLIP (input)",        "clip_v2"),
    ("TMA, HumanML3D",      "v2_tma_tmp/opentma_h3d"),
    ("TMA, Motion-X",       "v2_tma_tmp/opentma_motionx"),
    ("TMA, UniMoCap",       "v2_tma_tmp/opentma_unimocap"),
    ("MLD",                 "mld_v2"),
    ("MDM",                 "mdm_v2"),
    ("T2M-GPT",             "t2mgpt_v2"),
    ("MoMask",              "momask_v2"),
]


def load_runs(sub: str):
    """Every (seed x template) run for one system, as (Z_S, Z_T) pairs."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / sub / "**" / "Z_T.npy"), recursive=True)):
        d = Path(f).parent
        if not (d / "Z_S.npy").exists():
            continue
        out.append((np.load(d / "Z_S.npy"), np.load(d / "Z_T.npy")))
    return out


def leave_one_action_out(X, y, act):
    """X (n,d), y style label, act action label. Train on 23 actions, test on the 24th."""
    accs = []
    for held in np.unique(act):
        tr, te = act != held, act == held
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def build(runs, mode: str):
    X, y, act = [], [], []
    for Z_S, Z_T in runs:
        n_styles, n_actions, _ = Z_T.shape
        for j in range(n_styles):
            for a in range(n_actions):
                if mode == "raw":
                    v = Z_T[j, a]
                else:
                    v = Z_T[j, a] - Z_S[a]
                    if mode == "delta_unit":          # direction only, magnitude removed
                        v = v / (np.linalg.norm(v) + 1e-8)
                X.append(v)
                y.append(j); act.append(a)
    return np.stack(X), np.array(y), np.array(act)


def main():
    results = {}
    print(f"{'system':<18}{'raw':>10}{'delta':>11}{'delta (unit)':>13}{'chance':>9}")
    for label, sub in SYSTEMS:
        runs = load_runs(sub)
        if not runs:
            print(f"{label:<18}  (no data)"); continue
        n_styles = runs[0][1].shape[0]
        chance = 1.0 / n_styles
        row = {"n_runs": len(runs), "chance": chance}
        for mode in ("raw", "delta", "delta_unit"):
            X, y, act = build(runs, mode)
            m, s = leave_one_action_out(X, y, act)
            row[mode] = {"acc": m, "sd_over_folds": s}
        results[label] = row
        print(f"{label:<18}{row['raw']['acc']:>10.3f}{row['delta']['acc']:>11.3f}"
              f"{row['delta_unit']['acc']:>13.3f}{chance:>9.3f}")

    (ROOT / "results" / "style_decodability.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {ROOT}/results/style_decodability.json")


if __name__ == "__main__":
    main()
