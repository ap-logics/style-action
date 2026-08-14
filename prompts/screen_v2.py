"""
Anisotropy-corrected near-synonymy screen for the v2 grid, reproducing the
numbers quoted in grid_v2.py / the paper's Appendix A (centred-cosine
correlated pairs). CLIP embeddings are centred (common component removed)
before cosines, because raw CLIP word cosines all sit in ~[0.85, 0.95].

Run wherever CLIP is available (cluster):
  python prompts/screen_v2.py --grid grid_v2_all_templates.json
Flags style pairs with centred cosine >= --style_thresh and (action, style)
cells with |centred cosine| >= --cell_thresh.
"""
import argparse, itertools, json
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid_v2_all_templates.json")
    ap.add_argument("--style_thresh", type=float, default=0.45)
    ap.add_argument("--cell_thresh", type=float, default=0.5)
    args = ap.parse_args()

    import clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=dev, jit=False)
    model.eval()

    g = json.load(open(args.grid))[0]
    styles = g["styles"]
    actions = [a.replace("a person is ", "") for a in g["actions"]]

    def emb(words):
        with torch.no_grad():
            tok = clip.tokenize(words).to(dev)
            e = model.encode_text(tok).float().cpu().numpy()
        return e

    E_s, E_a = emb(styles), emb(actions)
    allE = np.vstack([E_s, E_a])
    centred_s = E_s - allE.mean(0)
    centred_a = E_a - allE.mean(0)
    ns = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S, A = ns(centred_s), ns(centred_a)

    print("== style-style centred cosines (flag >= %.2f)" % args.style_thresh)
    flagged = []
    for i, j in itertools.combinations(range(len(styles)), 2):
        c = float(S[i] @ S[j])
        if c >= args.style_thresh:
            flagged.append((styles[i], styles[j], round(c, 3)))
            print(f"  FLAG {styles[i]} ~ {styles[j]}: {c:.3f}")
    print("== action-style centred cosines (flag |c| >= %.2f)" % args.cell_thresh)
    cells = []
    for ai, a in enumerate(actions):
        for si, s in enumerate(styles):
            c = float(A[ai] @ S[si])
            if abs(c) >= args.cell_thresh:
                cells.append((a, s, round(c, 3)))
                print(f"  FLAG ({a}, {s}): {c:.3f}")
    out = {"flagged_style_pairs": flagged, "flagged_cells": cells,
           "style_thresh": args.style_thresh, "cell_thresh": args.cell_thresh}
    json.dump(out, open("results/screen_v2.json", "w"), indent=1)
    print("saved results/screen_v2.json")


if __name__ == "__main__":
    main()
