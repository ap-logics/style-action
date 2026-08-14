"""
Compound-style grid: do style operations compose?

The main paper measures one modifier at a time and explicitly declines to test
composition ("style words have no inverses in prompt space and we do not test
their composition"). This grid closes that gap. For every ordered pair of the
thirteen manner modifiers it emits

    "a person is <action> <adverb_j> and <adverb_k>"

alongside the thirteen singles, so singles and compounds come from one
extraction run under identical conditions and nothing depends on cross-run
comparability.

Emitted in the same schema as grid_v2.py, so extract/opentma_extract.py and the
CLIP extractor consume it unchanged. The "styles" entries are labels: a single
modifier for the singles, and "j|k" for the ordered compound.

Both orders are emitted deliberately. Whether the representation is order
sensitive is the question -- if "tiredly and angrily" and "angrily and tiredly"
land in the same place, composition is commutative at the prompt level, which is
a far better-founded test of group-like structure than any commutator computed
on fitted rotations.

Scale: 24 actions x (13 singles + 156 ordered pairs) = 4056 styled prompts plus
24 neutral. Deterministic text encoders only, so no seeds are needed.

Usage:
  python prompts/grid_compound.py --out grid_compound.json
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

from grid_v2 import ACTIONS_V2, STYLES_V2

TEMPLATE = 0  # adverb-final, the template Table 1 reports


def build(out_path: str | Path) -> None:
    styles = list(STYLES_V2)
    neutral = [f"a person is {ing}" for ing, _ in ACTIONS_V2]

    labels: list[str] = []
    styled: list[list[str]] = []

    # singles, identical to grid_v2 template 0
    for s in styles:
        labels.append(s)
        styled.append([f"a person is {ing} {s}" for ing, _ in ACTIONS_V2])

    # ordered pairs, both directions
    for a, b in combinations(styles, 2):
        for j, k in ((a, b), (b, a)):
            labels.append(f"{j}|{k}")
            styled.append([f"a person is {ing} {j} and {k}" for ing, _ in ACTIONS_V2])

    grid = [{
        "template": TEMPLATE,
        "neutral": neutral,
        "styled": styled,
        "actions": neutral,
        "styles": labels,
        "n_singles": len(styles),
    }]
    Path(out_path).write_text(json.dumps(grid, indent=1))
    n_pairs = len(labels) - len(styles)
    print(f"wrote {out_path}: {len(ACTIONS_V2)} actions x "
          f"({len(styles)} singles + {n_pairs} ordered pairs) = "
          f"{len(ACTIONS_V2) * len(labels)} styled prompts")
    print(f"  example single:   {styled[0][0]}")
    print(f"  example compound: {styled[len(styles)][0]}")
    print(f"  reversed:         {styled[len(styles) + 1][0]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="grid_compound.json")
    args = p.parse_args()
    build(args.out)
