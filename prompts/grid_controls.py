"""
Two control grids that answer objections to the main grid, both text-side.

NULL grid (the length/syntax floor).
  Appending a style adverb changes token count and sentence structure, so some
  part of the input's consistency could be a generic "this prompt is longer"
  component shared by every modifier rather than anything about manner. The
  control swaps the manner adverb for a non-manner adverb in the same slot:
  time, place and discourse adverbs, thirteen of them to match the thirteen
  modifiers. Whatever consistency this grid posts is the floor attributable to
  appending an adverb at all, and is the number the input should be read
  against instead of zero.

  Template 0 only (adverb-final). Template 1 would give "a person is today
  walking" and template 2 "in a today manner", neither of which is English.

PACE grid (the excluded dimension).
  Pace adverbs are excluded from the main grid because "running slowly" drifts
  toward walking, which is the confound under measurement. But pace is also the
  manner dimension most likely to be carried as a single global scalar, and so
  the best candidate for a coherent style direction: excluding it may bias
  against finding one. This grid re-runs the thirteen manner modifiers together
  with five pace adverbs in one extraction, so consistency is directly
  comparable, and the drift that motivated the exclusion is measured by basin
  escape rather than assumed.

Usage:
  python prompts/grid_controls.py --out_dir .
"""
from __future__ import annotations

import json
from pathlib import Path

from grid_v2 import ACTIONS_V2, STYLES_V2, _article

# Non-manner adverbs occupying the same syntactic slot. None describes how the
# action is performed; all are ordinary corpus English.
NULL_ADVERBS = [
    "today", "again", "now", "here", "there", "outside", "inside",
    "afterwards", "meanwhile", "everywhere", "somewhere", "nowadays", "indoors",
]

# Pace adverbs, excluded from the main grid by the pace rule.
PACE_STYLES = {
    "slowly": "slow",
    "quickly": "quick",
    "briskly": "brisk",
    "rapidly": "rapid",
    "swiftly": "swift",
}


def _templates(styles: dict[str, str], which: tuple[int, ...]):
    out = []
    for t in which:
        neutral = [f"a person is {ing}" if t in (0, 1) else f"a person {third}"
                   for ing, third in ACTIONS_V2]
        styled = []
        for s in styles:
            row = []
            for ing, third in ACTIONS_V2:
                if t == 0:
                    row.append(f"a person is {ing} {s}")
                elif t == 1:
                    row.append(f"a person is {s} {ing}")
                else:
                    adj = styles[s]
                    row.append(f"a person {third} in {_article(adj)} {adj} manner")
            styled.append(row)
        out.append({
            "template": t,
            "neutral": neutral,
            "styled": styled,
            "actions": [f"a person is {ing}" for ing, _ in ACTIONS_V2],
            "styles": list(styles),
        })
    return out


def build(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)

    null_styles = {a: a for a in NULL_ADVERBS}          # adjective form unused
    null_grid = _templates(null_styles, (0,))
    (out_dir / "grid_null.json").write_text(json.dumps(null_grid, indent=1))
    print(f"grid_null.json   : {len(ACTIONS_V2)} actions x "
          f"{len(NULL_ADVERBS)} non-manner adverbs x 1 template")
    print(f"  e.g. {null_grid[0]['styled'][0][0]!r}")

    combined = {**STYLES_V2, **PACE_STYLES}
    pace_grid = _templates(combined, (0, 1, 2))
    (out_dir / "grid_pace.json").write_text(json.dumps(pace_grid, indent=1))
    print(f"grid_pace.json   : {len(ACTIONS_V2)} actions x {len(combined)} "
          f"modifiers ({len(STYLES_V2)} manner + {len(PACE_STYLES)} pace) x 3 templates")
    print(f"  e.g. {pace_grid[0]['styled'][-1][0]!r}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()
    build(args.out_dir)
