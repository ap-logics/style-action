"""
Positive control: a non-style semantic contrast at the same extraction sites.

The estimator defence in Section 4.2 shows the method recovers a direction
where one demonstrably exists in CLIP. It does not show the method would find
a direction inside MDM's pooled x0 or T2M-GPT's dequantised codes if one were
there, because pooling, quantisation and ambient anisotropy could each suppress
the statistic on their own. The non-manner adverb grid is a null control and
cannot do this job: it asks what the measure reads when there is nothing to
find.

This grid asks the opposite question. Direction of travel is an attribute these
models must encode to generate correctly at all, and it enters the caption in
the same slot and the same way a manner adverb does. If consistency at the
generator sites is high for direction and near zero for manner, the collapse is
specific to style. If it is near zero for both, the honest claim becomes that
these latents do not support linear arithmetic for any attribute we tested,
which is a different and larger result.

Actions are restricted to those that take a direction of travel naturally, so
that every cell is a grammatical caption rather than a near-nonsense one.

Usage:
  python prompts/grid_positive.py --out grid_positive.json
"""
from __future__ import annotations

import json
from pathlib import Path

# Locomotion actions that accept a heading. The first eight are drawn from
# ACTIONS_V2, the primary grid; the last four are not, so any comparison with
# the manner grid must be restricted to the shared eight (see
# analysis/score_heading_control.py). Scoring all twelve against manner scored
# on twenty-four inflates the ratio.
ACTIONS_DIR: list[str] = [
    "walking", "running", "jogging", "jumping", "turning around",
    "leaning forward", "stepping sideways", "dancing", "crawling",
    "marching", "skipping", "sliding",
]

# direction of travel: non-manner, strongly represented in HumanML3D captions
DIRECTIONS: list[str] = [
    "forwards", "backwards", "to the left", "to the right",
    "in a circle", "in place",
]


def build(out_path: str | Path) -> None:
    neutral = [f"a person is {a}" for a in ACTIONS_DIR]
    styled = [[f"a person is {a} {d}" for a in ACTIONS_DIR] for d in DIRECTIONS]
    grid = [{
        "template": 0,
        "neutral": neutral,
        "styled": styled,
        "actions": neutral,
        "styles": DIRECTIONS,
    }]
    Path(out_path).write_text(json.dumps(grid, indent=1))
    print(f"wrote {out_path}: {len(ACTIONS_DIR)} actions x {len(DIRECTIONS)} "
          f"directions = {len(ACTIONS_DIR) * (len(DIRECTIONS) + 1)} prompts")
    print(f"  e.g. {styled[0][0]!r}")
    print(f"  e.g. {styled[4][7]!r}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="grid_positive.json")
    args = p.parse_args()
    build(args.out)
