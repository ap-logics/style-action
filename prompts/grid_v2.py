"""
Expanded corpus-mined prompt grid (v2): 24 actions x 13 styles.

Every action verb and every modifier below appears in the HumanML3D caption
corpus (see corpus_frequencies.json, mined from the 87k captions), so the
grid is in-distribution by construction. Actions are the most frequent
motion verbs with unambiguous single-person phrasings; modifiers are the
corpus's manner-only adverbs. Pace adverbs (slowly, quickly, briskly,
rapidly, swiftly) are excluded throughout: pace shifts action identity
(running slowly ~ jogging), which is the confound under measurement.

The original 8x7 grid (grid.py) is kept unchanged for reproducibility of
the v1 results.

Near-synonymy screen (anisotropy-corrected: centred CLIP embeddings, since
raw cosines of bare words all sit in [0.85, 0.95] and absolute thresholds
are meaningless): two style pairs are outliers, angrily~aggressively (0.52)
and angrily~nervously (0.49), so per-style results for these should be
treated as correlated rather than independent. One weak style-action cell:
(stretching, gracefully), the v2 analogue of v1's (dancing, gracefully).

Usage:
  python grid_v2.py --out grid_v2_all_templates.json
"""
from __future__ import annotations
import json
from pathlib import Path

# (present-continuous phrase, third-person phrase)
ACTIONS_V2: list[tuple[str, str]] = [
    ("walking",                 "walks"),
    ("running",                 "runs"),
    ("jogging",                 "jogs"),
    ("jumping",                 "jumps"),
    ("sitting down",            "sits down"),
    ("standing up",             "stands up"),
    ("turning around",          "turns around"),
    ("bending over",            "bends over"),
    ("picking something up",    "picks something up"),
    ("throwing",                "throws"),
    ("waving their hand",       "waves their hand"),
    ("kicking",                 "kicks"),
    ("stretching",              "stretches"),
    ("leaning forward",         "leans forward"),
    ("squatting",               "squats"),
    ("swinging their arms",     "swings their arms"),
    ("pushing something",       "pushes something"),
    ("dancing",                 "dances"),
    ("lifting something",       "lifts something"),
    ("reaching forward",        "reaches forward"),
    ("stepping sideways",       "steps sideways"),
    ("raising their arms",      "raises their arms"),
    ("lowering their arms",     "lowers their arms"),
    ("placing something down",  "places something down"),
]

# adverb -> adjective (for the "in a(n) X manner" template)
STYLES_V2: dict[str, str] = {
    "angrily": "angry",
    "tiredly": "tired",
    "happily": "happy",
    "nervously": "nervous",
    "gracefully": "graceful",
    "heavily": "heavy",
    "confidently": "confident",
    "carefully": "careful",
    "casually": "casual",
    "gently": "gentle",
    "cautiously": "cautious",
    "calmly": "calm",
    "aggressively": "aggressive",
}


def _article(adj: str) -> str:
    return "an" if adj[0] in "aeiou" else "a"


def build(out_path: str | Path):
    styles = list(STYLES_V2)
    templates = []
    for t in range(3):
        neutral, styled = [], []
        for ing, third in ACTIONS_V2:
            if t in (0, 1):
                neutral.append(f"a person is {ing}")
            else:
                neutral.append(f"a person {third}")
        for s in styles:
            row = []
            for ing, third in ACTIONS_V2:
                if t == 0:
                    row.append(f"a person is {ing} {s}")
                elif t == 1:
                    row.append(f"a person is {s} {ing}")
                else:
                    adj = STYLES_V2[s]
                    row.append(f"a person {third} in {_article(adj)} {adj} manner")
            styled.append(row)
        templates.append({
            "template": t,
            "neutral": neutral,
            "styled": styled,
            "actions": [f"a person is {ing}" for ing, _ in ACTIONS_V2],
            "styles": styles,
        })
    Path(out_path).write_text(json.dumps(templates, indent=1))
    n = len(ACTIONS_V2) * (len(styles) + 1) * 3
    print(f"wrote {out_path}: {len(ACTIONS_V2)} actions x {len(styles)} styles"
          f" x 3 templates = {n} prompts")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="grid_v2_all_templates.json")
    args = p.parse_args()
    build(args.out)
