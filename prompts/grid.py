"""
8×8 action-style prompt grid.

Actions: 8 semantically distinct motion categories, well-represented in
HumanML3D.  Present-continuous phrasing ("a person is walking") matches
the sentence-style annotations the models were trained on.

Styles: neutral baseline (s0, empty modifier) + 7 expressive adverbs.
Chosen to be clearly manner-only with no action-adjacent reading in any
row.  "slowly" was excluded because "running slowly" crosses toward
walking semantics — exactly the confound we are measuring.

Known weak cell: (dancing, gracefully) — low contrast by design.
The graceful prior for dancing means style effect will be attenuated;
this is noted as a limitation rather than fixed.

Styled prompt format: action + " " + adverb  ("a person is walking angrily").
"""
from __future__ import annotations
from pathlib import Path
import json

ACTIONS: list[str] = [
    "a person is walking",
    "a person is running",
    "a person is jumping",
    "a person is sitting down",
    "a person is waving their hand",
    "a person is kicking",
    "a person is throwing",
    "a person is dancing",
]

STYLES: list[str] = [
    "",             # s0 — neutral baseline (no modifier)
    "angrily",      # emotional — frustration / aggression
    "tiredly",      # emotional/physical state — low energy
    "happily",      # emotional — positive valence
    "nervously",    # emotional — anxiety
    "gracefully",   # manner — fluid / light (weak cell with dancing)
    "heavily",      # manner — weight / force emphasis
    "confidently",  # manner — assured bearing
]


# Paraphrase templates for the robustness check. Each template is a pair of
# format functions (neutral, styled) applied to the base action phrase
# ("walking", "sitting down", ...) and the adverb/adjective.
#
# T2 deliberately moves the adverb before the verb phrase to test whether
# adverb position matters. T3 uses "in a(n) X manner" with the adjective
# form — NOT "an angry person is walking", because that template breaks for
# "heavily" ("a heavy person" reads as body weight, not movement manner).
STYLE_ADJ: dict[str, str] = {
    "angrily": "angry",
    "tiredly": "tired",
    "happily": "happy",
    "nervously": "nervous",
    "gracefully": "graceful",
    "heavily": "heavy",
    "confidently": "confident",
}

_BASE = [a.removeprefix("a person is ") for a in ACTIONS]


def _article(adj: str) -> str:
    return "an" if adj[0] in "aeiou" else "a"


TEMPLATES: list[tuple] = [
    # T1 — adverb-final (primary grid)
    (lambda act: f"a person is {act}",
     lambda act, sty: f"a person is {act} {sty}"),
    # T2 — adverb-first
    (lambda act: f"a person is {act}",
     lambda act, sty: f"a person is {sty} {act}"),
    # T3 — adjectival manner phrase
    (lambda act: f"a person {_third_person(act)}",
     lambda act, sty: f"a person {_third_person(act)} in "
                      f"{_article(STYLE_ADJ[sty])} {STYLE_ADJ[sty]} manner"),
]


def _third_person(act: str) -> str:
    """'walking' -> 'walks', 'sitting down' -> 'sits down', etc."""
    head, _, rest = act.partition(" ")
    stems = {
        "walking": "walks", "running": "runs", "jumping": "jumps",
        "sitting": "sits", "waving": "waves", "kicking": "kicks",
        "throwing": "throws", "dancing": "dances",
    }
    verb = stems[head]
    return f"{verb} {rest}".strip()


def build_grid(template: int = 0) -> tuple[list[str], list[list[str]], list[str], list[str]]:
    """
    Per-style layout matching the paper's K_S / K_T^j formulation.

    Args:
        template : paraphrase template index (0 = primary adverb-final grid)

    Returns:
        neutral_prompts : 8 prompts, one per action (style s0)
        styled_prompts  : 7 lists of 8 prompts — styled_prompts[j][a] is
                          action a under style j+1 (skipping s0)
        actions         : the 8 action phrases
        styles          : the 7 non-neutral modifiers
    """
    neutral_fmt, styled_fmt = TEMPLATES[template]
    neutral = [neutral_fmt(a) for a in _BASE]
    styled = [[styled_fmt(a, s) for a in _BASE] for s in STYLES[1:]]
    return neutral, styled, list(ACTIONS), STYLES[1:]


def flat_prompts() -> tuple[list[str], list[int]]:
    """
    All 64 unique prompts as a flat list for the generation stage.
    Index 0-7: neutral. Index 8-63: styled, ordered style-major.
    Returns (prompts, action_labels).
    """
    neutral, styled, _, _ = build_grid()
    prompts = list(neutral)
    labels = list(range(len(ACTIONS)))
    for style_row in styled:
        prompts.extend(style_row)
        labels.extend(range(len(ACTIONS)))
    return prompts, labels


def save_grid(out_dir: str | Path = ".") -> None:
    out = Path(out_dir)
    prompts, labels = flat_prompts()

    (out / "prompts.txt").write_text("\n".join(prompts))
    (out / "action_labels.json").write_text(json.dumps(labels))

    print(f"Saved {len(prompts)} prompts to {out}")
    for i, p_ in enumerate(prompts):
        print(f"  [{i:02d}] {p_}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=".", help="Output directory")
    args = p.parse_args()
    save_grid(args.out)
