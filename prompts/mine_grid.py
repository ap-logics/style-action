"""
Mine action-verb and manner-adverb frequencies from the HumanML3D caption
corpus, using the POS tags that ship inside each caption file
(format: raw_text#word/TAG word/TAG...#start#end).

Outputs ranked frequency tables for (a) verb lemmas and (b) -ly adverbs,
which serve two purposes:
  1. grid design — actions/styles drawn from the corpus are in-distribution
     by construction, closing the OOD objection at the design stage
  2. the frequency covariate — regress per-cell coupling against corpus
     frequency to test the OOD-fragility vs representational accounts

Usage:
  python mine_grid.py --texts /path/to/texts --top 40
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

# verbs that are grammatical scaffolding or camera/actor boilerplate,
# not motion actions
STOP_VERBS = {
    "be", "do", "have", "go", "get", "make", "take", "start", "stop",
    "begin", "continue", "return", "appear", "seem", "look", "use",
    "keep", "put", "come", "try", "then",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", default="corpus_frequencies.json")
    args = p.parse_args()

    verb_counts: Counter = Counter()
    adv_counts: Counter = Counter()
    n_captions = 0

    for f in Path(args.texts).glob("*.txt"):
        for line in f.read_text(errors="ignore").splitlines():
            parts = line.split("#")
            if len(parts) < 2:
                continue
            n_captions += 1
            for tok in parts[1].split():
                if "/" not in tok:
                    continue
                word, tag = tok.rsplit("/", 1)
                word = word.lower()
                if tag == "VERB" and word.isalpha() and word not in STOP_VERBS:
                    verb_counts[word] += 1
                elif tag == "ADV" and word.endswith("ly") and word.isalpha():
                    adv_counts[word] += 1

    print(f"{n_captions} captions parsed\n")
    print(f"Top {args.top} action verbs:")
    for w, c in verb_counts.most_common(args.top):
        print(f"  {w:<16} {c}")
    print(f"\nTop {args.top} -ly adverbs:")
    for w, c in adv_counts.most_common(args.top):
        print(f"  {w:<16} {c}")

    Path(args.out).write_text(json.dumps({
        "n_captions": n_captions,
        "verbs": dict(verb_counts.most_common(300)),
        "adverbs": dict(adv_counts.most_common(300)),
    }, indent=1))
    print(f"\nSaved full tables to {args.out}")


if __name__ == "__main__":
    main()
