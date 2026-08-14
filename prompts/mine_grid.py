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
  # HumanML3D (POS tags embedded in the caption files):
  python mine_grid.py --texts /path/to/texts --top 40
  # Motion-X / any plain-text corpus (one caption per line or per file;
  # requires spaCy: pip install spacy && python -m spacy download en_core_web_sm):
  python mine_grid.py --texts /path/to/motionx/texts --format plain --top 40
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
    p.add_argument("--format", choices=["humanml3d", "plain"], default="humanml3d",
                   help="humanml3d: '#'-delimited files with word/TAG tokens; "
                        "plain: untagged captions, tagged with spaCy")
    p.add_argument("--glob", default="*.txt", help="filename pattern under --texts")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", default="corpus_frequencies.json")
    args = p.parse_args()

    nlp = None
    if args.format == "plain":
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    verb_counts: Counter = Counter()
    adv_counts: Counter = Counter()
    n_captions = 0

    def tally(word, tag):
        word = word.lower()
        if tag == "VERB" and word.isalpha() and word not in STOP_VERBS:
            verb_counts[word] += 1
        elif tag == "ADV" and word.endswith("ly") and word.isalpha():
            adv_counts[word] += 1

    files = sorted(Path(args.texts).rglob(args.glob))
    if not files:
        raise SystemExit(f"no files matching {args.glob} under {args.texts}")
    if args.format == "humanml3d":
        for f in files:
            for line in f.read_text(errors="ignore").splitlines():
                parts = line.split("#")
                if len(parts) < 2:
                    continue
                n_captions += 1
                for tok in parts[1].split():
                    if "/" not in tok:
                        continue
                    word, tag = tok.rsplit("/", 1)
                    tally(word, tag)
    else:
        def caption_stream():
            for f in files:
                for line in f.read_text(errors="ignore").splitlines():
                    line = line.split("#")[0].strip()   # tolerate stray hml3d format
                    if line:
                        yield line
        for doc in nlp.pipe(caption_stream(), batch_size=256):
            n_captions += 1
            for tok in doc:
                tally(tok.text, tok.pos_)

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
