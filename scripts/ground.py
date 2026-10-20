#!/usr/bin/env python3
"""Stage 6 - verify that every answer is grounded in the source corpus.

An answer the source does not support is a fabrication, and a fabrication
teaches the model to invent. This stage measures, for each row, what share
of the answer's distinctive content words actually appear in the text
extracted from the row's own source document (and, failing that, in the
rest of the corpus).

Grounding is scored, not enforced by default. A hard threshold is a policy
decision that depends on the corpus: a textbook restates its own phrasing,
so a genuinely-grounded answer can still score low when the extraction was
lossy. The score is reported per row, per subject and as a histogram, and
--drop-below removes the worst offenders when a clean set is wanted.

Usage:
  python ground.py --output ./dataset
  python ground.py --output ./dataset --drop-below 0.55 --write-grounded
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, read_json, read_jsonl, write_jsonl  # noqa: E402

# Words too common to prove anything. A content word that appears in every
# textbook on the shelf is not evidence of grounding.
STOP = set("""the a an and or but of in on at to for with is are was were be been being
this that these those it its as by from into such which who whom whose what when where
why how if then than there their they them he she his her we you your i our us not no
can could will would shall should may might must have has had do does did done
also more most other some any each all both few many much very same so too only
about after before above below between during over under again further once
following find state write explain mention describe give note question answer
step steps solution example e.g i.e ie eg etc use used using make made take taken
new old first second third last next one two three four five six seven eight nine ten
cm mm m kg g m/s km m2 cm2 cm3 km2 degree degrees angle angles given called named
term terms concept defined define means refers refer known
such various several less least present presented based shown
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{3,}|\d+(?:\.\d+)?%?")


def content_words(text):
    """Distinctive words of a text: long enough, alphabetic, not boilerplate."""
    out = set()
    for w in WORD_RE.findall(text or ""):
        lw = w.lower().strip("-'")
        if len(lw) < 4 or lw in STOP:
            continue
        out.add(lw)
    return out


def build_index(corpus):
    """Per-document and whole-corpus word sets, for grounding lookups."""
    per_doc = {}
    whole = set()
    for rel, text in corpus.items():
        words = content_words(text)
        per_doc[rel] = words
        whole |= words
    return per_doc, whole


def score_row(row, per_doc, whole):
    """Share of the answer's content words that the corpus actually contains.

    Scored against the row's OWN document first, which is the strict test: an
    answer has to come from the document it is attributed to. The rest of
    the corpus is a weaker fallback, because a fact restated in another book
    is still grounded - it just is not attributable to this document.

    Returns (own_score, corpus_score, n_words). The own score is the one
    that matters for provenance.
    """
    words = content_words(row.get("output", ""))
    if not words:
        return 1.0, 1.0, 0
    own = per_doc.get(row.get("source", ""), set())
    own_hits = sum(1 for w in words if w in own)
    any_hits = sum(1 for w in words if w in own or w in whole)
    return own_hits / len(words), any_hits / len(words), len(words)


def main():
    ap = argparse.ArgumentParser(description="Check that answers are grounded "
                                             "in the extracted corpus")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--drop-below", type=float, default=None,
                    help="remove rows whose grounding score is below this")
    ap.add_argument("--write-grounded", action="store_true",
                    help="write train_grounded.jsonl / eval_grounded.jsonl")
    args = ap.parse_args()

    load_config(args.config)
    corpus = read_json(os.path.join(args.output, "corpus.json"), {}) or {}
    if not corpus:
        print("corpus.json not found - run extract.py first", file=sys.stderr)
        sys.exit(1)

    per_doc, whole = build_index(corpus)
    print(f"[ground] corpus: {len(corpus)} documents, "
          f"{len(whole):,} distinct content words")

    report = {}
    for split in ("train", "eval"):
        path = os.path.join(args.output, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        rows = read_jsonl(path)
        scored = []
        for r in rows:
            own, anyw, n = score_row(r, per_doc, whole)
            scored.append((own, anyw, n, r))
        if not scored:
            continue
        vals = [s for s, _, _, _ in scored]
        corpus_vals = [s for _, s, _, _ in scored]
        buckets = collections.Counter()
        for v in vals:
            if v >= 0.85:
                buckets["high (>=0.85)"] += 1
            elif v >= 0.55:
                buckets["medium (0.55-0.85)"] += 1
            else:
                buckets["low (<0.55)"] += 1
        avg = sum(vals) / len(vals)
        avg_corpus = sum(corpus_vals) / len(corpus_vals)
        print(f"\n[ground] {split}: {len(rows)} rows")
        print(f"    own-document grounding : avg {avg:.3f} | "
              f"min {min(vals):.3f} | max {max(vals):.3f}")
        print(f"    any-corpus grounding   : avg {avg_corpus:.3f}")
        for k in ("high (>=0.85)", "medium (0.55-0.85)", "low (<0.55)"):
            print(f"    {k}: {buckets[k]}")
        by_subject = collections.defaultdict(list)
        for s, _, _, r in scored:
            by_subject[r.get("subject", "General")].append(s)
        for sub, sv in sorted(by_subject.items()):
            print(f"    {sub}: avg {sum(sv)/len(sv):.3f} ({len(sv)} rows)")

        worst = sorted(scored, key=lambda x: x[0])[:10]
        if worst and worst[0][0] < 0.55:
            print("    least grounded rows:")
            for s, _, n, r in worst:
                print(f"      {s:.2f}  {r.get('instruction', '')[:68]}")

        report[split] = scored

        if args.drop_below is not None and args.write_grounded:
            kept = [(s, a, n, r) for s, a, n, r in scored if s >= args.drop_below]
            dropped = len(scored) - len(kept)
            out = os.path.join(args.output, f"{split}_grounded.jsonl")
            write_jsonl(out, [r for _, _, _, r in kept])
            print(f"    dropped {dropped} row(s) below {args.drop_below}; "
                  f"wrote {len(kept)} -> {out}")

    if not report:
        print("no train.jsonl / eval.jsonl found - run assemble.py first",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
