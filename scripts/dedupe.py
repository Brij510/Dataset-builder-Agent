#!/usr/bin/env python3
"""Prune near-duplicate source documents from the extracted corpus.

A textbook and its scan (or the same book in two folders) produce identical
content. Left in place they waste generation effort and, worse, they break the
document-holdout split: the model would see the same material in both halves
even though the two halves come from "different" files.

Runs on corpus.json and rewrites it in place (with a backup).

Usage:
  python dedupe.py --output ./dataset [--threshold 0.60] [--dry-run]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, read_json, shingles, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Remove near-duplicate documents")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Jaccard similarity above which the second document is dropped")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="ignore documents shorter than this when comparing")
    ap.add_argument("--min-shingles", type=int, default=30,
                    help="ignore documents with fewer distinct shingles")
    ap.add_argument("--shingle-n", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thr = args.threshold if args.threshold is not None else 0.75
    min_chars = args.min_chars
    min_shingles = args.min_shingles
    n = args.shingle_n or cfg["split"].get("shingle_n", 5)

    path = os.path.join(args.output, "corpus.json")
    corpus = read_json(path)
    if not corpus:
        print(f"[dedupe] no corpus at {path}", file=sys.stderr)
        sys.exit(1)

    doc_sh = {}
    for src, text in corpus.items():
        if len(text or "") < min_chars:
            continue
        s = shingles(text or "", n)
        if len(s) >= min_shingles:
            doc_sh[src] = s
    keys = sorted(doc_sh)
    dup_of = {}
    for i, a_key in enumerate(keys):
        if a_key in dup_of:
            continue
        a = doc_sh[a_key]
        if not a:
            continue
        for b_key in keys[i + 1:]:
            if b_key in dup_of:
                continue
            b = doc_sh[b_key]
            if not b:
                continue
            inter = len(a & b)
            if not inter:
                continue
            # Jaccard, not containment: two different question papers share
            # boilerplate and score high on containment, but their union is far
            # larger than their intersection, so Jaccard separates them.
            if inter / len(a | b) > thr:
                dup_of[b_key] = a_key

    print(f"[dedupe] {len(corpus)} documents, {len(doc_sh)} comparable, "
          f"jaccard threshold {thr}")
    if dup_of:
        for dup, keep in sorted(dup_of.items()):
            print(f"[dedupe]   drop {dup}  (duplicate of {keep})")
        if not args.dry_run:
            backup = path + ".pre-dedupe"
            if not os.path.exists(backup):
                write_json(backup, corpus)
            write_json(path, {k: v for k, v in corpus.items() if k not in dup_of})
            print(f"[dedupe] wrote {len(corpus) - len(dup_of)} documents "
                  f"(backup: {backup})")
    else:
        print("[dedupe] no near-duplicates found")


if __name__ == "__main__":
    main()
