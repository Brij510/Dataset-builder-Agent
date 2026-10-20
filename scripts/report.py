#!/usr/bin/env python3
"""Dataset quality report.

Prints (and optionally writes) a markdown report covering volume, subject
balance, atom-kind mix, question diversity, source coverage, length
statistics and the health of the train/eval split.

Usage:
  python report.py --output ./dataset [--write report.md]
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, shingles  # noqa: E402


def stat(values):
    if not values:
        return {}
    return {
        "min": min(values), "median": int(statistics.median(values)),
        "mean": int(statistics.mean(values)), "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="dataset directory")
    ap.add_argument("--write", help="also write a markdown report here")
    args = ap.parse_args()

    d = args.output
    train = read_jsonl(os.path.join(d, "train.jsonl"))
    ev = read_jsonl(os.path.join(d, "eval.jsonl"))
    atoms = read_jsonl(os.path.join(d, "atoms.jsonl"))
    corpus = {}
    cp = os.path.join(d, "corpus.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            corpus = json.load(f)

    L = []
    A = L.append
    A("# Dataset Report\n")
    A(f"- train rows: **{len(train):,}**")
    A(f"- eval rows: **{len(ev):,}**")
    A(f"- source documents: **{len(corpus):,}**")
    A(f"- knowledge atoms: **{len(atoms):,}**\n")

    A("## Subject balance\n")
    for split, rows in (("train", train), ("eval", ev)):
        c = collections.Counter(r.get("subject", "General") for r in rows)
        A(f"**{split}**\n")
        A("| subject | rows | share |")
        A("|---|---|---|")
        tot = max(len(rows), 1)
        for k, v in c.most_common():
            A(f"| {k} | {v:,} | {v/tot*100:.1f}% |")
        A("")

    if atoms:
        A("## Atom kind mix\n")
        c = collections.Counter(a.get("kind") for a in atoms)
        A("| kind | atoms |")
        A("|---|---|")
        for k, v in c.most_common():
            A(f"| {k} | {v:,} |")
        A("")

    A("## Question diversity\n")
    openers = collections.Counter(" ".join(r["instruction"].split()[:2]).lower()
                                   for r in train)
    A(f"Distinct two-word openers: **{len(openers):,}**")
    A("\n| opener | rows |")
    A("|---|---|")
    for k, v in openers.most_common(12):
        A(f"| {k} | {v:,} |")
    A("")

    A("## Length profile\n")
    A("| metric | question chars | answer chars |")
    A("|---|---|---|")
    qs = stat([len(r["instruction"]) for r in train])
    ans = stat([len(r["output"]) for r in train])
    for m in ("min", "median", "mean", "max"):
        A(f"| {m} | {qs.get(m, 0)} | {ans.get(m, 0)} |")
    A("")

    A("## Source coverage\n")
    src = collections.Counter(r.get("source", "?") for r in train)
    A(f"Documents contributing train rows: **{len(src):,}**")
    A(f"Rows per document: min {min(src.values()) if src else 0}, "
      f"median {int(statistics.median(src.values())) if src else 0}, "
      f"max {max(src.values()) if src else 0}")
    top = src.most_common(5)
    A("\n| document | rows |")
    A("|---|---|")
    for k, v in top:
        A(f"| {k} | {v:,} |")
    A("")

    A("## Split health\n")
    if train and ev:
        tr_i = {r["instruction"].lower().strip() for r in train}
        shared = sum(1 for r in ev if r["instruction"].lower().strip() in tr_i)
        tr_sh = set()
        for r in train:
            tr_sh |= shingles(r["output"])
        worst = 0.0
        for r in ev:
            s = shingles(r["output"])
            if s:
                worst = max(worst, len(s & tr_sh) / len(s))
        redundant = 0
        scored = 0
        for r in train:
            if r.get("kind") == "restate":
                continue
            q = set(re.findall(r"[a-z]{4,}", r["instruction"].lower()))
            a = set(re.findall(r"[a-z]{4,}", r["output"].lower()))
            if len(a) < 4:
                continue
            scored += 1
            if len(a - q) / len(a) < 0.35:
                redundant += 1
        A(f"- eval/train shared instructions: **{shared}**")
        A(f"- max eval/train answer overlap: **{worst:.3f}**")
        A(f"- rows whose answer adds almost no new information: **{redundant:,}** "
          f"({redundant/max(scored,1)*100:.1f}%)")
        A(f"- eval share of total: **{len(ev)/max(len(train)+len(ev),1)*100:.1f}%**")
    A("")

    report = "\n".join(L)
    print(report)
    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nwritten -> {args.write}")


if __name__ == "__main__":
    main()
