#!/usr/bin/env python3
"""Stage 4 - merge sources, balance subjects, and split into train/eval with
leakage control.

Split contract:
  1. Eval rows come from HELD-OUT source documents (model never sees that material)
  2. Any train row whose answer overlaps an eval answer by more than
     `max_overlap` (symmetric 5-gram shingle check) is dropped
  3. Any eval row that still overlaps train is dropped
  4. Per-subject caps applied after the split
  5. Gold rows (hand-written) always stay in train

Usage:
  python assemble.py --output ./dataset [--eval-frac 0.07] [--max-overlap 0.10]
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (chatml, load_config, md5, norm_key, read_jsonl,  # noqa: E402
                    shingles, summarize_counts, write_jsonl)

PRIORITY = {"gold": 0, "llm": 1, "atoms": 2, "v1": 3, "manual": 1}


def load_rows(path, default_origin):
    out = []
    for o in read_jsonl(path):
        q = (o.get("instruction") or "").strip()
        a = (o.get("output") or "").strip()
        if not q or not a:
            continue
        out.append({"instruction": q, "output": a,
                    "subject": o.get("subject", "General"),
                    "source": o.get("source", "unknown"),
                    "kind": o.get("kind", ""),
                    "origin": o.get("origin", default_origin)})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instruction", "input", "output", "subject"])
        for r in rows:
            w.writerow([r["instruction"], "", r["output"], r["subject"]])


def write_jsonl_chat(path, rows):
    out = []
    for r in rows:
        rec = {"text": chatml(r["instruction"], r["output"]), "subject": r["subject"],
               "instruction": r["instruction"], "input": "", "output": r["output"]}
        if r.get("kind"):
            rec["kind"] = r["kind"]
        out.append(rec)
    write_jsonl(path, out)


def main():
    ap = argparse.ArgumentParser(description="Assemble + split train/eval")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--eval-frac", type=float)
    ap.add_argument("--max-overlap", type=float)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra qa jsonl files to merge (lowest priority)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scfg = cfg["split"]
    if args.eval_frac is not None:
        scfg["eval_frac"] = args.eval_frac
    if args.max_overlap is not None:
        scfg["max_overlap"] = args.max_overlap
    n = scfg.get("shingle_n", 5)

    out_dir = args.output
    sources = [
        (os.path.join(out_dir, "gold.jsonl"), "gold"),
        (os.path.join(out_dir, "qa_all.jsonl"), "atoms"),
    ]
    for p in args.extra:
        sources.append((p, "v1"))

    raw = []
    for path, origin in sources:
        if os.path.exists(path):
            rows = load_rows(path, origin)
            for r in rows:
                r["origin"] = r["origin"] if r["origin"] in PRIORITY else origin
            raw += rows
    if not raw:
        print("no input rows found - run qa.py first", file=sys.stderr)
        sys.exit(1)

    # ---- dedupe, honouring origin priority
    raw.sort(key=lambda r: PRIORITY.get(r["origin"], 9))
    seen_i, seen_a, merged = set(), set(), []
    for r in raw:
        ik = r["instruction"].lower()[:150]
        ak = md5(norm_key(r["output"]))
        if ik in seen_i or ak in seen_a:
            continue
        seen_i.add(ik); seen_a.add(ak)
        merged.append(r)
    print(f"[assemble] {len(raw)} input rows -> {len(merged)} unique")
    print(f"[assemble] by subject: {summarize_counts(merged)}")

    # ---- drop near-duplicate source documents before splitting.
    # A textbook and its scan produce identical content; if both survive, the
    # document-holdout split cannot work and the eval set stops being honest.
    doc_sh = {}
    for r in merged:
        doc_sh.setdefault(r["source"], set())
    for r in merged:
        doc_sh[r["source"]] |= shingles(r["output"], n)
    doc_list = list(doc_sh.items())
    dup_of = {}
    for i in range(len(doc_list)):
        if doc_list[i][0] in dup_of:
            continue
        for j in range(i + 1, len(doc_list)):
            if doc_list[j][0] in dup_of:
                continue
            a, b = doc_sh[doc_list[i][0]], doc_sh[doc_list[j][0]]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter / min(len(a), len(b)) > 0.80:
                dup_of[doc_list[j][0]] = doc_list[i][0]
    if dup_of:
        print(f"[assemble] {len(dup_of)} near-duplicate documents detected; "
              f"dropping their rows")
        merged = [r for r in merged if r["source"] not in dup_of]
        seen_i, seen_a = set(), set()
        merged2 = []
        for r in merged:
            ik = r["instruction"].lower()[:150]
            ak = md5(norm_key(r["output"]))
            if ik in seen_i or ak in seen_a:
                continue
            seen_i.add(ik); seen_a.add(ak)
            merged2.append(r)
        merged = merged2

    # ---- pick held-out source documents for eval
    by_src = collections.defaultdict(list)
    for r in merged:
        if r["origin"] == "gold":
            continue                      # gold always stays in train
        by_src[r["source"] or "unknown"].append(r)
    srcs = list(by_src.keys())
    import random
    random.seed(2024)
    random.shuffle(srcs)
    target = int(len(merged) * scfg["eval_frac"])
    per_doc_cap = max(1, int(target * 0.5))   # one document must not take over eval
    eval_rows = []
    eval_ids = set()
    used_sources = set()
    for s in srcs:
        if len(eval_rows) >= target:
            break
        rows = by_src[s]
        if not rows or len(rows) > 1500:
            continue
        for r in rows[:per_doc_cap]:
            eval_ids.add(id(r))
            eval_rows.append(r)
        used_sources.add(s)

    # ---- symmetric overlap filter between train candidates and eval
    eval_sh = [shingles(r["output"], n) for r in eval_rows]
    eval_inv = collections.defaultdict(list)
    for ei, s in enumerate(eval_sh):
        for g in s:
            eval_inv[g].append(ei)

    def leaks(text):
        s = shingles(text, n)
        if not s:
            return False
        hit = collections.Counter()
        for g in s:
            for ei in eval_inv.get(g, ()):
                hit[ei] += 1
        return any(c / max(len(eval_sh[ei]), 1) > scfg["max_overlap"]
                   for ei, c in hit.items())

    train_rows, dropped = [], 0
    for r in merged:
        if r["source"] in used_sources:
            continue           # whole held-out document stays out of train
        if leaks(r["output"]):
            dropped += 1
            continue
        train_rows.append(r)

    # Fallback: too few documents to hold one out without gutting train.
    # Switch to a row-level holdout instead (shingle filter still applies).
    if len(train_rows) < 0.2 * len(merged) and len(merged) > 10:
        print("[assemble] document holdout would gut train; "
              "falling back to row-level holdout")
        eval_rows = [r for r in merged if id(r) in eval_ids]
        if len(eval_rows) < target:
            extra = [r for r in merged if id(r) not in eval_ids]
            random.shuffle(extra)
            eval_rows += extra[:max(0, target - len(eval_rows))]
        eval_ids = {id(r) for r in eval_rows}
        train_rows, dropped = [], 0
        for r in merged:
            if id(r) in eval_ids:
                continue
            if leaks(r["output"]):
                dropped += 1
                continue
            train_rows.append(r)

    print(f"[assemble] held-out docs {len(used_sources)} | eval {len(eval_rows)} | "
          f"train {len(train_rows)} (dropped {dropped} for overlap)")

    # ---- balance
    caps = scfg.get("subject_caps", {})
    by_subj = collections.defaultdict(list)
    for r in train_rows:
        by_subj[r["subject"]].append(r)
    train_rows = []
    for s, lst in by_subj.items():
        random.shuffle(lst)
        train_rows += lst[:caps.get(s, 10 ** 9)]
    random.shuffle(train_rows)

    # ---- final cleanup: drop any eval row still overlapping train
    tr_sh = set()
    for r in train_rows:
        tr_sh |= shingles(r["output"], n)
    keep = []
    for r in eval_rows:
        s = shingles(r["output"], n)
        ov = (len(s & tr_sh) / len(s)) if s else 0.0
        if ov <= scfg["max_overlap"]:
            keep.append(r)
    if len(keep) != len(eval_rows):
        print(f"[assemble] dropped {len(eval_rows) - len(keep)} leaky eval rows")
    eval_rows = keep

    worst = 0.0
    for r in eval_rows:
        s = shingles(r["output"], n)
        if s:
            worst = max(worst, len(s & tr_sh) / len(s))
    print(f"[assemble] max eval/train answer overlap: {worst:.3f} "
          f"(limit {scfg['max_overlap']})")
    print(f"[assemble] TRAIN {len(train_rows)} {summarize_counts(train_rows)}")
    print(f"[assemble] EVAL  {len(eval_rows)} {summarize_counts(eval_rows)}")

    write_jsonl_chat(os.path.join(out_dir, "train.jsonl"), train_rows)
    write_csv(os.path.join(out_dir, "train.csv"), train_rows)
    write_jsonl_chat(os.path.join(out_dir, "eval.jsonl"), eval_rows)
    write_csv(os.path.join(out_dir, "eval.csv"), eval_rows)
    print(f"[assemble] wrote train.jsonl/csv and eval.jsonl/csv to {out_dir}")


if __name__ == "__main__":
    main()
