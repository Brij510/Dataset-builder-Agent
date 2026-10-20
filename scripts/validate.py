#!/usr/bin/env python3
"""Stage 5 - validate the generated dataset.

Hard failures (exit 1): bad JSON, broken ChatML, CSV/JSONL mismatch,
watermark noise, duplicate instructions, eval/train instruction overlap,
answer-overlap above the configured limit.
Soft warnings: short answers, markdown fences, low template diversity.

Usage:
  python validate.py --output ./dataset [--max-overlap 0.10]
"""
import argparse
import collections
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_idf, load_config, read_jsonl, shingles,  # noqa: E402
                    weighted_overlap)

WATERMARK = re.compile(r"prashant\s*kirad|shobhit\s*nirwan|created with|"
                       r"downloaded from|all rights reserved", re.I)


def main():
    ap = argparse.ArgumentParser(description="Validate a generated dataset")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--max-overlap", type=float)
    ap.add_argument("--min-answer-chars", type=int, default=25)
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures (use in CI)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    limit = args.max_overlap if args.max_overlap is not None \
        else cfg["split"]["max_overlap"]
    n = cfg["split"].get("shingle_n", 5)

    errors, warns = [], []
    data = {}
    for split in ("train", "eval"):
        jp = os.path.join(args.output, f"{split}.jsonl")
        cp = os.path.join(args.output, f"{split}.csv")
        if not os.path.exists(jp):
            errors.append(f"{split}: missing {jp}")
            continue
        rows = read_jsonl(jp)
        for i, o in enumerate(rows, 1):
            if not isinstance(o, dict):
                errors.append(f"{split} row {i}: not a JSON object")
                continue
            for required in ("text", "instruction", "output"):
                if required not in o:
                    errors.append(f"{split} row {i}: missing key '{required}'")
            t = o.get("text", "") or ""
            if t.count("<|im_start|>") != 2 or t.count("<|im_end|>") != 2:
                errors.append(f"{split} row {i}: broken ChatML markers")
            if "<|im_start|>user" not in t or "<|im_start|>assistant" not in t:
                errors.append(f"{split} row {i}: missing user/assistant roles")
            if WATERMARK.search(t):
                errors.append(f"{split} row {i}: contains watermark noise")
            if "```" in t:
                warns.append(f"{split} row {i}: contains a markdown fence")
            if len(o.get("output", "")) < args.min_answer_chars:
                warns.append(f"{split} row {i}: very short answer")
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                cr = list(csv.reader(f))
            if cr and cr[0] != ["instruction", "input", "output", "subject"]:
                errors.append(f"{split}: unexpected CSV header {cr[0] if cr else None}")
            if len(cr) - 1 != len(rows):
                errors.append(f"{split}: CSV has {len(cr)-1} rows, JSONL has {len(rows)}")
        data[split] = rows
        print(f"{split}: {len(rows)} rows | subjects "
              f"{dict(collections.Counter(r.get('subject') for r in rows))}")

    if "train" in data and "eval" in data:
        tr, ev = data["train"], data["eval"]
        tr_i = {(r.get("instruction") or "").lower().strip() for r in tr}
        shared = sum(1 for r in ev
                     if (r.get("instruction") or "").lower().strip() in tr_i)
        if shared:
            errors.append(f"eval instructions also present in train: {shared}")

        tr_sh = set()
        for r in tr:
            tr_sh |= shingles(r.get("output") or "", n)
        # the same IDF-weighted measure assemble.py uses, so the two agree
        idf = build_idf([r.get("output") or "" for r in tr + ev], n)
        worst, bad = 0.0, 0
        for r in ev:
            s = shingles(r.get("output") or "", n)
            if s:
                ov = weighted_overlap(s, tr_sh, idf)
                worst = max(worst, ov)
                if ov > limit:
                    bad += 1
        print(f"\nLeakage: shared instructions {shared} | "
              f"max answer overlap {worst:.3f} (limit {limit})")
        if bad:
            errors.append(f"{bad} eval rows exceed the overlap limit")

        # near-duplicate questions: same meaning, different wording.
        # Exact dedupe in the generator cannot catch these, and they waste
        # training signal while making the eval set look easier than it is.
        qsh = {}
        for i, r in enumerate(tr):
            qsh[i] = shingles((r.get("instruction") or "").lower(), 4)
        inv = collections.defaultdict(list)
        for i, ss in qsh.items():
            for g in ss:
                inv[g].append(i)
        dup_pairs = set()
        for g, ids in inv.items():
            if len(ids) > 12:
                continue
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b]
                    if i > j:
                        i, j = j, i
                    if (i, j) in dup_pairs:
                        continue
                    sa, sb = qsh[i], qsh[j]
                    if not sa or not sb:
                        continue
                    if len(sa & sb) / min(len(sa), len(sb)) > 0.80:
                        dup_pairs.add((i, j))
        if dup_pairs:
            # pairs inside one document are generator artefacts (the same atom
            # re-asked in a different template); pairs across documents are the
            # corpus restating the same fact and are a separate concern
            same_doc = sum(1 for i, j in dup_pairs
                           if (tr[i].get("source") or "") == (tr[j].get("source") or "")
                           and (tr[i].get("output") or "") == (tr[j].get("output") or ""))
            cross_doc = len(dup_pairs) - same_doc
            print(f"Near-duplicate question pairs in train: {len(dup_pairs)} "
                  f"({same_doc} same question+answer, {cross_doc} same question, "
                  f"different answer)")
            if same_doc > max(20, len(tr) * 0.01):
                warns.append(
                    f"{same_doc} rows repeat both the question and the answer - "
                    f"these come from qa.max_per_answer > 1. Set it to 1 for "
                    f"cleaner data, or grow the corpus instead")
            if cross_doc > max(20, len(tr) * 0.02):
                warns.append(
                    f"{cross_doc} near-duplicate questions have different answers - "
                    f"the corpus states the same fact in several places")

        prefixes = collections.Counter(" ".join((r.get("instruction") or "").split()[:2]).lower()
                                       for r in tr)
        print(f"Distinct question openers: {len(prefixes)}")
        print("Top openers:", prefixes.most_common(8))
        if len(prefixes) < 5 and len(tr) > 500:
            warns.append("very low question diversity - check the qa.py rotation")

        # A good pair adds information. If almost every content word of the
        # answer is already in the question, the row teaches the model to echo.
        # "restate" rows quote the statement on purpose (paraphrase training),
        # so they are counted separately rather than treated as a defect.
        echo = 0
        restate = 0
        scored = 0
        for r in tr:
            if r.get("kind") == "restate":
                restate += 1
                continue
            q = set(re.findall(r"[a-z]{4,}", (r.get("instruction") or "").lower()))
            a = set(re.findall(r"[a-z]{4,}", (r.get("output") or "").lower()))
            if len(a) < 4:
                continue
            scored += 1
            novelty = len(a - q) / len(a)
            if novelty < 0.35:
                echo += 1
        if restate:
            print(f"Restate-style rows (quote the statement on purpose): {restate}")
        if scored:
            pct = echo / scored * 100
            print(f"Rows whose answer adds almost no new information: {echo} "
                  f"({pct:.1f}%)")
            if pct > 20:
                warns.append(f"{pct:.0f}% of rows are echo rows - the model will "
                             f"learn to repeat the question, not answer it")

        alen = [len(r.get("output") or "") for r in tr]
        qlen = [len(r.get("instruction") or "") for r in tr]
        print(f"Answer chars : min {min(alen)} median {int(statistics.median(alen))} max {max(alen)}")
        print(f"Question chars: min {min(qlen)} median {int(statistics.median(qlen))} max {max(qlen)}")

    if warns:
        print(f"\nWarnings ({len(warns)}):")
        for w in warns[:15]:
            print("  -", w)
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:30]:
            print("  -", e)
        sys.exit(1)
    if args.strict and warns:
        print(f"\nSTRICT MODE: {len(warns)} warning(s) treated as failures")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
