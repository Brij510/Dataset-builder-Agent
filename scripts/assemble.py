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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_idf, chatml, load_config, md5, norm_key,  # noqa: E402
                    read_jsonl, shingles, summarize_counts, weighted_overlap,
                    write_jsonl)

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

    # ---- drop near-duplicate source documents BEFORE content dedupe.
    # If a textbook and its scan are both present, the content dedupe below
    # would delete the duplicate rows and leave the document-level signal
    # invisible, so this has to run first.
    doc_text = {}
    for r in raw:
        doc_text.setdefault(r["source"], [])
        doc_text[r["source"]].append(r["output"])
    doc_sh = {src: set().union(*[shingles(t, n) for t in texts]) if texts else set()
              for src, texts in doc_text.items()}
    doc_list = sorted(doc_sh.items())
    dup_of = {}
    for i in range(len(doc_list)):
        if doc_list[i][0] in dup_of:
            continue
        a = doc_sh[doc_list[i][0]]
        for j in range(i + 1, len(doc_list)):
            if doc_list[j][0] in dup_of:
                continue
            b = doc_sh[doc_list[j][0]]
            if not a or not b:
                continue
            inter = len(a & b)
            if not inter:
                continue
            # Jaccard, matching dedupe.py: containment overestimates similarity
            # for documents that share boilerplate but differ in content
            if inter / len(a | b) > 0.75:
                dup_of[doc_list[j][0]] = doc_list[i][0]
    if dup_of:
        print(f"[assemble] {len(dup_of)} near-duplicate documents detected; "
              f"dropping their rows")
        raw = [r for r in raw if r["source"] not in dup_of]

    # ---- dedupe, honouring origin priority.
    # The answer bound must match qa.py's max_per_answer, otherwise the
    # allowance the generator made is undone here and rows silently vanish.
    max_per_answer = cfg["qa"].get("max_per_answer", 1)
    raw.sort(key=lambda r: PRIORITY.get(r["origin"], 9))
    seen_i, seen_a, merged = set(), collections.Counter(), []
    for r in raw:
        ik = r["instruction"].lower()[:150]
        ak = md5(norm_key(r["output"]))
        if ik in seen_i or seen_a[ak] >= max_per_answer:
            continue
        seen_i.add(ik); seen_a[ak] += 1
        merged.append(r)
    print(f"[assemble] {len(raw)} input rows -> {len(merged)} unique")
    print(f"[assemble] by subject: {summarize_counts(merged)}")

    # ---- pick held-out source documents for eval
    #
    # The eval set must (a) come from documents the model never sees, and
    # (b) mirror the subject mix of the training set, so the score is not
    # dominated by whichever subject happens to have small documents.
    # Each subject gets a quota proportional to its share of the rows, and
    # whole documents are taken until the quota is met.
    by_src = collections.defaultdict(list)
    for r in merged:
        if r["origin"] == "gold":
            continue                      # gold always stays in train
        by_src[r["source"] or "unknown"].append(r)

    import random
    random.seed(2024)
    by_subject_docs = collections.defaultdict(list)
    for src, rows in by_src.items():
        if not rows or len(rows) > 1500:
            continue
        by_subject_docs[rows[0].get("subject", "General")].append(src)

    subject_rows = collections.Counter()
    for r in merged:
        subject_rows[r.get("subject", "General")] += 1

    target = int(len(merged) * scfg["eval_frac"])
    eval_rows = []
    eval_ids = set()
    used_sources = set()

    def take(src):
        """A held-out document joins the eval set WHOLE.

        Truncating it would silently drop the leftover rows, because the whole
        document is excluded from train.
        """
        for r in by_src[src]:
            eval_ids.add(id(r))
            eval_rows.append(r)
        used_sources.add(src)

    for sub in sorted(by_subject_docs):
        docs = by_subject_docs[sub]
        # prefer documents that land close to the subject's quota
        quota = max(1, int(target * subject_rows[sub] / max(len(merged), 1)))
        docs.sort(key=lambda src: (abs(len(by_src[src]) - quota), len(by_src[src])))
        taken = 0
        for src in docs:
            if taken >= quota:
                break
            take(src)
            taken += len(by_src[src])
        if not taken and docs:
            # this subject has no document at all; take the smallest one
            smallest = min(docs, key=lambda src: len(by_src[src]))
            take(smallest)

    # ---- symmetric overlap filter between train candidates and eval.
    # Weighted by IDF so shared boilerplate does not count as leakage - only
    # distinctive content does.
    idf = build_idf([r["output"] for r in merged], n)
    eval_sh = [shingles(r["output"], n) for r in eval_rows]
    eval_inv = collections.defaultdict(list)
    for ei, s in enumerate(eval_sh):
        for g in s:
            eval_inv[g].append(ei)

    def leaks(text, limit):
        s = shingles(text, n)
        if not s:
            return False
        for g in s:
            for ei in eval_inv.get(g, ()):
                if weighted_overlap(s, eval_sh[ei], idf) > limit:
                    return True
        return False

    def build_train(limit):
        rows, dropped = [], 0
        for r in merged:
            if r["source"] in used_sources:
                continue       # whole held-out document stays out of train
            if leaks(r["output"], limit):
                dropped += 1
                continue
            rows.append(r)
        return rows, dropped

    # The filter must never empty the training set. If it would, relax the
    # limit rather than leave the user with no training data.
    train_rows, dropped = [], 0
    for limit in (scfg["max_overlap"], scfg["max_overlap"] * 2, 1.0):
        train_rows, dropped = build_train(limit)
        if len(train_rows) >= 0.2 * len(merged) or limit >= 1.0:
            break
    if dropped and limit > scfg["max_overlap"]:
        print(f"[assemble] relaxed the overlap filter to {limit:.2f} to keep "
              f"a usable training set")

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
        for limit in (scfg["max_overlap"], scfg["max_overlap"] * 2, 1.0):
            train_rows, dropped = [], 0
            for r in merged:
                if id(r) in eval_ids:
                    continue
                if leaks(r["output"], limit):
                    dropped += 1
                    continue
                train_rows.append(r)
            if len(train_rows) >= 0.2 * len(merged) or limit >= 1.0:
                break

    print(f"[assemble] held-out docs {len(used_sources)} | eval {len(eval_rows)} | "
          f"train {len(train_rows)} (dropped {dropped} for overlap)")

    # ---- if a subject lost its only document, give one back, otherwise that
    # subject disappears from training entirely
    def subject_of(rows):
        return rows[0].get("subject", "General") if rows else None

    # subjects that still have rows in the training set. This must come from
    # train_rows, not from "everything not in eval" - the leak filter can
    # remove a row without putting it in eval.
    train_subjects = {r.get("subject", "General") for r in train_rows}
    eval_subjects = {subject_of(by_src[src]) for src in used_sources}
    for sub in sorted(eval_subjects - train_subjects):
        # this subject would vanish from training - hand one document back
        for src in sorted(used_sources):
            if subject_of(by_src[src]) != sub:
                continue
            used_sources.discard(src)
            if len(eval_rows) - len(by_src[src]) < 1:
                continue          # never hand back so much that eval empties
            for r in by_src[src]:
                eval_ids.discard(id(r))
            eval_rows = [e for e in eval_rows
                         if e.get("source") != src]
            # the handed-back rows belong to train again
            train_rows, dropped = [], 0
            for limit in (scfg["max_overlap"], scfg["max_overlap"] * 2, 1.0):
                train_rows, dropped = [], 0
                for r in merged:
                    if id(r) in eval_ids:
                        continue
                    if leaks(r["output"], limit):
                        dropped += 1
                        continue
                    train_rows.append(r)
                if len(train_rows) >= 0.2 * len(merged) or limit >= 1.0:
                    break
            break

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
        ov = weighted_overlap(s, tr_sh, idf) if s else 0.0
        if ov <= scfg["max_overlap"]:
            keep.append(r)
    if len(keep) != len(eval_rows):
        if not keep and eval_rows:
            # never ship an empty eval set: keep the least-overlapping rows
            scored = sorted(eval_rows,
                            key=lambda r: weighted_overlap(
                                shingles(r["output"], n), tr_sh, idf))
            keep = scored[:max(1, len(eval_rows) // 4)]
            print(f"[assemble] every eval row overlapped train; kept the "
                  f"{len(keep)} least-overlapping rows")
        else:
            print(f"[assemble] dropped {len(eval_rows) - len(keep)} leaky eval rows")
    eval_rows = keep

    worst = 0.0
    for r in eval_rows:
        s = shingles(r["output"], n)
        if s:
            worst = max(worst, weighted_overlap(s, tr_sh, idf))
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
