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
from common import (build_idf, build_overlap_index, chatml, load_config,  # noqa: E402
                    md5, norm_key, max_overlap, read_jsonl, shingles,
                    summarize_counts, symmetric_overlap, weighted_overlap,
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
        w.writerow(["instruction", "input", "output", "subject", "source"])
        for r in rows:
            w.writerow([r["instruction"], "", r["output"], r["subject"],
                        r.get("source", "")])


def write_jsonl_chat(path, rows):
    out = []
    for r in rows:
        rec = {"text": chatml(r["instruction"], r["output"]), "subject": r["subject"],
               "instruction": r["instruction"], "input": "", "output": r["output"]}
        if r.get("kind"):
            rec["kind"] = r["kind"]
        # provenance: every row must point back at the document it came from
        if r.get("source"):
            rec["source"] = r["source"]
        if r.get("origin"):
            rec["origin"] = r["origin"]
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
    # distinctive content does. Measured in BOTH directions: a short train
    # answer inside a long eval answer is a leak even though eval-to-train
    # scores it near zero.
    #
    # The index is rebuilt from the CURRENT eval set every time it is
    # consulted. Holding a stale index across a change to the eval set makes
    # the filter judge rows against material that is no longer in eval, which
    # silently drops rows that belong in neither split.
    def eval_index():
        sh = [shingles(r["output"], n) for r in eval_rows]
        return sh, build_overlap_index(sh), build_idf(
            [r["output"] for r in eval_rows + merged], n)

    def build_train(limit):
        """Train rows that do not leak into the current eval set."""
        ev_sh, ev_inv, idf = eval_index()
        rows, dropped = [], 0
        for r in merged:
            if id(r) in eval_ids:
                continue           # this row IS in eval
            if r["source"] in used_sources:
                continue           # whole held-out document stays out of train
            s = shingles(r["output"], n)
            if s and max_overlap(s, ev_inv, ev_sh, idf, symmetric=True)[0] > limit:
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
            train_rows, dropped = build_train(limit)
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
            if len(eval_rows) - len(by_src[src]) < 1:
                continue          # never hand back so much that eval empties
            # remove the document from eval FIRST, then rebuild train against
            # the new eval set - otherwise the handed-back rows are judged
            # against themselves and end up in neither split
            for r in by_src[src]:
                eval_ids.discard(id(r))
            eval_rows = [e for e in eval_rows
                         if e.get("source") != src]
            used_sources.discard(src)
            train_rows, dropped = [], 0
            for limit in (scfg["max_overlap"], scfg["max_overlap"] * 2, 1.0):
                train_rows, dropped = build_train(limit)
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

    # ---- final cleanup: remove the leaking TRAIN rows, never the eval rows.
    #
    # The train set is abundant and the eval set is scarce, so a leak is
    # always resolved by dropping the train row. Dropping the eval row
    # instead loses the only copy of that document's material, and when it
    # happens to be the last document of a subject, that subject disappears
    # from the eval set entirely.
    #
    # The IDF basis is recomputed over the final train+eval sets so this
    # script and validate.py measure exactly the same thing.
    idf = build_idf([r.get("output") or "" for r in train_rows + eval_rows], n)
    ev_sh_list = [shingles(r.get("output") or "", n) for r in eval_rows]
    ev_inv = build_overlap_index(ev_sh_list)

    # Measured symmetrically, and against the row it actually conflicts with
    # rather than the union of all eval shingles. The union overstates
    # overlap: two eval rows that each share a different fragment with a
    # train row can push the union measure over the limit even though no
    # single eval row really duplicates it.
    kept_train, dropped_final = [], 0
    for r in train_rows:
        s = shingles(r.get("output") or "", n)
        if not s:
            kept_train.append(r)
            continue
        worst, _ = max_overlap(s, ev_inv, ev_sh_list, idf, symmetric=True)
        if worst <= scfg["max_overlap"]:
            kept_train.append(r)
        else:
            dropped_final += 1
    if dropped_final:
        print(f"[assemble] dropped {dropped_final} train rows that still "
              f"overlapped the eval set")
    train_rows = kept_train

    # ---- safety net: no row may vanish silently.
    #
    # Every stage above can drop a row for a good reason, but a row that ends
    # up in NEITHER split is data loss the user never asked for and cannot
    # see. Such a row is offered back to train - but it must clear the same
    # leak filter as every other train row, or returning it would undo the
    # guarantee this script just made. A row that cannot clear the filter is
    # genuinely a duplicate of eval material, so it is reported and dropped.
    in_train = {id(r) for r in train_rows}
    in_eval = {id(r) for r in eval_rows}
    orphans = [r for r in merged
               if id(r) not in in_train and id(r) not in in_eval]
    if orphans:
        restored, conflicting = [], 0
        for r in orphans:
            s = shingles(r["output"], n)
            if s and max_overlap(s, ev_inv, ev_sh_list, idf,
                                 symmetric=True)[0] > scfg["max_overlap"]:
                conflicting += 1
                continue
            restored.append(r)
        if restored:
            train_rows += restored
            random.shuffle(train_rows)
        print(f"[assemble] {len(orphans)} row(s) matched neither split; "
              f"returned {len(restored)} to train"
              + (f", dropped {conflicting} that duplicate eval material"
                 if conflicting else ""),
              file=sys.stderr)

    # Recompute against the final train rows and report the true number.
    idf = build_idf([r.get("output") or "" for r in train_rows + eval_rows], n)
    tr_sh_list = [shingles(r.get("output") or "", n) for r in train_rows]
    tr_inv = build_overlap_index(tr_sh_list)

    worst = 0.0
    for r in eval_rows:
        s = shingles(r["output"], n)
        if s:
            worst = max(worst, max_overlap(s, tr_inv, tr_sh_list, idf,
                                           symmetric=True)[0])
    print(f"[assemble] max eval/train answer overlap: {worst:.3f} "
          f"(limit {scfg['max_overlap']})")
    if worst > scfg["max_overlap"]:
        # Every candidate train row conflicted with some eval row. The eval
        # set is what the user needs, so keep it and say plainly that the
        # corpus cannot be split at this limit.
        print(f"[assemble] WARNING: no train row can be separated from the "
              f"eval set at max_overlap={scfg['max_overlap']}. The corpus "
              f"restates the same facts across documents; lower "
              f"split.max_overlap, or hold out more source documents.",
              file=sys.stderr)
    print(f"[assemble] TRAIN {len(train_rows)} {summarize_counts(train_rows)}")
    print(f"[assemble] EVAL  {len(eval_rows)} {summarize_counts(eval_rows)}")

    write_jsonl_chat(os.path.join(out_dir, "train.jsonl"), train_rows)
    write_csv(os.path.join(out_dir, "train.csv"), train_rows)
    write_jsonl_chat(os.path.join(out_dir, "eval.jsonl"), eval_rows)
    write_csv(os.path.join(out_dir, "eval.csv"), eval_rows)
    print(f"[assemble] wrote train.jsonl/csv and eval.jsonl/csv to {out_dir}")


if __name__ == "__main__":
    main()
