#!/usr/bin/env python3
"""Stage 3 - turn knowledge atoms into diverse, grounded QA pairs.

Templates are selected round-robin per atom kind so the dataset does not
collapse into a single repeated question form. Answers stay verbatim from the
extracted source text. An optional LLM backend rewrites only the QUESTION.

Usage:
  python qa.py --output ./dataset [--target 20000] [--config config.json]
"""
import argparse
import collections
import json
import os
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (clean_answer, is_bad_answer, load_config, md5,  # noqa: E402
                    norm_key, read_jsonl, summarize_counts, write_jsonl)

# ------------------------------------------------------------------ templates

DEFINITION_T = [
    "Define {c}.",
    "What is {c}?",
    "Give the definition of {c}.",
    "\"{c}\" - what does this term mean?",
    "Explain {c} in one or two sentences.",
    "What do you understand by {c}?",
    "Write a short note on {c}.",
]
LIST_T = [
    "What does {c} include?",
    "List the components of {c}.",
    "Explain the parts of {c}.",
    "State the items covered under {c}.",
    "Name the elements of {c}.",
]
# (question template, answer lead-in) pairs - kept separate so answers read naturally
LIST_PAIRS = [
    ("What does {c} include?", "{c} includes"),
    ("List the components of {c}.", "The components of {c} are"),
    ("Explain the parts of {c}.", "The parts of {c} are"),
    ("State the items covered under {c}.", "{c} covers"),
    ("Name the elements of {c}.", "The elements of {c} are"),
]
FORMULA_T = [
    "State the relation: {s}",
    "Write down the formula given here: {s}",
    "Which formula is written here: {s}?",
    "State the formula and the quantity it gives.",
]
COMPARE_T = [
    "Explain the difference stated here: {s}",
    "What contrast is being made in this statement?\n{s}",
    "Compare the two things mentioned: {s}",
    "Point out the difference mentioned below.\n{s}",
]
CAUSE_T = [
    "Why is it that {s}?",
    "Give the reason for the following.\n{s}",
    "Explain the cause-and-effect relationship below.\n{s}",
    "What leads to the outcome described here?\n{s}",
]
PROCESS_T = [
    "Describe the process.\n{s}",
    "What are the steps involved here?\n{s}",
    "Explain the procedure.\n{s}",
    "How is this carried out?\n{s}",
]
TEMPORAL_T = [
    "When did the following happen and what was it?\n{s}",
    "Give the historical detail.\n{s}",
    "State the time and event described.\n{s}",
]
EXAMPLE_T = [
    "Give an example from the source that illustrates the point.\n{s}",
    "What example is provided here?\n{s}",
    "Illustrate the idea with the example given.\n{s}",
]
RULE_T = [
    "State the rule given here.\n{s}",
    "What condition or rule is mentioned?\n{s}",
    "Explain the condition described.\n{s}",
]
PURPOSE_T = [
    "What is the purpose described here?\n{s}",
    "Explain what this is used for.\n{s}",
    "State the use of the following.\n{s}",
]
FACT_T = [
    "State the following fact.\n{s}",
    "Explain this point.\n{s}",
    "What does the given statement convey?\n{s}",
    "Expand on the following statement.\n{s}",
    "Answer in one sentence: {s}",
    "Rewrite the following point in your own words.\n{s}",
]

TEMPLATES = {
    "definition": DEFINITION_T, "list": LIST_T, "formula": FORMULA_T,
    "compare": COMPARE_T, "cause": CAUSE_T, "process": PROCESS_T,
    "temporal": TEMPORAL_T, "example": EXAMPLE_T, "rule": RULE_T,
    "purpose": PURPOSE_T, "fact": FACT_T,
}

# templates that embed the statement inline vs. only for definitions
INLINE = {"formula", "compare", "cause", "process", "temporal", "example",
          "rule", "purpose", "fact"}


def cloze(statement):
    """Blank the longest capitalised content word -> fill-in-the-blank."""
    words = statement.split()
    cands = [(len(w), i) for i, w in enumerate(words)
             if len(w) > 6 and w[0].isupper() and w.isalpha()]
    if not cands:
        return None
    _, idx = max(cands)
    return (" ".join(words[:idx] + ["________"] + words[idx + 1:]),
            " ".join(words[:idx] + [words[idx]] + words[idx + 1:]))


def build_for_atom(atom, subject, rot, cfg):
    kind = atom["kind"]
    concept = atom.get("concept", "")
    stmt = atom.get("statement", "")
    answer = clean_answer(stmt)
    if not answer or is_bad_answer(answer, cfg["answer_min"], cfg["answer_max"]):
        return []
    short = answer[:-1] if answer.endswith(".") else answer
    tpls = TEMPLATES.get(kind, FACT_T)

    out = []
    if kind == "definition" and concept:
        i = rot["d"] % len(DEFINITION_T); rot["d"] += 1
        out.append((DEFINITION_T[i].format(c=concept), f"{concept} is {short}."))
        i = (rot["d2"] * 3 + 1) % len(DEFINITION_T); rot["d2"] += 1
        if i != rot.get("last_d", -1):
            out.append((DEFINITION_T[i].format(c=concept), f"{concept} is {short}."))
    elif kind == "list" and concept:
        i = rot["l"] % len(LIST_T); rot["l"] += 1
        q, lead = LIST_PAIRS[i]
        out.append((q.format(c=concept), f"{lead} {short}."))
    else:
        i = rot["g"] % len(tpls); rot["g"] += 1
        t = tpls[i]
        q = t.format(c=concept, s=short)
        out.append((q, answer))
        if len(out) < cfg.get("per_atom", 2):
            j = (rot["g2"] * 5 + 2) % len(tpls); rot["g2"] += 1
            if j != i:
                q2 = tpls[j].format(c=concept, s=short)
                out.append((q2, answer))

    if rot["c"] % max(cfg.get("cloze_every", 3), 1) == 0 and kind != "definition":
        cz = cloze(short)
        if cz:
            out.append((f"Fill in the blank:\n{cz[0]}", cz[1]))
    rot["c"] += 1
    return out


# ------------------------------------------------------------------ LLM backend

LLM_PROMPT = """You are building a tutoring dataset from a teacher's notes.
Write {n} DIFFERENT question-answer pairs using ONLY the source text below.

Output one JSON object per line, exactly:
{{"text":"<|im_start|>user\\nQUESTION<|im_end|>\\n<|im_start|>assistant\\nANSWER<|im_end|>"}}

Rules:
- Output ONLY JSON lines. No markdown fences, no commentary.
- Each QUESTION must be a natural exam question (mix: "Define X", "Explain why...",
  "State the formula for...", "Compare A and B", "List the ...", "Give an example of...").
- Each ANSWER must be complete, accurate and grounded ONLY in the source text.
- Do not repeat the same question wording across pairs.
- Never mention "source text" or "notes" in the question.

SOURCE TEXT:
{chunk}
"""


def llm_call(chunk, subject, n, cfg):
    llm = cfg["llm"]
    payload = json.dumps({
        "model": llm.get("model", "auto"),
        "messages": [
            {"role": "system", "content": "Output only JSON lines in ChatML format."},
            {"role": "user", "content": LLM_PROMPT.format(n=n, subject=subject, chunk=chunk)},
        ],
        "max_tokens": llm.get("max_tokens", 1500),
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        llm["endpoint"].rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {llm.get('api_key', '')}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=llm.get("timeout", 240)) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    texts = []
    for raw in content.split("\n"):
        raw = raw.strip().strip("`")
        if not raw.startswith("{"):
            s = raw.find('{"text"'); e = raw.rfind("}")
            if s != -1 and e > s:
                raw = raw[s:e + 1]
            else:
                continue
        try:
            o = json.loads(raw.rstrip(","))
        except Exception:
            continue
        t = o.get("text")
        if isinstance(t, str) and t.count("<|im_start|>") >= 2:
            texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser(description="Generate QA pairs from knowledge atoms")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--target", type=int)
    ap.add_argument("--per-source-cap", type=int)
    ap.add_argument("--llm", action="store_true", help="force LLM backend on")
    ap.add_argument("--llm-only", action="store_true", help="only LLM-generated rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.target:
        cfg["qa"]["target"] = args.target
    if args.per_source_cap:
        cfg["qa"]["per_source_cap"] = args.per_source_cap
    if args.llm:
        cfg["llm"]["enabled"] = True

    atoms = read_jsonl(os.path.join(args.output, "atoms.jsonl"))
    if not atoms:
        print("atoms.jsonl not found - run atoms.py first", file=sys.stderr)
        sys.exit(1)

    rows = []
    seen_i, seen_a = set(), set()
    per_source = collections.Counter()
    rot = collections.defaultdict(int)
    qcfg = cfg["qa"]

    if not args.llm_only:
        for atom in atoms:
            if len(rows) >= qcfg["target"]:
                break
            if per_source[atom["source"]] >= qcfg["per_source_cap"]:
                continue
            for q, a in build_for_atom(atom, atom["subject"], rot, qcfg):
                ik = q.lower()[:150]
                ak = md5(norm_key(a))
                if ik in seen_i or ak in seen_a:
                    continue
                seen_i.add(ik); seen_a.add(ak)
                rows.append({"instruction": q, "input": "", "output": a,
                             "subject": atom["subject"], "source": atom["source"],
                             "kind": atom["kind"], "origin": "atoms"})
                per_source[atom["source"]] += 1

    if cfg["llm"].get("enabled"):
        import threading as _t
        lock = _t.Lock()
        corpus = read_json(os.path.join(args.output, "corpus.json"), {}) or {}
        chunks = []
        for rel, text in corpus.items():
            if not text or len(text) < 400:
                continue
            paras = [x.strip() for x in text.split("\n") if x.strip()]
            cur, n = [], 0
            for para in paras:
                if n + len(para) > 1000 and cur:
                    chunks.append((rel, " ".join(cur)))
                    cur, n = [], 0
                cur.append(para); n += len(para) + 1
            if cur:
                chunks.append((rel, " ".join(cur)))
        print(f"[qa] LLM pass over {len(chunks)} chunks", flush=True)

        def run_chunk(item):
            rel, chunk = item
            subj = atoms[0]["subject"] if atoms else "General"
            for t in llm_call(chunk, subj, 3, cfg):
                u, a = t.split("<|im_start|>user", 1)[-1].split("<|im_end|>")[0], \
                       t.split("<|im_start|>assistant", 1)[-1].split("<|im_end|>")[0]
                with lock:
                    ik = u.lower()[:150]
                    ak = md5(norm_key(a))
                    if ik in seen_i or ak in seen_a:
                        continue
                    seen_i.add(ik); seen_a.add(ak)
                    rows.append({"instruction": u.strip(), "input": "", "output": a.strip(),
                                 "subject": subj, "source": rel, "kind": "llm",
                                 "origin": "llm"})

        with ThreadPoolExecutor(max_workers=cfg["llm"].get("workers", 3)) as ex:
            futs = [ex.submit(run_chunk, c) for c in chunks]
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if i % 25 == 0:
                    print(f"  llm {i}/{len(chunks)} | rows={len(rows)}", flush=True)

    out = os.path.join(args.output, "qa_all.jsonl")
    write_jsonl(out, rows)
    print(f"[qa] {len(rows)} QA pairs")
    print(f"[qa] by subject: {summarize_counts(rows)}")
    print(f"[qa] by kind: {dict(collections.Counter(r.get('kind') for r in rows))}")
    print(f"[qa] sources used: {len(per_source)}")
    print(f"[qa] -> {out}")


if __name__ == "__main__":
    main()
