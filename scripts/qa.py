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
    "Explain the difference involving {t}.",
    "How is {t} different from the alternative described?",
    "Compare {t} with the other option given here.",
]
CAUSE_T = [
    "Why does {t} happen as described?",
    "What causes {t}?",
    "What is the reason behind {t}?",
    "Explain why {t} occurs.",
]
PROCESS_T = [
    "Describe the process of {t}.",
    "How is {t} carried out?",
    "What are the steps involved in {t}?",
    "Explain the procedure for {t}.",
]
TEMPORAL_T = [
    "When did {t} happen and what was it?",
    "Give the historical detail about {t}.",
    "State the time and event of {t}.",
]
EXAMPLE_T = [
    "Give an example of {t}.",
    "What example illustrates {t}?",
    "Illustrate {t} with the example given.",
]
RULE_T = [
    "State the rule that applies to {t}.",
    "What condition applies to {t}?",
    "Explain the condition described for {t}.",
]
PURPOSE_T = [
    "What is the purpose of {t}?",
    "Explain what {t} is used for.",
    "State the use of {t}.",
]
FACT_T = [
    "Explain {t}.",
    "What does the source say about {t}?",
    "Elaborate on {t}.",
    "Discuss {t}.",
    "What are the key points about {t}?",
    "Write a short note on {t}.",
]
# a small number of templates intentionally quote the statement (teaches
# summarising / rephrasing) - kept low so most questions do not leak the answer
RESTATE_T = [
    "State the following fact.\n{s}",
    "Explain this point.\n{s}",
    "Answer in one sentence: {s}",
    "Rewrite the following point in your own words.\n{s}",
]

TEMPLATES = {
    "definition": DEFINITION_T, "list": LIST_T, "formula": FORMULA_T,
    "compare": COMPARE_T, "cause": CAUSE_T, "process": PROCESS_T,
    "temporal": TEMPORAL_T, "example": EXAMPLE_T, "rule": RULE_T,
    "purpose": PURPOSE_T, "fact": FACT_T, "restate": RESTATE_T,
}

# templates that embed the statement inline vs. only for definitions
INLINE = {"formula", "compare", "cause", "process", "temporal", "example",
          "rule", "purpose", "fact", "restate"}

# the first verb-ish token marks the end of the subject noun phrase
_TOPIC_SPLIT = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|"
    r"will|would|shall|should|may|might|must|means|refers|denotes|contains|"
    r"consists|includes|forms|gives|produces|requires|helps|allows|shows|"
    r"called|known|used|found|occurred|established|created|become|becomes|"
    r"increase|decrease|remain|stay|come|go|make|take|give|get|use|apply|"
    r"refers to|is called|is known as)\b", re.I)

_LEADING_JUNK = re.compile(
    r"^(the|a|an|this|that|these|those|it|its|there|here|in|on|at|by|for|"
    r"with|from|as|and|or|but|however|therefore|thus|also|now|then|when|"
    r"if|so|because|since|while|although|though|after|before|during|"
    r"according to|as we|we have|you have|in class|note that|it is|this is)\s+",
    re.I)


def topic_of(sentence):
    """Extract the subject noun phrase so a question can ask ABOUT the fact
    without quoting it. Prevents the model from learning to echo the prompt."""
    s = sentence.strip()
    m = _TOPIC_SPLIT.search(s)
    head = s[: m.start()] if m else ""
    head = _LEADING_JUNK.sub("", head).strip(" ,;:-")
    head = re.sub(r"^[^A-Za-z0-9]+", "", head)
    words = head.split()

    def usable(ws):
        if not (2 <= len(ws) <= 9):
            return False
        joined = " ".join(ws)
        if "," in joined:
            return False
        letters = sum(c.isalpha() for c in joined)
        if letters / max(len(joined), 1) < 0.8:
            return False
        if ws[0].lower() in ("you", "we", "it", "they", "he", "she", "there",
                             "this", "that", "these", "those", "what", "which",
                             "who", "when", "where", "why", "how", "now", "can",
                             "do", "does", "did", "is", "are", "in", "fact"):
            return False
        if joined.lower().startswith(("can you", "do you", "did you", "what is",
                                      "what are", "how do", "how can", "why do")):
            return False
        return True

    if not usable(words):
        # fall back to the leading words of the statement itself
        words = " ".join(s.split()[:5]).strip(" ,;:-").split()
        if not usable(words):
            return "this topic"
    return " ".join(words)


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

    out = []          # list of (question, answer, is_restate)
    if kind == "definition" and concept:
        i = rot["d"] % len(DEFINITION_T); rot["d"] += 1
        out.append((DEFINITION_T[i].format(c=concept), f"{concept} is {short}.", False))
        i = (rot["d2"] * 3 + 1) % len(DEFINITION_T); rot["d2"] += 1
        if i != rot.get("last_d", -1):
            out.append((DEFINITION_T[i].format(c=concept), f"{concept} is {short}.", False))
    elif kind == "list" and concept:
        i = rot["l"] % len(LIST_T); rot["l"] += 1
        q, lead = LIST_PAIRS[i]
        out.append((q.format(c=concept), f"{lead.format(c=concept)} {short}.", False))
    else:
        # topic-based question: asks ABOUT the fact without quoting it
        topic = topic_of(short)
        tpls = TEMPLATES.get(kind, FACT_T)
        i = rot["g"] % len(tpls); rot["g"] += 1
        out.append((tpls[i].format(t=topic, s=short, c=concept), answer,
                    kind == "formula"))
        if len(out) < cfg.get("per_atom", 2):
            j = (rot["g2"] * 5 + 2) % len(tpls); rot["g2"] += 1
            if j != i:
                out.append((tpls[j].format(t=topic, s=short, c=concept), answer,
                            kind == "formula"))
        # a minority of items deliberately quote the statement (paraphrase training)
        if rot["r"] % 6 == 0:
            k = rot["r2"] % len(RESTATE_T); rot["r2"] += 1
            out.append((RESTATE_T[k].format(s=short), answer, True))
    rot["r"] += 1

    if rot["c"] % max(cfg.get("cloze_every", 3), 1) == 0 and kind != "definition":
        cz = cloze(short)
        if cz:
            out.append((f"Fill in the blank:\n{cz[0]}", cz[1], False))
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
            for q, a, restate in build_for_atom(atom, atom["subject"], rot, qcfg):
                ik = q.lower()[:150]
                ak = md5(norm_key(a))
                if ik in seen_i or ak in seen_a:
                    continue
                seen_i.add(ik); seen_a.add(ak)
                rows.append({"instruction": q, "input": "", "output": a,
                             "subject": atom["subject"], "source": atom["source"],
                             "kind": "restate" if restate else atom["kind"],
                             "origin": "atoms"})
                per_source[atom["source"]] += 1

    if cfg["llm"].get("enabled"):
        import threading as _t
        lock = _t.Lock()
        corpus = read_json(os.path.join(args.output, "corpus.json"), {}) or {}
        # map each source file to the subject its atoms were tagged with
        subj_of = {}
        for a in atoms:
            subj_of.setdefault(a["source"], a["subject"])
        default_subj = atoms[0]["subject"] if atoms else "General"
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
            subj = subj_of.get(rel, default_subj)
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
