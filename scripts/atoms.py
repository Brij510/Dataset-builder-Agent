#!/usr/bin/env python3
"""Stage 2 - mine typed knowledge atoms out of the extracted corpus.

Each atom is a self-contained factual statement classified into a kind that
determines which question archetypes are generated from it later.

Usage:
  python atoms.py --output ./dataset [--subjects Math,Science] [--config config.json]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (read_json, write_jsonl, load_config, subject_for,  # noqa: E402
                    summarize_counts, unwrap, detect_subject)

DEF_RE = re.compile(
    r"^([A-Z][A-Za-z0-9 ,\-\(\)\/\.']{2,55}?)\s+"
    r"(?:is|are|was|were|refers to|means|denotes|is defined as|is called|"
    r"is known as|is the|are the)\s+(.{25,320})$")

LIST_RE = re.compile(r"^([A-Z][A-Za-z0-9 \-]{2,40}):\s+(.{20,300})$")

BAD_CONCEPT = re.compile(
    r"^(you have|we have|in class|as we|it is|this is|there|here|when|if|for|"
    r"also|now|then|however|therefore|thus|note|remember|see|figure|table|"
    r"chapter|unit|page|question|answer|example \d|what|which|who|whom|"
    r"whose|where|why|how|you|we|they|it|he|she|him|her|his|hers|me|my|"
    r"mine|myself|your|our|their|from|with|without|on|at|by|under|over|"
    r"through|during|after|before|between|above|below|into|onto|per|via|"
    r"given|following|above given|the same|all the|any of|one of)\b", re.I)

# a concept containing a clause boundary is a fragment, not a term
BAD_CONCEPT_CLAUSE = re.compile(
    r",\s*(which|who|where|when|that|and|but|is|are|was|were|if|because|"
    r"while|although|though|so)\b", re.I)

CAUSE_RE = re.compile(
    r"\b(because|since|due to|as a result|therefore|so that|this leads to|"
    r"results in|causes)\b", re.I)

COMPARE_RE = re.compile(
    r"\b(different from|unlike|whereas|while|as opposed to|similar to|"
    r"in contrast)\b", re.I)

FORMULA_RE = re.compile(r"[=≤≥≈]|\bformula\b|\btheorem\b|\bidentity\b", re.I)

PROC_RE = re.compile(
    r"\b(process|procedure|steps?|method|is done by|are done by|can be prepared|"
    r"can be made|steps involved)\b", re.I)

TEMP_RE = re.compile(
    r"\b(in \d{4}|on \d{1,2} (?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)|during the (?:revolt|war)|"
    r"\d{1,2}th century)\b", re.I)

TEMP_VERB = re.compile(
    r"\b(was|were|is|are|established|founded|formed|occurred|passed|created|"
    r"introduced|began|launched)\b", re.I)

EXAMPLE_RE = re.compile(
    r"\b(for example|e\.g\.|such as|for instance|like this|such as follows)\b", re.I)

RULE_RE = re.compile(
    r"\b(must|cannot|never|always|only if|is required|is necessary|"
    r"is mandatory|should be)\b", re.I)

PURPOSE_RE = re.compile(
    r"\b(used to|so as to|in order to|helps to|allows|helps|ensures|"
    r"which helps|to prevent)\b", re.I)


INCOMPLETE_TAIL = re.compile(
    r"\b(and|or|but|of|to|in|on|at|by|for|with|from|as|that|which|whose|"
    r"is|are|was|were|be|been|has|have|had|namely|such|than|into|"
    r"between|among|during|through|about|like|when|while|because)\s*[.,;:]?$", re.I)

VERBISH = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|"
    r"will|would|shall|should|may|might|must|means|refers|denotes|contains|"
    r"consists|includes|forms|gives|produces|requires|helps|allows|shows|"
    r"called|known|used|found|occurred|established|created|become|becomes|"
    r"increase|decrease|remain|stay|come|go|make|take|give|get|use|apply)\b", re.I)


# Only unambiguous list markers: a number, a parenthesised label, or a
# roman numeral carrying punctuation. A bare word like "A" or "In" is never
# an enumerator, even though both look like roman numerals.
ENUMERATOR = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[.)]?\s*[,.:]?\s*"                       # 10.1 , 3. , 12)
    r"|\([a-z]{1,4}\)\s*[,.:]?\s*"                            # (i) , (iv)
    r"|(?<![a-z])(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)[.)]\s+"  # iv. , x.
    r")+",
    re.I)


def strip_enumerator(s):
    """Remove a leading section number or list marker and repair the capital.

    '10.1 (i), the line PQ and the circle have no common point.'
    -> 'The line PQ and the circle have no common point.'
    """
    m = ENUMERATOR.match(s)
    if not m:
        return s
    rest = s[m.end():].strip()
    if not rest or not rest[0].isalpha():
        return s
    return rest[0].upper() + rest[1:]


def looks_complete(sent):
    """Reject mid-sentence fragments and dangling clauses."""
    s = sent.strip()
    if not s or not s[0].isupper():
        return False
    if INCOMPLETE_TAIL.search(s):
        return False
    if not VERBISH.search(s) and not FORMULA_RE.search(s):
        return False
    # too many commas relative to length reads like a table fragment
    if s.count(",") > max(2, len(s) // 40):
        return False
    if len(s.split()) < 6:
        return False
    return True


def split_sentences(text, min_len=30, max_len=400):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if min_len <= len(p.strip()) <= max_len]


def classify(sent, cfg):
    """Return (kind, concept, statement) for one sentence."""
    m = DEF_RE.match(sent)
    if m:
        concept, stmt = m.group(1).strip(" .:-'"), m.group(2).strip()
        if (1 <= len(concept.split()) <= 7 and len(stmt) > 25
                and not BAD_CONCEPT.match(concept)
                and not BAD_CONCEPT_CLAUSE.search(concept)
                and sum(c.isalpha() for c in concept) / max(len(concept), 1) > 0.75):
            return "definition", concept, stmt

    ml = LIST_RE.match(sent)
    if ml and len(ml.group(1).split()) <= 6:
        return "list", ml.group(1).strip(), ml.group(2).strip()

    if FORMULA_RE.search(sent):
        return "formula", "", sent
    if COMPARE_RE.search(sent):
        return "compare", "", sent
    if CAUSE_RE.search(sent):
        return "cause", "", sent
    if PROC_RE.search(sent):
        return "process", "", sent
    if TEMP_RE.search(sent) and TEMP_VERB.search(sent):
        return "temporal", "", sent
    if EXAMPLE_RE.search(sent):
        return "example", "", sent
    if RULE_RE.search(sent):
        return "rule", "", sent
    if PURPOSE_RE.search(sent):
        return "purpose", "", sent
    return "fact", "", sent


def main():
    ap = argparse.ArgumentParser(description="Mine knowledge atoms from corpus")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config")
    ap.add_argument("--subjects", default="", help="comma list, default General")
    ap.add_argument("--subject-map", default="", help='json, e.g. {"Math/":"Math"}')
    ap.add_argument("--min-sentence", type=int)
    ap.add_argument("--max-sentence", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.min_sentence:
        cfg["atoms"]["min_sentence"] = args.min_sentence
    if args.max_sentence:
        cfg["atoms"]["max_sentence"] = args.max_sentence

    subject_map = cfg.get("subject_map", {})
    if args.subject_map:
        import json
        subject_map.update(json.loads(args.subject_map))
    default_subject = (args.subjects.split(",")[0].strip() if args.subjects
                       else cfg.get("subjects", ["General"])[0])

    corpus = read_json(os.path.join(args.output, "corpus.json"), {}) or {}
    if not corpus:
        print("corpus.json not found - run extract.py first", file=sys.stderr)
        sys.exit(1)

    total_chars = sum(len(v or "") for v in corpus.values())
    if total_chars > 200_000_000:
        print(f"[atoms] WARNING: corpus holds {total_chars/1e6:.0f}M characters; "
              f"it is fully in memory here. Split it into several corpora and run "
              f"the pipeline per batch, then merge the train/eval files.", flush=True)

    a = cfg["atoms"]
    auto_detect = cfg.get("auto_subject", True)
    known = cfg.get("subjects") or None
    atoms = []
    for rel, text in sorted(corpus.items()):
        if not text:
            continue
        subj = subject_for(rel, subject_map, default_subject)
        if subj == default_subject and auto_detect and len(subject_map) == 0:
            subj = detect_subject(text, known, default_subject,
                                  extra_keywords=cfg.get("subject_keywords"))
        text = unwrap(text)
        for line in (l.strip() for l in text.split("\n")):
            if not (a["min_sentence"] <= len(line) <= a["max_sentence"] * 3):
                continue
            s = re.sub(r"^[\(\)\-\*\u2022\u25CF\u25AA\u25A0\u00B7\d\.\s]{0,6}(?=[A-Z])",
                       "", line)
            if len(s) < a["min_sentence"]:
                continue
            for sent in (split_sentences(s, a["min_sentence"], a["max_sentence"])
                         if len(s) > a["min_sentence"] * 2 else [s]):
                sent = strip_enumerator(sent)
                if len(sent.split()) < a["min_words"]:
                    continue
                if a.get("require_complete", True) and not looks_complete(sent):
                    continue
                kind, concept, stmt = classify(sent, cfg)
                # a source sentence that is itself a question is not a fact
                if stmt.rstrip().endswith("?"):
                    continue
                atoms.append({"kind": kind, "concept": concept, "statement": stmt,
                              "subject": subj, "source": rel})

    out = os.path.join(args.output, "atoms.jsonl")
    write_jsonl(out, atoms)
    kinds = {}
    for at in atoms:
        kinds[at["kind"]] = kinds.get(at["kind"], 0) + 1
    print(f"[atoms] {len(atoms)} atoms from {len(corpus)} documents")
    print(f"[atoms] by kind: {kinds}")
    print(f"[atoms] by subject: {summarize_counts(atoms)}")
    print(f"[atoms] -> {out}")


if __name__ == "__main__":
    main()
