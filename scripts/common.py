"""Shared utilities for the instruction-dataset-builder pipeline."""
import hashlib
import json
import os
import re
import collections

# --------------------------------------------------------------------- config

DEFAULT_CONFIG = {
    "input": "./docs",
    "output": "./dataset",
    "subjects": ["General"],
    "subject_map": {},          # {"Math/": "Math", "Science/": "Science"}
    "extract": {
        "pages": 4,             # pages per PDF to read
        "workers": 4,           # process-pool size
        "min_chars": 300,       # drop documents with less text than this
        "extensions": [".pdf", ".docx", ".txt", ".md"],
    },
    "atoms": {
        "min_sentence": 30,     # characters
        "max_sentence": 400,
        "min_words": 5,
        "require_complete": True,   # drop mid-sentence fragments (quality vs yield)
    },
    "qa": {
        "target": 20000,
        "per_source_cap": 220,
        "per_atom": 2,
        "cloze_every": 3,       # every Nth atom also yields a fill-in-the-blank item
        "answer_min": 40,
        "answer_max": 700,
    },
    "split": {
        "eval_frac": 0.07,
        "max_overlap": 0.10,
        "subject_caps": {"Math": 12000, "Science": 12000, "History": 6000},
        "shingle_n": 5,
    },
    "llm": {
        "enabled": False,
        "endpoint": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "model": "auto",
        "workers": 3,
        "timeout": 240,
        "max_tokens": 1500,
    },
}


def load_config(path, overrides=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


# --------------------------------------------------------------------- text

NOISE_PATTERNS = [
    r"Prashant\s*Kirad", r"Shobhit\s*Nirwan", r"Created with\s+\S+",
    r"Downloaded from", r"All rights reserved", r"www\.[a-z0-9\.\-]+",
    r"©\s*\d{4}", r"Page\s+\d+", r"\bVol\.\s*\d+\b", r"\bISBN\b[^\n]*",
    r"^\s*Page\s+\d+\s*$",
]

ARTIFACT_PATTERNS = [
    r"\.\s*\.\s*", r"\(\s*\)", r"\bTable\s+\d+", r"^\s*Fig(ure)?\.?\s*\d+",
    r"\bEx\.\s*\d+[a-z]?\b", r"\bQ\.\s*\d+", r"\bAns\.?\s*\d+",
    r"[\uFFFD]+", r"\s+[|/\\]\s+",
]


def clean_text(text, extra_noise=()):
    for p in list(NOISE_PATTERNS) + list(extra_noise):
        text = re.sub(p, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # de-hyphenate across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def unwrap(text, max_line=110):
    """Re-join PDF hard-wrapped lines into logical sentences.

    PDF extraction emits one visual line per row, so a sentence is often split
    across several lines. Joining them before sentence splitting is what stops
    mid-sentence fragments from becoming training questions.

    Short lines are treated as headings and left alone; bullet and numbered
    lines always start a new unit so lists do not collapse into one blob.
    """
    BULLET_START = re.compile(
        r"^[\(\)\-\*\u2022\u25CF\u25AA\u25A0\u00B7\u2023\u25AA\u2219]|^[a-z0-9]{1,3}[.)]\s+"
        r"|^\d+[.)]\s+|^[A-Z][.)]\s+")
    out = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        if not buf:
            buf = line
            continue
        # previous line ends a sentence -> new unit
        if re.search(r"[.!?:;]$", buf) or len(buf) > max_line:
            flush()
            buf = line
            continue
        # a bullet / numbered item always starts a new unit
        if BULLET_START.match(line):
            flush()
            buf = line
            continue
        # a short previous line is a heading, not a wrapped sentence
        if len(buf) < 45:
            flush()
            buf = line
            continue
        # the new line looks like a new heading (Title Case with few words)
        if len(line.split()) <= 6 and line[:1].isupper() and not buf.endswith((",", "-")):
            flush()
            buf = line
            continue
        if buf.endswith("-"):
            buf = buf[:-1] + line
        else:
            buf = buf + " " + line
    flush()
    return "\n".join(out)


def clean_answer(a):
    """Turn a raw sentence into a clean, self-contained answer."""
    a = a.strip()
    for p in ARTIFACT_PATTERNS:
        a = re.sub(p, " ", a)
    a = re.sub(r"^[\s\-•●■▪*o\d\.\)\(]{0,8}(?=[A-Za-z])", "", a)
    a = re.sub(r"\s{2,}", " ", a)
    a = re.sub(r"\s+([,.;:)])", r"\1", a)
    a = re.sub(r"([(])\s+", r"\1", a)
    a = a.strip()
    if a and not a.endswith((".", "!", "?", ":")):
        a += "."
    return a


def is_bad_answer(a, min_len=40, max_len=700):
    if len(a) < min_len or len(a) > max_len:
        return True
    if len(a.split()) < 7:
        return True
    if len(re.findall(r"\d", a)) / max(len(a), 1) > 0.28:
        return True
    if sum(c.isalpha() or c.isspace() for c in a) / len(a) < 0.72:
        return True
    if re.match(r"^(and|or|but|so|then|which|that|whose|whom|than|also|however|therefore|thus)\b",
               a, re.I):
        return True
    if re.search(r"(.)\1{6,}", a):
        return True
    if re.search(r"[a-zA-Z]{2,}\d{2,}[a-zA-Z]", a):
        return True
    return False


def norm_key(text):
    """Normalized form used for dedupe keys."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def shingles(text, n=5):
    """Set of n-gram fingerprints used for near-duplicate detection."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    if len(w) < n:
        return {md5(" ".join(w))} if w else set()
    return {md5(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


# --------------------------------------------------------------------- io

def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def chatml(user, assistant):
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>")


def split_chat(text):
    user = ans = ""
    for part in text.split("<|im_end|>"):
        if "<|im_start|>user" in part:
            user = part.split("<|im_start|>user", 1)[1].strip()
        elif "<|im_start|>assistant" in part:
            ans = part.split("<|im_start|>assistant", 1)[1].strip()
    clean = lambda s: s.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    return clean(user), clean(ans)


def subject_for(rel_path, subject_map, default="General"):
    for prefix, subj in (subject_map or {}).items():
        if rel_path.startswith(prefix) or prefix.lower() in rel_path.lower():
            return subj
    top = rel_path.replace("\\", "/").split("/")[0]
    return subject_map.get(top, default)


def summarize_counts(rows, key="subject"):
    return dict(collections.Counter(r.get(key, "General") for r in rows))
