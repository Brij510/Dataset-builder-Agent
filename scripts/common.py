"""Shared utilities for the dataset-builder pipeline."""
import collections
import hashlib
import json
import math
import os
import re

# --------------------------------------------------------------------- config

DEFAULT_CONFIG = {
    "input": "./docs",
    "output": "./dataset",
    "subjects": ["General"],   # labels the auto-detector may choose from
    "subject_map": {},          # {"Math/": "Math", "Science/": "Science"}
    "auto_subject": True,       # infer a subject from content when no map matches
    "subject_keywords": {},      # extra label -> keyword list for auto-detection
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
        "max_per_answer": 1,     # how many questions may share one answer
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


def _deep_merge(base, override):
    """Merge override into base without dropping sibling keys."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                user = json.load(f)
            except json.JSONDecodeError as e:
                raise SystemExit(f"config file {path} is not valid JSON: {e}")
        if not isinstance(user, dict):
            raise SystemExit(f"config file {path} must contain a JSON object")
        cfg = _deep_merge(cfg, user)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg, path=""):
    """Fail loudly on nonsense configuration instead of half-running."""
    problems = []
    for key in ("eval_frac", "max_overlap"):
        v = cfg["split"].get(key)
        if not isinstance(v, (int, float)):
            problems.append(f"split.{key} must be a number, got {v!r}")
        elif not 0 <= v <= 1:
            problems.append(f"split.{key} must be between 0 and 1, got {v}")
    if not isinstance(cfg["qa"].get("target"), int) or cfg["qa"]["target"] < 0:
        problems.append(f"qa.target must be a non-negative integer, got {cfg['qa'].get('target')!r}")
    for key in ("min_sentence", "max_sentence"):
        if cfg["atoms"].get(key, 0) <= 0:
            problems.append(f"atoms.{key} must be positive")
    if cfg["atoms"].get("min_sentence", 0) > cfg["atoms"].get("max_sentence", 0):
        problems.append("atoms.min_sentence must not exceed atoms.max_sentence")
    if cfg["extract"].get("pages", 1) <= 0:
        problems.append("extract.pages must be positive")
    if cfg["split"].get("shingle_n", 5) < 2:
        problems.append("split.shingle_n must be at least 2")
    if problems:
        msg = "invalid configuration" + (f" in {path}" if path else "") + ":\n  - " + "\n  - ".join(problems)
        raise SystemExit(msg)


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
    # answer-key section headers that survive extraction and end up in a
    # question: "State the relation: ANSWERS/HINTS 251 (ii) ..."
    r"\bANSWERS?\s*/\s*HINTS?\b\s*\d*", r"^\s*HINTS?\s+\d+\s*",
]


def clean_text(text, extra_noise=()):
    if not text:
        return ""
    text = text.replace("\ufeff", "")                          # strip UTF-8 BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")      # normalise line endings
    text = text.replace("\u00a0", " ")                         # non-breaking space
    for p in list(NOISE_PATTERNS) + list(extra_noise):
        text = re.sub(p, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # de-hyphenate across lines
    # PDF extraction often glues words together after punctuation
    text = re.sub(r",(?=[a-z])", ", ", text)
    text = re.sub(r"\.(?=[a-z]{2,}\b)", ". ", text)
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
    # a line ending in one of these is grammatically unfinished, so the next
    # line continues it even if the next line is short
    DANGLING = re.compile(
        r"\b(its|the|a|an|of|in|on|at|to|for|and|or|but|is|are|was|were|"
        r"be|been|has|have|had|as|by|with|from|that|which|this|these|those|"
        r"called|known|termed|referred|known as|known as the|such as|"
        r"called its|known as its|including|except|besides|along with|"
        r"consists of|comprises|contains|provides|gives|per|via)\s*$", re.I)
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
        if re.search(r"[.!?:;]$", buf) and not DANGLING.search(buf):
            flush()
            buf = line
            continue
        if len(buf) > max_line:
            flush()
            buf = line
            continue
        # a bullet / numbered item always starts a new unit
        if BULLET_START.match(line):
            flush()
            buf = line
            continue
        # the previous line is grammatically unfinished -> always join
        if DANGLING.search(buf):
            if buf.endswith("-"):
                buf = buf[:-1] + line
            else:
                buf = buf + " " + line
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


def build_idf(texts, n=5):
    """Inverse document frequency over answer shingles.

    Boilerplate shared by many answers ("is defined as", "is a unit of") must
    not count as leakage - only distinctive content should. IDF weighting does
    exactly that: a shingle that appears in one document weighs far more than
    one that appears in every document.

    The smooth form `log((N+1)/(df+1))` is used rather than `log(N/df)+1`
    because the latter never reaches zero. A shingle present in every answer
    still carries a full unit of weight, and since a 5-gram window over a
    shared template produces many such shingles, the boilerplate outweighs
    the distinctive content and the measure reports leakage where there is
    none. Here a universally common shingle weighs exactly nothing.
    """
    df = collections.Counter()
    n_docs = 0
    for t in texts:
        n_docs += 1
        df.update(set(shingles(t or "", n)))
    if not n_docs:
        return {}
    return {g: math.log((n_docs + 1) / (c + 1)) for g, c in df.items()}


def weighted_overlap(a_sh, b_sh, idf):
    """Fraction of a_sh's IDF weight that also appears in b_sh.

    Falls back to the raw shingle ratio when a_sh carries no distinctive
    weight at all. Reporting zero there would hide a genuine duplicate
    behind a corpus made entirely of boilerplate.
    """
    if not a_sh:
        return 0.0
    num = sum(idf.get(g, 0.0) for g in (a_sh & b_sh))
    den = sum(idf.get(g, 0.0) for g in a_sh)
    if den:
        return num / den
    shared = len(a_sh & b_sh)
    return shared / len(a_sh) if a_sh else 0.0


def symmetric_overlap(a_sh, b_sh, idf):
    """Overlap between two shingle sets, measured in both directions.

    One direction is not enough. A short train answer sitting inside a long
    eval answer scores near 1.0 when measured train-to-eval, but near 0.0
    when measured eval-to-train. A leak that only shows up in one direction
    is still a leak, so the worse of the two is the number that counts.
    """
    return max(weighted_overlap(a_sh, b_sh, idf),
               weighted_overlap(b_sh, a_sh, idf))


def build_overlap_index(row_shingles):
    """Invert shingle -> rows, so overlap is checked only against candidates.

    Comparing every row against every other row is quadratic. A row can only
    share a shingle with a row that actually contains that shingle, so the
    inverted index turns the search into a lookup over the few real
    candidates.
    """
    index = collections.defaultdict(list)
    for i, sh in enumerate(row_shingles):
        for g in sh:
            index[g].append(i)
    return index


def max_overlap(a_sh, index, row_shingles, idf, symmetric=True):
    """Highest overlap between a_sh and any indexed row.

    Returns (worst_overlap, other_shingles) so the caller can drop the
    offending row.
    """
    if not a_sh:
        return 0.0, set()
    seen = set()
    worst, other = 0.0, set()
    for g in a_sh:
        for i in index.get(g, ()):
            if i in seen:
                continue
            seen.add(i)
            b_sh = row_shingles[i]
            ov = (symmetric_overlap(a_sh, b_sh, idf) if symmetric
                  else weighted_overlap(a_sh, b_sh, idf))
            if ov > worst:
                worst, other = ov, b_sh
    return worst, other


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


CHATML_MARKER = re.compile(r"<\|im_(?:start|end)\|>")


def sanitize_chatml(text):
    """Neutralise literal ChatML markers that came from the source material.

    A document that itself contains `<|im_end|>` (a tutorial showing ChatML
    examples, for instance) would otherwise inject a stray marker into the
    row and break the format for the whole training file.
    """
    return CHATML_MARKER.sub("", text or "")


def chatml(user, assistant):
    return (f"<|im_start|>user\n{sanitize_chatml(user)}<|im_end|>\n"
            f"<|im_start|>assistant\n{sanitize_chatml(assistant)}<|im_end|>")


def split_chat(text):
    user = ans = ""
    for part in text.split("<|im_end|>"):
        if "<|im_start|>user" in part:
            user = part.split("<|im_start|>user", 1)[1].strip()
        elif "<|im_start|>assistant" in part:
            ans = part.split("<|im_start|>assistant", 1)[1].strip()
    clean = lambda s: s.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    return clean(user), clean(ans)


SUBJECT_KEYWORDS = {
    "Math": ["theorem", "polynomial", "equation", "triangle", "circle", "radius",
             "coordinate", "probability", "trigonometry", "ap", "arithmetic",
             "progression", "surface area", "volume", "algebra", "geometry",
             "quadratic", "factor", "divisor", "real number", "rational",
             "irrational", "pythagoras", "tangent", "sector", "cylinder", "cone",
             "sphere", "matrix", "binomial", "statistics", "mean", "median"],
    "Science": ["atom", "electron", "proton", "neutron", "chemical", "reaction",
                "acid", "base", "salt", "oxidation", "reduction", "photosynthesis",
                "respiration", "cell", "tissue", "organ", "blood", "heart", "brain",
                "genetics", "heredity", "evolution", "magnetic", "electric",
                "current", "voltage", "resistance", "ohm", "reflection", "refraction",
                "lens", "mirror", "force", "motion", "energy", "power", "wave",
                "sound", "light", "ozone", "ecosystem", "food chain", "pollution",
                "metal", "non-metal", "periodic table", "isotope", "solution"],
    "Social Science": ["democracy", "nationalism", "constitution", "parliament",
                        "election", "federalism", "secularism", "caste", "gender",
                        "globalization", "liberalization", "unemployment", "poverty",
                        "population", "development", "sector", "agriculture",
                        "industrialisation", "revolution", "treaty", "empire",
                        "colonial", "print culture", "money and credit", "gdp",
                        "human resources", "water resources", "minerals", "energy"],
    "History": ["empire", "dynasty", "treaty", "revolution", "king", "emperor",
                "war", "battle", "century", "civilisation", "colonial", "independence",
                "mughal", "maurya", "gupta", "bronze age"],
    "Geography": ["latitude", "longitude", "climate", "monsoon", "soil", "river",
                  "plateau", "mountain", "population density", "urbanisation",
                  "resources", "map", "hemisphere", "erosion", "glacier"],
    "Economics": ["gdp", "inflation", "market", "supply", "demand", "monopoly",
                  "tax", "budget", "bank", "loan", "credit", "unemployment",
                  "sector", "national income", "fiscal"],
    "English": ["poem", "novel", "prose", "literary", "character", "narrative",
                "dialogue", "author", "theme", "protagonist"],
    "Computer Science": ["algorithm", "variable", "loop", "array", "function",
                         "python", "java", "programming", "code", "syntax",
                         "database", "sql", "html", "css"],
}


def detect_subject(text, known=None, default="General", min_hits=3, extra_keywords=None):
    """Infer a subject label from document content by keyword scoring.

    Used when no explicit subject_map entry matches. Only a label from
    `known` (or the built-in table) can be returned. `extra_keywords` lets a
    config file add its own labels without editing this module.

    `known` narrows the candidate pool. Pass the labels you actually care
    about; passing a pool that contains none of them (or only the default)
    makes detection impossible, so the default is used for every document.
    """
    if not text:
        return default
    low = text.lower()
    best, best_score = default, 0
    table = dict(SUBJECT_KEYWORDS)
    for name, words in (extra_keywords or {}).items():
        table[name] = list(table.get(name, [])) + list(words)
    pool = {k: v for k, v in table.items() if known is None or k in known}
    for name, words in pool.items():
        score = 0
        for w in words:
            n = low.count(w)
            if n:
                score += min(n, 8) * (2 if len(w) > 5 else 1)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= min_hits else default


def subject_for(rel_path, subject_map, default="General"):
    """Resolve a document's subject from its path.

    The most specific prefix wins, so {"Math/": "Math", "Math/NCERT/": "Math
    Advanced"} maps a file under Math/NCERT/ to "Math Advanced" and never to
    "Math". A prefix only matches at a path boundary - "Math" must not match
    "AppliedMathematics" - and the longest match wins, not the dict order.
    """
    rel = (rel_path or "").replace("\\", "/")
    best_len, best = -1, default
    for prefix, subj in (subject_map or {}).items():
        p = prefix.replace("\\", "/")
        if not p:
            continue
        if not rel.startswith(p):
            continue
        if not p.endswith("/") and len(rel) > len(p) and rel[len(p)] != "/":
            # "Math" must not match "Mathematics"; a prefix that already ends
            # in "/" is unambiguous and needs no boundary check
            continue
        if len(p) > best_len:
            best_len, best = len(p), subj
    if best_len >= 0:
        return best
    top = rel.split("/")[0]
    return (subject_map or {}).get(top, default)


def summarize_counts(rows, key="subject"):
    return dict(collections.Counter(r.get(key, "General") for r in rows))
