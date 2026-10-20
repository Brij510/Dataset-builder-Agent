---
name: "dataset-builder"
description: "Builds high-quality instruction-tuning datasets (train/eval JSONL + CSV) from raw documents - PDFs, DOCX, TXT, Markdown - by extracting text, mining knowledge atoms (definitions, formulas, comparisons, processes, causes, lists), generating diverse grounded QA pairs, splitting with leakage control, and verifying that every answer is grounded in its source. Use when creating SFT or fine-tuning data for LLMs from textbooks, notes, question banks, legal or medical corpora, or when asked to turn documents into a training dataset, build a RAG/eval set, or scale QA generation to 10k+ rows."
---

# Dataset Builder

## What This Skill Does

Converts a folder of raw documents into a production-ready instruction-tuning dataset with a
train/eval split that cannot leak. The pipeline is seven stages:

```
raw docs -> [1] extract -> [2] dedupe -> [3] atoms -> [4] QA -> [5] split -> [6] validate -> [7] ground
```

Every stage is standalone, resumable and cached. Nothing is fabricated: each answer is grounded
in text that was actually extracted from a source document, and every row keeps a `source` pointer
back to the file it came from.

## When not to use this

- The source is **image-only scans** - run OCR (`ocrmypdf`) first, this pipeline
  reads text layers only.
- You need answers the source does not contain. This pipeline never invents
  facts; if the corpus lacks an answer, the answer is not in the dataset.
- The task is retrieval, not instruction tuning. For a RAG index, keep the
  chunks and skip stages 3-5.
- The corpus is a handful of files and a hand-written prompt fits better.

## Prerequisites

- Python 3.9+
- `pdfplumber` (PDFs), `python-docx` (DOCX) - run `scripts/setup.sh`
- Optional: an OpenAI-compatible endpoint for LLM-assisted question phrasing
  (`llm.endpoint` + `llm.api_key` in the config)

## Quick Start

```bash
# One-shot: extract -> dedupe -> atoms -> qa -> split -> validate -> ground
python scripts/pipeline.py --input ./my_docs --output ./dataset --config config/config.example.json

# Or stage by stage (each is resumable and cached)
python scripts/extract.py  --input ./my_docs --output ./dataset
python scripts/dedupe.py   --output ./dataset
python scripts/atoms.py    --output ./dataset --subjects Math,Science
python scripts/qa.py       --output ./dataset --target 20000
python scripts/assemble.py --output ./dataset --eval-frac 0.07
python scripts/validate.py --output ./dataset
python scripts/ground.py   --output ./dataset
```

Outputs in `./dataset`:

| File | Purpose |
|---|---|
| `train.jsonl` / `train.csv` | ChatML training rows |
| `eval.jsonl` / `eval.csv` | Held-out evaluation rows (zero leakage) |
| `corpus.json` | Cached extracted text per source file |
| `atoms.jsonl` | Mined knowledge atoms |
| `qa_all.jsonl` | All generated QA before splitting |
| `train_grounded.jsonl` / `eval_grounded.jsonl` | Optional: rows that pass a grounding threshold |

## Step-by-Step Guide

### 1. Extract text from raw documents

```bash
python scripts/extract.py --input ./my_docs --output ./dataset --pages 4 --workers 4
```

- Walks the input tree, reads `.pdf`, `.docx`, `.txt`, `.md`
- Writes `corpus.json`: `{relative_path: cleaned_text}`
- **Uses processes, not threads** - PDF parsing is GIL-bound, threads give no speedup
- Caches per file and is **resumable**: rerun after adding files, only new ones are parsed
- Reconstructs **text blocks** so boxed sidebars and two-column pages are not
  spliced into each other (this is where a large share of lost content and
  corrupted sentences come from)
- Strips watermarks, page numbers, BOM, CRLF, de-hyphenates line breaks, drops noise-only lines
- `--limit N` processes only the first N documents (quick trials)
- A corrupt file is skipped, not fatal

### 2. Remove near-duplicate documents

```bash
python scripts/dedupe.py --output ./dataset
```

A textbook and its scan produce identical content. Left in place they waste generation effort and
break the document-holdout split, because the model would see the same material in both halves
even though the files differ. Detection uses Jaccard similarity over 5-gram
shingles (not containment, which flags any two documents that share
boilerplate); the second copy of a >75% pair is dropped.
`corpus.json.pre-dedupe` keeps a backup.

### 3. Mine knowledge atoms

```bash
python scripts/atoms.py --output ./dataset --subjects Math,Science,History
```

Classifies each sentence into a typed atom:

| Kind | Detected by | Question style produced |
|---|---|---|
| `definition` | `X is/refers to/means ...` | Define, What is, Explain the term |
| `formula` | contains `=`, `theorem`, `formula` | State the formula, Which formula |
| `list` | `Heading: item, item, item` | List the components, What does it include |
| `compare` | `unlike`, `whereas`, `in contrast` | Explain the difference, Compare |
| `cause` | `because`, `leads to`, `results in` | Why..., Give the reason |
| `process` | `process`, `steps`, `procedure` | Describe the process, Steps involved |
| `temporal` | `in 1947`, `on 15 August` | When did..., Historical detail |
| `example` | `for example`, `e.g.`, `such as` | Give an example of... |
| `rule` | `must`, `cannot`, `never`, `is required` | State the rule, What condition... |
| `purpose` | `used to`, `in order to`, `helps to` | What is the purpose of... |
| `fact` | fallback | Explain the topic, Elaborate on... |

Also emits `cloze` (fill-in-the-blank) items for a configurable fraction of atoms.

Subjects come from `subject_map` (path prefix -> label). With no map match, `auto_subject: true`
infers the label from document content; add your own labels with `subject_keywords` in the config.

### 4. Generate QA pairs

```bash
python scripts/qa.py --output ./dataset --target 20000 --per-source-cap 220
```

- Multiple archetypes per atom, selected **round-robin** so the dataset does not collapse
  into one repeated template
- Questions ask **about a topic** extracted from the statement, never quote the answer
- Per-source cap prevents one large textbook from flooding the dataset
- Dedupe on normalized instruction, and on normalized answer bounded by
  `max_per_answer` (default 1 - raising it doubles the rows but creates
  near-duplicate questions)
- Optional LLM backend (`llm.enabled: true`) rewrites questions into natural exam phrasing
  while keeping answers verbatim from the source
- Re-running merges with the existing `qa_all.jsonl`, so an LLM pass is never lost
- `--kinds` and `--subjects` restrict generation to specific atom kinds or subjects

### 5. Assemble and split (leakage control)

```bash
python scripts/assemble.py --output ./dataset --eval-frac 0.07 --max-overlap 0.10
```

Split strategy, in order of importance:

1. **Source-document holdout** - eval rows come from whole documents the model
   never sees, with each subject given a quota proportional to its share of the
   rows so the eval set mirrors the training mix
2. **IDF-weighted shingle filter** - any train row sharing more than
   `max_overlap` (default 10%) of an eval answer's 5-gram shingles is dropped,
   measured **in both directions** and against the single row it actually
   conflicts with. Shingles are weighted by inverse document frequency, so
   shared boilerplate ("is defined as") does not count as leakage - only
   distinctive content does. A shingle present in *every* answer carries
   exactly zero weight.
3. **Final cleanup** - a leak is always resolved by dropping the **train** row,
   never the eval row. The train set is abundant and the eval set is scarce;
   dropping an eval row loses the only copy of that material, and if it is the
   last document of a subject, that subject disappears from the eval set
4. **Subject balancing** - per-subject caps, applied after the split

Always read the printed `max eval/train answer overlap` line. It must be `<= max-overlap`, and
`validate.py` must print the same number - the two measure leakage identically, so a
disagreement means a bug rather than a dataset problem.

### 6. Validate and report

```bash
python scripts/validate.py --output ./dataset
python scripts/report.py  --output ./dataset --write report.md
```

`validate.py` checks: valid JSON per line, ChatML markers, CSV/JSONL row-count parity, required
keys, no watermark noise, no markdown fences, duplicate instructions, eval/train instruction
overlap, answer-shingle overlap, near-duplicate question pairs, and **echo rows** (rows whose
answer adds almost no new information beyond the question). Exits non-zero on any hard failure.

`report.py` writes a markdown quality report: subject balance, atom-kind mix, question diversity,
source coverage, length profile and split health.

### 7. Check that answers are grounded

```bash
python scripts/ground.py --output ./dataset
```

An answer the source does not support is a fabrication, and a fabrication teaches the model to
invent. `ground.py` scores every row by what share of the answer's distinctive content words
actually appear in the text it was extracted from, and reports the result per row, per subject
and as a histogram.

Two numbers are reported. **Own-document grounding** is the strict one: the answer must be
supported by the document it is attributed to. **Any-corpus grounding** is the weaker fallback -
a fact restated in another book is still grounded, it just is not attributable to this document.

Grounding is scored, not enforced, because a hard threshold depends on the corpus: a textbook
restates its own phrasing, so a genuinely-grounded answer can still score low when the PDF
extraction was lossy. To drop the weakest rows:

```bash
python scripts/ground.py --output ./dataset --drop-below 0.55 --write-grounded
```

This writes `train_grounded.jsonl` and `eval_grounded.jsonl` alongside the originals.

## Advanced Options

### Avoiding echo rows

A question that already contains the answer teaches the model to repeat the prompt. The generator
therefore asks about a **topic** extracted from the statement ("Explain the factors affecting...")
rather than quoting the statement. A small fraction of rows deliberately quote it - those are
tagged `kind: "restate"` and exist to teach paraphrase. `validate.py` reports the echo rate
separately; keep it under ~15%.

### LLM-assisted question phrasing

Set in the config:

```json
"llm": { "enabled": true, "endpoint": "http://127.0.0.1:8000/v1", "api_key": "...", "model": "auto" }
```

The LLM only rewrites the **question**; the answer stays verbatim from the extracted source, so
grounding is preserved. If the endpoint is down, the run continues with the deterministic rows.
See [docs/PIPELINE.md](docs/PIPELINE.md).

### Hand-written "gold" rows

Put curated pairs in `gold.jsonl` inside the output directory. They are merged first, so they
win on conflict, and they are always kept in **train** (never leaked into eval).

### Multiple generators

Run `qa.py` more than once with different settings, or add rows to `qa_all.jsonl` from your
own generator. `assemble.py` merges by instruction/answer dedupe with priority:
`gold > llm > atoms > your rows`.

## Training

`scripts/finetune_template.py` is a ready Unsloth script wired to the generated files.
Copy it, point `TRAIN_FILE` / `EVAL_FILE` at your dataset, and run:

```bash
pip install unsloth
python finetune.py
```

See [docs/TUNING.md](docs/TUNING.md) for VRAM sizing, LoRA rank guidance, and T4 limits.

## Monitoring a remote training run

`scripts/monitor.py` polls an Unsloth Studio run over the Chrome DevTools Protocol and appends a
status line to a log:

```bash
python scripts/monitor.py --url https://host.trycloudflare.com/studio --interval 60
```

It needs `scripts/cdp.py` next to it and a Chrome started with `--remote-debugging-port=9222`.

## Testing

```bash
python scripts/check.py            # pytest if available, else unittest
```

71 tests cover the full pipeline over a synthetic corpus (BOM, CRLF, unicode, corrupt files,
near-duplicate documents, single-document corpus, every missing-input path), the LLM backend
against a mock server, DOCX/Markdown input, block-aware PDF extraction, config validation,
subject-stratified splitting, and determinism.

`tests/test_regressions.py` pins down the specific defects that were found and fixed, so they
cannot come back: prefix precedence and detection pools, the IDF weight of universal
boilerplate, symmetric overlap, rows that vanish from both splits, a subject losing its only
document, the splitter and the validator reporting different overlap numbers, provenance, and
the grounding score.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Extraction is very slow | Threads used for PDF parsing | Use `extract.py` (process pool), not threads |
| `pdfplumber` "FontBBox" warnings | Broken font descriptor in source PDF | Harmless, they are suppressed |
| Few rows produced | Source text has short lines / no definition cues | Raise `--pages`, lower `atoms.min_sentence` |
| Eval set too small | Aggressive overlap filter | Raise `--max-overlap` or `--eval-frac` |
| Model repeats one question form | Template rotation disabled | Ensure `qa.py` ran (not a hand-rolled loop) |
| LLM pass returns nothing | Endpoint down or unreachable | Check `llm.endpoint`; the deterministic rows still land |

More in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Resources

- `config/config.example.json` - all tunables with comments
- `docs/PIPELINE.md` - stage contracts, data formats, LLM backend details
- `docs/DESIGN.md` - why the pipeline is shaped this way, and the measurements
  behind each decision (read before changing the heuristics)
- `docs/TUNING.md` - fine-tuning guidance (Unsloth, QLoRA, VRAM budgets)
- `docs/TROUBLESHOOTING.md` - extended problem/fix table
- `tests/` - the test suite
