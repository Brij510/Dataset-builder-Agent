---
name: "Instruction Dataset Builder"
description: "Builds high-quality instruction-tuning datasets (train/eval JSONL + CSV) from raw documents - PDFs, DOCX, TXT, Markdown - by extracting text, mining knowledge atoms (definitions, formulas, comparisons, processes, causes, lists), generating diverse grounded QA pairs, and performing leakage-controlled train/eval splits. Use when creating SFT or fine-tuning data for LLMs from textbooks, notes, question banks, legal or medical corpora, or when asked to turn documents into a training dataset, build a RAG/eval set, or scale QA generation to 10k+ rows."
---

# Instruction Dataset Builder

## What This Skill Does

Converts a folder of raw documents into a production-ready instruction-tuning dataset with a
train/eval split that cannot leak. The pipeline is six stages:

```
raw docs -> [1] extract -> [2] dedupe -> [3] atoms -> [4] QA -> [5] split -> [6] validate
```

Every stage is standalone, resumable and cached. Nothing is fabricated: each answer is grounded
in text that was actually extracted from a source document, and every row keeps a `source` pointer
back to the file it came from.

## Prerequisites

- Python 3.9+
- `pdfplumber` (PDFs), `python-docx` (DOCX) - run `scripts/setup.sh`
- Optional: an OpenAI-compatible endpoint for LLM-assisted question phrasing
  (`llm.endpoint` + `llm.api_key` in the config)

## Quick Start

```bash
# One-shot: extract -> dedupe -> atoms -> qa -> split -> validate
python scripts/pipeline.py --input ./my_docs --output ./dataset --config config/config.example.json

# Or stage by stage (each is resumable and cached)
python scripts/extract.py  --input ./my_docs --output ./dataset
python scripts/dedupe.py   --output ./dataset
python scripts/atoms.py    --output ./dataset --subjects Math,Science
python scripts/qa.py       --output ./dataset --target 20000
python scripts/assemble.py --output ./dataset --eval-frac 0.07
python scripts/validate.py --output ./dataset
```

Outputs in `./dataset`:

| File | Purpose |
|---|---|
| `train.jsonl` / `train.csv` | ChatML training rows |
| `eval.jsonl` / `eval.csv` | Held-out evaluation rows (zero leakage) |
| `corpus.json` | Cached extracted text per source file |
| `atoms.jsonl` | Mined knowledge atoms |
| `qa_all.jsonl` | All generated QA before splitting |

## Step-by-Step Guide

### 1. Extract text from raw documents

```bash
python scripts/extract.py --input ./my_docs --output ./dataset --pages 4 --workers 4
```

- Walks the input tree, reads `.pdf`, `.docx`, `.txt`, `.md`
- Writes `corpus.json`: `{relative_path: cleaned_text}`
- **Uses processes, not threads** - PDF parsing is GIL-bound, threads give no speedup
- Caches per file and is **resumable**: rerun after adding files, only new ones are parsed
- Strips watermarks, page numbers, BOM, CRLF, de-hyphenates line breaks, drops noise-only lines
- `--limit N` processes only the first N documents (quick trials)
- A corrupt file is skipped, not fatal

### 2. Remove near-duplicate documents

```bash
python scripts/dedupe.py --output ./dataset
```

A textbook and its scan produce identical content. Left in place they waste generation effort and
break the document-holdout split, because the model would see the same material in both halves
even though the files differ. Detection uses 5-gram shingles over document text; the second copy
of a >60% overlap pair is dropped. `corpus.json.pre-dedupe` keeps a backup.

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
- Dedupe on normalized instruction, and on normalized answer bounded by `max_per_answer`
- Optional LLM backend (`llm.enabled: true`) rewrites questions into natural exam phrasing
  while keeping answers verbatim from the source
- Re-running merges with the existing `qa_all.jsonl`, so an LLM pass is never lost

### 5. Assemble and split (leakage control)

```bash
python scripts/assemble.py --output ./dataset --eval-frac 0.07 --max-overlap 0.10
```

Split strategy, in order of importance:

1. **Source-document holdout** - eval rows come from whole documents the model never sees
2. **Shingle overlap filter** - any train row sharing more than `max_overlap` (default 10%) of
   5-gram answer shingles with an eval row is dropped, measured **in both directions**
3. **Final cleanup** - any eval row still overlapping is dropped
4. **Subject balancing** - per-subject caps, applied after the split

Always read the printed `max eval/train answer overlap` line. It must be `<= max-overlap`.

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
python -m unittest discover -s tests -p 'test_*.py' -v
```

43 tests cover the full pipeline over a synthetic corpus (BOM, CRLF, unicode, corrupt files,
near-duplicate documents, single-document corpus, every missing-input path), the LLM backend
against a mock server, DOCX/Markdown input, config validation, and determinism.

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
- `docs/TUNING.md` - fine-tuning guidance (Unsloth, QLoRA, VRAM budgets)
- `docs/TROUBLESHOOTING.md` - extended problem/fix table
- `tests/` - the test suite
