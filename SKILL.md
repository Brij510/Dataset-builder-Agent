---
name: "Instruction Dataset Builder"
description: "Builds high-quality instruction-tuning datasets (train/eval JSONL + CSV) from raw documents - PDFs, DOCX, TXT, Markdown - by extracting text, mining knowledge atoms (definitions, formulas, comparisons, processes, causes, lists), generating diverse grounded QA pairs, and performing leakage-controlled train/eval splits. Use when creating SFT or fine-tuning data for LLMs from textbooks, notes, question banks, legal or medical corpora, or when asked to turn documents into a training dataset, build a RAG/eval set, or scale QA generation to 10k+ rows."
---

# Instruction Dataset Builder

## What This Skill Does

Converts a folder of raw documents into a production-ready instruction-tuning dataset with a
train/eval split that cannot leak. The pipeline is five stages:

```
raw docs -> [1] text corpus -> [2] knowledge atoms -> [3] QA pairs -> [4] split -> [5] validate
```

Every stage is a standalone, resumable, cached script. Nothing is fabricated: each answer is
grounded in text that was actually extracted from a source document, and every row keeps a
`source` pointer back to the file it came from.

## Prerequisites

- Python 3.9+
- `pdfplumber` (PDFs), `python-docx` (DOCX) — run `scripts/setup.sh` to install
- Optional: an OpenAI-compatible endpoint for LLM-assisted question phrasing
  (set `llm.endpoint` + `llm.api_key` in the config)

## Quick Start

```bash
# 1. One-shot: extract -> atoms -> qa -> split -> validate
python scripts/pipeline.py --input ./my_docs --output ./dataset --config config/config.example.json

# 2. Or run stages individually (each is resumable and cached)
python scripts/extract.py  --input ./my_docs --output ./dataset
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
- **Uses processes, not threads** — PDF parsing is GIL-bound, threads give no speedup
- Caches per file and is **resumable**: rerun after adding files, only new ones are parsed
- Strips watermarks, page numbers, de-hyphenates line breaks, drops low-signal lines

### 2. Mine knowledge atoms

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
| `fact` | fallback | State the fact, Explain this point |

Also emits `cloze` (fill-in-the-blank) items for a configurable fraction of atoms.

### 3. Generate QA pairs

```bash
python scripts/qa.py --output ./dataset --target 20000 --per-source-cap 220
```

- Multiple archetypes per atom, selected **round-robin** so the dataset does not collapse
  into one repeated template
- Per-source cap prevents one large textbook from flooding the dataset
- Dedupe on normalized instruction *and* normalized answer
- Optional LLM backend (`llm.enabled: true`) rewrites questions into natural exam phrasing
  while keeping answers verbatim from the source

### 4. Assemble and split (leakage control)

```bash
python scripts/assemble.py --output ./dataset --eval-frac 0.07 --max-overlap 0.10
```

Split strategy, in order of importance:

1. **Source-document holdout** - eval rows come from whole documents the model never sees
2. **Shingle overlap filter** - any train row sharing >10% of 5-gram answer shingles with an
   eval row is dropped (symmetric check, so short/long answer pairs cannot slip through)
3. **Final cleanup** - any eval row still overlapping is removed
4. **Subject balancing** - per-subject caps, applied after the split

Always read the printed `Max eval/train answer overlap` line. It must be `<= max-overlap`.

### 5. Validate

```bash
python scripts/validate.py --output ./dataset
```

Checks: valid JSON per line, ChatML markers, CSV/JSONL row-count parity, no watermark noise,
no markdown fences, duplicate instructions, eval/train instruction overlap, answer-shingle
overlap, and question-template diversity. Exits non-zero on any hard failure.

## Advanced Options

### LLM-assisted question phrasing

Set in the config:

```json
"llm": { "enabled": true, "endpoint": "http://127.0.0.1:8000/v1", "api_key": "...", "model": "auto" }
```

The LLM only rewrites the **question**; the answer stays verbatim from the extracted source,
so grounding is preserved. See [docs/PIPELINE.md](docs/PIPELINE.md).

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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Extraction is very slow | Threads used for PDF parsing | Use `extract.py` (process pool), not threads |
| `pdfplumber` "FontBBox" warnings | Broken font descriptor in source PDF | Harmless, they are suppressed |
| Few rows produced | Source text has short lines / no definition cues | Raise `--pages`, lower min sentence length in config |
| Eval set too small | Aggressive overlap filter | Raise `--max-overlap` or `--eval-frac` |
| Eval overlap still high | Near-duplicate documents in the corpus | Re-run with a larger `--min-shingle-overlap` split seed |
| Model repeats one question form | Template rotation disabled | Ensure `qa.py` ran (not a hand-rolled loop) |

More in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Resources

- `config/config.example.json` - all tunables with comments
- `docs/PIPELINE.md` - stage contracts, data formats, LLM backend details
- `docs/TUNING.md` - fine-tuning guidance (Unsloth, QLoRA, VRAM budgets)
- `docs/TROUBLESHOOTING.md` - extended problem/fix table
- `resources/examples/` - a tiny end-to-end example with expected outputs
