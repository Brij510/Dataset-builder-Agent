# Instruction Dataset Builder

Build production-quality instruction-tuning datasets (train/eval JSONL + CSV) from raw documents.
Designed for small-model fine-tuning (Qwen3-0.6B / 1.7B) on modest hardware.

```
raw docs -> extract -> dedupe -> atoms -> qa -> split -> validate
```

## What makes it different

- **Grounded** - every answer is text that was actually extracted from a source document, and
  every row keeps a `source` pointer.
- **Leakage-controlled** - eval rows come from held-out *documents*, stratified by subject so the
  eval mix mirrors the training mix; train rows that overlap eval answers are removed with a
  symmetric, IDF-weighted shingle check (boilerplate does not count as leakage).
- **No echo rows** - questions ask *about* a topic instead of quoting the answer, so the model
  learns to elaborate rather than to repeat the prompt.
- **Diverse** - 11 atom kinds and ~60 question archetypes selected round-robin.
- **Duplicate-proof** - near-duplicate documents are detected and removed before generation, so a
  textbook and its scan cannot straddle the train/eval boundary.
- **Resumable** - extraction caches per file; every stage can be re-run safely.
- **Fast** - process-pool extraction (threads do not work - pdfplumber is GIL-bound), shingle-based
  dedupe, deterministic splits.
- **Block-aware** - textbook sidebars and two-column pages are reconstructed as blocks, so a
  worked example is never spliced into the paragraph beside it.
- **Tested** - 54 tests over a synthetic corpus, a mock LLM server, and every failure path.

## Install

```bash
./scripts/setup.sh
```

## Use

```bash
python scripts/pipeline.py --input ./my_docs --output ./dataset \
    --config config/config.example.json --target 20000
```

Or stage by stage:

```bash
python scripts/extract.py  --input ./my_docs --output ./dataset
python scripts/dedupe.py   --output ./dataset
python scripts/atoms.py    --output ./dataset --subjects Math,Science
python scripts/qa.py       --output ./dataset --target 20000
python scripts/assemble.py --output ./dataset --eval-frac 0.07
python scripts/validate.py --output ./dataset
python scripts/report.py   --output ./dataset --write report.md
```

## Output

| File | Contents |
|---|---|
| `train.jsonl` / `train.csv` | ChatML training rows |
| `eval.jsonl` / `eval.csv` | held-out evaluation rows |
| `corpus.json` | cached extracted text per source file |
| `atoms.jsonl` | typed knowledge atoms |
| `qa_all.jsonl` | all QA before the split |

## Train

```bash
cp scripts/finetune_template.py ./finetune.py
MODEL_ID=Qwen/Qwen3-1.7B EPOCHS=2 python finetune.py
```

See [docs/TUNING.md](docs/TUNING.md).

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Layout

```
instruction-dataset-builder/
├── SKILL.md
├── README.md
├── requirements.txt
├── .github/workflows/tests.yml
├── config/config.example.json
├── scripts/
│   ├── common.py              shared utilities (cleaning, unwrap, shingles, IO)
│   ├── extract.py             stage 1 - text extraction
│   ├── dedupe.py              stage 2 - near-duplicate document removal
│   ├── atoms.py               stage 3 - knowledge atom mining
│   ├── qa.py                  stage 4 - QA generation
│   ├── assemble.py            stage 5 - split with leakage control
│   ├── validate.py            stage 6 - validation
│   ├── pipeline.py            orchestrator
│   ├── report.py              markdown quality report
│   ├── monitor.py             watch a remote Unsloth Studio run (CDP)
│   ├── cdp.py                 minimal Chrome DevTools Protocol client
│   ├── finetune_template.py   Unsloth training template
│   ├── evaluate_template.py   eval harness
│   └── setup.sh
├── tests/
│   ├── fixtures.py            synthetic corpus with edge cases
│   ├── test_pipeline.py       end-to-end + unit tests
│   ├── test_llm_backend.py    LLM backend against a mock server
│   └── test_formats_and_config.py
└── docs/
    ├── PIPELINE.md
    ├── DESIGN.md
    ├── TUNING.md
    └── TROUBLESHOOTING.md
```
