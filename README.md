# Instruction Dataset Builder

Build production-quality instruction-tuning datasets (train/eval JSONL + CSV) from raw
documents. Designed for small-model fine-tuning (Qwen3-0.6B / 1.7B) on modest hardware.

```
raw docs -> [1] text corpus -> [2] knowledge atoms -> [3] QA pairs -> [4] split -> [5] validate
```

## What makes it different

- **Grounded** - every answer is text that was actually extracted from a source
  document, and every row keeps a `source` pointer.
- **Leakage-controlled** - eval rows come from held-out *documents*; train rows
  that overlap eval answers are removed with a symmetric 5-gram shingle check.
- **Diverse** - 11 atom kinds and ~60 question archetypes selected round-robin,
  so the dataset does not collapse into one repeated template.
- **No echo rows** - questions ask *about* a topic instead of quoting the answer,
  so the model learns to elaborate rather than to repeat the prompt.
- **Resumable** - extraction caches per file; every stage can be re-run safely.
- **Fast** - process-pool extraction (threads do not work - pdfplumber is
  GIL-bound), shingle-based dedupe, deterministic splits.

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
python scripts/atoms.py    --output ./dataset --subjects Math,Science
python scripts/qa.py       --output ./dataset --target 20000
python scripts/assemble.py --output ./dataset --eval-frac 0.07
python scripts/validate.py --output ./dataset
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

## Layout

```
instruction-dataset-builder/
├── SKILL.md
├── README.md
├── config/config.example.json
├── scripts/
│   ├── common.py              shared utilities (cleaning, unwrap, shingles, IO)
│   ├── extract.py             stage 1
│   ├── atoms.py               stage 2
│   ├── qa.py                  stage 3
│   ├── assemble.py            stage 4
│   ├── validate.py            stage 5
│   ├── pipeline.py            orchestrator
│   ├── report.py              markdown quality report
│   ├── monitor.py             watch a remote Unsloth Studio run (CDP)
│   ├── cdp.py                 minimal Chrome DevTools Protocol client
│   ├── finetune_template.py   Unsloth training template
│   ├── evaluate_template.py   eval harness
│   └── setup.sh
└── docs/
    ├── PIPELINE.md
    ├── TUNING.md
    └── TROUBLESHOOTING.md
```
