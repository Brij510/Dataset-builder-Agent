# Troubleshooting

## Extraction

**`Could not get FontBBox from font descriptor`**
A broken font descriptor in the source PDF. Harmless - pdfplumber prints it and
continues. If it floods the terminal, redirect stderr.

**Extraction is extremely slow**
pdfplumber is pure Python and GIL-bound. A `ThreadPoolExecutor` gives **zero**
speedup; you need processes. `scripts/extract.py` uses `ProcessPoolExecutor` -
if you wrote your own version, use processes too.

**A few very large PDFs stall the run**
Files with hundreds of pages spend most of their time in `pdfplumber.open()`.
The `--pages` cap limits text extraction but not the open cost. Options: skip
files above a size threshold, or pre-split them.

**Nothing extracted**
Check `--ext` matches your files, and that the text is not a pure scan (image
PDF). Scanned documents need OCR (e.g. `ocrmypdf`) before this pipeline.

**Corpus looks empty for one subject**
Path prefixes in `subject_map` must match the relative path exactly as stored
in `corpus.json`. Print the keys to check:
`python -c "import json;print(list(json.load(open('dataset/corpus.json')))[:20])"`

## Atoms and QA

**Very few rows produced**
The text lines are too short or contain no definition cues. Raise `--pages`,
lower `atoms.min_sentence`, and check a sample of `corpus.json` by eye.

**Every question starts the same way**
Template rotation is not running. Use `scripts/qa.py` rather than a hand-rolled
loop, and confirm `per_atom` is 2.

**Answers look like table fragments**
Raise `qa.answer_min` and tighten the `is_bad_answer` heuristics in
`common.py` - the digit-density check is the main guard.

**One textbook dominates the dataset**
`per_source_cap` exists for this. Lower it, or tighten `subject_caps`.

**Concept names contain whole sentences**
The `definition` regex is too greedy. Tighten the concept capture group in
`atoms.py` (currently `{2,55}` characters, 1-7 words) and make sure
`BAD_CONCEPT` rejects sentences that begin with pronouns or fillers.

## Split and validation

**Eval set is tiny**
The overlap filter is dropping most candidates because the corpus contains
near-duplicate documents. Raise `--max-overlap` to `0.2` and re-run, or exclude
one copy of each duplicate pair from the input tree.

**`Max eval/train answer overlap` is above the limit**
There are duplicate documents (original and scan) in the corpus. Deduplicate the
input tree, then re-run `assemble.py`.

**Validation reports shared instructions between train and eval**
The two splits were produced by different runs. Re-run `assemble.py` - it
regenerates both halves atomically.

**CSV row count differs from JSONL**
A field contains an embedded newline that the CSV writer quoted, so physical
lines differ from records. `validate.py` compares *records* (via `csv.reader`),
not lines. If it still fails, a row is missing the required columns.

## Fine-tuning

**Out of memory**
Lower `BATCH`, keep `ACCUM` so the effective batch is unchanged, make sure
`load_in_4bit=True` and `use_gradient_checkpointing="unsloth"` are set.

**Colab session died mid-run**
Mount Drive and set `output_dir` to a Drive path so checkpoints survive. Resume
from the last checkpoint in `output_dir/checkpoint-*`.

**Training is slower than expected**
Lower `MAX_SEQ` to roughly the p95 of your token lengths. `2048` when the data
is 70 tokens wastes most of the compute.

**Model output quality is poor**
Check the eval split has no leakage (`validate.py`), confirm the model sees the
`text` field in ChatML, raise `RANK` to 32-64, and train at least 2 epochs.

**Run will not finish before the deadline**
Cut `EPOCHS` first. A completed 1-epoch run is far more useful than a 3-epoch
run that gets killed at 80%.

## LLM backend

**Endpoint times out**
Lower `llm.workers` to 1-2 and raise `llm.timeout`. Some local servers serialise
requests and cannot handle concurrency.

**LLM output is not JSON lines**
Some models wrap output in prose or fences. `qa.py` already strips fences and
salvages embedded JSON; if a model refuses, use a stricter system prompt or
skip the LLM pass.

**LLM answers drift from the source**
Never let the LLM write answers. The prompt forbids it, and the safest setup is
to use the LLM only for the question and keep the answer from the extracted
text.
