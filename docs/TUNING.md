# Fine-Tuning Guide

## Model selection

| Model | Params | VRAM (QLoRA 4-bit) | Best for |
|---|---|---|---|
| Qwen3-0.6B | 0.6B | ~3 GB | fast iteration, narrow factual domains |
| Qwen3-1.7B | 1.7B | ~5 GB | **default recommendation** - best quality/VRAM ratio |
| Qwen3-4B | 4B | ~9 GB | stronger reasoning, still fits a 16GB card |
| Qwen3-8B | 8B | ~16 GB | needs a 24GB card, or 4-bit + heavy quantisation |

Start with **Qwen3-1.7B**. Move up only if the eval numbers plateau.

## QLoRA vs full fine-tune

Use **QLoRA** unless you have a reason not to:

- 16GB cards cannot full-fine-tune anything above ~0.5B comfortably
- QLoRA on 20k+ rows regularises well; full FT tends to overfit and forgets
- Merging the adapter afterwards gives a standalone model with zero inference
  overhead, so there is no runtime cost to using LoRA

Full fine-tune only makes sense for a large corpus (500k+ rows) on a 40GB+ GPU.

## Starting values

```bash
MODEL_ID=Qwen/Qwen3-1.7B \
EPOCHS=2 LR=2e-4 RANK=32 BATCH=4 ACCUM=8 MAX_SEQ=512 \
python finetune.py
```

- **max_seq_length** - set it to roughly the p95 of your token lengths. If the
  median row is 70 tokens, `512` is generous and much faster than `2048`.
- **rank / alpha** - `rank=32, alpha=2*rank` suits knowledge injection on
  20k-100k rows. `rank=16` for smaller sets, `rank=64` for very large ones.
- **batch** - keep the effective batch (batch x accum) in 16-64. Smaller batches
  generalise better on small datasets.
- **epochs** - 2 is a good default. Watch the eval loss; if it rises while the
  training loss falls, you are overfitting.
- **lr** - `2e-4` for LoRA. Lower it (`1e-4`) if the loss spikes.

## Hardware notes

**Tesla T4 (16GB, sm_75)**
- No bfloat16 support - the template detects this and falls back to fp16
- No FlashAttention-2 (needs sm_80+) - Unsloth falls back automatically
- ~0.2-0.3 steps/s for a 1.7B at batch 32
- Budget 1-2 hours for 2 epochs over 20k rows

**A100 40/80GB**
- bfloat16 and FlashAttention-2 both available
- Use `batch=8, accum=4`, `max_seq_length=1024`

**Colab**
- Sessions die on idle disconnect - mount Drive and set `output_dir` to a Drive
  path, and enable checkpointing (`save_steps=200`, already in the template)
- T4 only, 16GB

## Budgeting a run against a deadline

```
steps = (rows / effective_batch) * epochs
minutes = steps / steps_per_second / 60
```

Measure `steps_per_second` from the first 50 steps of the live run, then set
epochs so the run finishes with 20-30 minutes of slack for export and
evaluation. If the deadline is hard, cut epochs first - a 1-epoch run beats a
run that never finishes.

## Export

`MERGE=1` (default) writes `model-out-merged/`, a standalone model with the
LoRA delta folded into the base weights. Push that to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login      # paste a token with write access
huggingface-cli upload your-org/qwen3-1.7b-cbse ./model-out-merged
```

Or from Python:

```python
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)
api.create_repo("your-org/qwen3-1.7b-cbse", repo_type="model", private=False)
api.upload_folder("your-org/qwen3-1.7b-cbse", "model-out-merged")
```

## Evaluating

```bash
MODEL_DIR=./model-out-merged python evaluate.py --eval eval.jsonl
```

Reports mean token-F1, keyword recall on the top-5 reference terms, and prints
sample generations for eyeballing. Token-F1 above ~0.6 on a 1.7B model trained on
20k grounded rows usually means the model is reproducing the source material
faithfully. Below ~0.4 usually means under-training or a bad split.

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Loss starts at 2-3 and never drops | bad prompt format | ensure rows use ChatML `text` field |
| Loss drops to 0 immediately | leakage between train and eval | re-split, check the overlap number |
| Model answers verbatim with no style | LoRA rank too low, or too few steps | raise rank to 32-64, raise epochs |
| Model repeats one question form | template rotation not applied | regenerate with `qa.py` |
| Answers get truncated | max_seq_length too small | raise to 1024 |
| VRAM out of memory | batch too large | lower batch, keep accum, enable gradient checkpointing |
