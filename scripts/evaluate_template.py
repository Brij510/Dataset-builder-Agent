#!/usr/bin/env python3
"""Evaluation harness for a fine-tuned model against the generated eval split.

Reports token-level F1, keyword recall, and sample generations.

Usage:
  MODEL_DIR=./model-out-merged python evaluate.py
"""
import argparse
import collections
import json
import os
import re
import statistics

STOP = set("""a an the of to in and or is are was were be been for with on at by as that this
these those it its from into which who what when where why how if then than there their they them
he she his her we you your i our us not no can could will would shall should may might must have
has had do does did also more most other some any each all both few many much very same so too
only about after before above below between during over under again further once using used use""".split())


def norm_words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2]


def token_f1(pred, ref):
    p, r = collections.Counter(norm_words(pred)), collections.Counter(norm_words(ref))
    if not p or not r:
        return 0.0
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def keyword_recall(pred, ref, k=5):
    rk = [w for w, _ in collections.Counter(norm_words(ref)).most_common(k)]
    if not rk:
        return 0.0
    pk = set(norm_words(pred))
    return sum(1 for w in rk if w in pk) / len(rk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "model-out-merged"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"eval rows: {len(rows)}")
    print("subjects:", dict(collections.Counter(r.get("subject") for r in rows)))

    if not os.path.isdir(args.model):
        print(f"model dir '{args.model}' not found - skipping generation eval")
        return

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=None, load_in_4bit=True)
    model.eval()

    f1s, krs = [], []
    for i, r in enumerate(rows):
        prompt = r["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gen = gen.split("<|im_end|>")[0].strip()
        f1s.append(token_f1(gen, r["output"]))
        krs.append(keyword_recall(gen, r["output"]))
        if i < args.show:
            print(f"\n--- SAMPLE {i+1} ---")
            print(f"Q  : {r['instruction'][:140]}")
            print(f"REF: {r['output'][:200]}")
            print(f"GEN: {gen[:200]}")

    print(f"\n=== RESULTS over {len(f1s)} eval rows ===")
    print(f"Mean token-F1       : {statistics.mean(f1s):.4f}")
    print(f"Median token-F1     : {statistics.median(f1s):.4f}")
    print(f"Mean keyword recall : {statistics.mean(krs):.4f} (top-5)")
    print(f"Strong matches (>0.8): {sum(1 for x in f1s if x > 0.8)}/{len(f1s)}")


if __name__ == "__main__":
    main()
