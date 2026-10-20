#!/usr/bin/env python3
"""Orchestrator - runs the whole pipeline end to end.

  extract -> dedupe -> atoms -> qa -> assemble -> validate

Each stage is resumable; re-running skips work already cached.

Usage:
  python pipeline.py --input ./docs --output ./dataset --config config.json
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(stage, script, argv):
    cmd = [sys.executable, os.path.join(HERE, script)] + argv
    print(f"\n=== {stage} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[pipeline] stage '{stage}' failed with exit code {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"[pipeline] {stage} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Full dataset pipeline")
    ap.add_argument("--input", required=True, help="root folder of raw documents")
    ap.add_argument("--output", required=True, help="dataset working directory")
    ap.add_argument("--config")
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--eval-frac", type=float, default=0.07)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-dedupe", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cfg_arg = ["--config", args.config] if args.config else []

    run("1/7 extract", "extract.py",
        ["--input", args.input, "--output", args.output,
         "--pages", str(args.pages), "--workers", str(args.workers)] + cfg_arg)

    if not args.skip_dedupe:
        run("2/7 dedupe", "dedupe.py",
            ["--output", args.output] + cfg_arg)

    run("3/7 atoms", "atoms.py",
        ["--output", args.output] + cfg_arg +
        (["--subjects", args.subjects] if args.subjects else []))

    run("4/7 qa", "qa.py",
        ["--output", args.output, "--target", str(args.target)] + cfg_arg +
        (["--llm"] if args.llm else []))

    run("5/7 assemble", "assemble.py",
        ["--output", args.output, "--eval-frac", str(args.eval_frac)] + cfg_arg)

    if not args.skip_validate:
        run("6/7 validate", "validate.py", ["--output", args.output] + cfg_arg)

    run("7/7 ground", "ground.py", ["--output", args.output] + cfg_arg)

    print("\n[pipeline] complete")
    print(f"  {os.path.join(args.output, 'train.jsonl')}")
    print(f"  {os.path.join(args.output, 'eval.jsonl')}")


if __name__ == "__main__":
    main()
