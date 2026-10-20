# Example output

Three rows in the final ChatML JSONL format. The same data in CSV form has the
columns `instruction,input,output,subject`.

Run `python scripts/validate.py --output <dir>` against a directory containing
`train.jsonl`/`train.csv` and `eval.jsonl`/`eval.csv` to check syntax, ChatML
structure, CSV parity, duplicate instructions and eval/train leakage.
