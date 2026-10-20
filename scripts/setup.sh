#!/usr/bin/env bash
# Install the Python dependencies needed by the pipeline.
set -euo pipefail

PY=${PYTHON:-python3}

echo "==> installing core dependencies"
$PY -m pip install --quiet --upgrade pip

# pdfplumber is the only hard requirement for PDF input.
# python-docx is optional (only needed for .docx sources).
if ! $PY -c "import pdfplumber" >/dev/null 2>&1; then
  $PY -m pip install --quiet pdfplumber || \
  $PY -m pip install --quiet --break-system-packages pdfplumber
fi

if ! $PY -c "import docx" >/dev/null 2>&1; then
  $PY -m pip install --quiet python-docx || \
  $PY -m pip install --quiet --break-system-packages python-docx
fi

echo "==> optional fine-tuning stack (install manually when needed)"
cat <<'EOF'
  pip install unsloth transformers datasets trl accelerate bitsandbytes
EOF

echo "==> verifying"
$PY - <<'EOF'
import importlib
for m in ("pdfplumber",):
    try:
        importlib.import_module(m)
        print(f"  ok: {m}")
    except Exception as e:
        raise SystemExit(f"  MISSING: {m} ({e})")
try:
    importlib.import_module("docx")
    print("  ok: docx (optional)")
except Exception:
    print("  skip: docx (optional, only needed for .docx input)")
EOF

echo "==> done. Next: python scripts/pipeline.py --input ./docs --output ./dataset"
