#!/usr/bin/env python3
"""Developer entry point: run the skill's test suite.

Prefers pytest, falls back to unittest. Works from any working directory,
which local development and the CI workflow both need.

Usage:
  python scripts/check.py
  python scripts/check.py -k split
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")

for p in (HERE, TESTS, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

argv = [a for a in sys.argv[1:] if a not in ("-v", "--verbose")]

try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    raise SystemExit(pytest.main(["-q", TESTS] + argv))

import unittest

loader = unittest.TestLoader()
suite = loader.discover(TESTS, pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
