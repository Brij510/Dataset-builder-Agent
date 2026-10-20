#!/usr/bin/env python3
"""Test the LLM backend of qa.py against a mock OpenAI-compatible server.

Spins up a tiny HTTP server that returns ChatML JSON lines, then checks that
qa.py correctly parses the reply, tags the rows as origin=llm, and keeps the
source subject label.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

from fixtures import write  # noqa: E402

REPLY_LINES = [
    '{"text":"<|im_start|>user\\nDefine a monoid.<|im_end|>\\n'
    '<|im_start|>assistant\\nA monoid is a set with an associative operation and an identity.<|im_end|>"}',
    '{"text":"<|im_start|>user\\nWhat is a group?<|im_end|>\\n'
    '<|im_start|>assistant\\nA group is a monoid in which every element has an inverse.<|im_end|>"}',
    "not json at all",
    '{"text":"<|im_start|>user\\nbroken markers"}',
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = "\n".join(REPLY_LINES).encode()
        resp = json.dumps({"choices": [{"message": {"content": body.decode()}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


class LLMBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_llm_rows_are_parsed_and_tagged(self):
        tmp = tempfile.mkdtemp(prefix="dsb_llm_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out, exist_ok=True)
            write(os.path.join(out, "corpus.json"),
                  json.dumps({"Math/groups.txt": "A group is a monoid in which every "
                                                "element has an inverse. " * 20}))
            write(os.path.join(out, "atoms.jsonl"), json.dumps({
                "kind": "definition", "concept": "group",
                "statement": "A group is a monoid in which every element has an inverse.",
                "subject": "Math", "source": "Math/groups.txt"}) + "\n")
            cfg = os.path.join(tmp, "cfg.json")
            write(cfg, json.dumps({
                "llm": {"enabled": True, "endpoint": f"http://127.0.0.1:{self.port}/v1",
                        "api_key": "test", "model": "mock", "workers": 1, "timeout": 30},
                "qa": {"target": 100, "per_source_cap": 50, "per_atom": 2,
                       "max_per_answer": 2, "cloze_every": 3},
            }))
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "qa.py"),
                                "--output", out, "--config", cfg, "--llm"],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            rows = [json.loads(l) for l in
                    open(os.path.join(out, "qa_all.jsonl"), encoding="utf-8") if l.strip()]
            llm_rows = [r for r in rows if r.get("origin") == "llm"]
            self.assertEqual(len(llm_rows), 2, f"expected 2 LLM rows, got {len(llm_rows)}")
            for r in llm_rows:
                self.assertEqual(r["subject"], "Math", "LLM row lost its subject label")
                self.assertTrue(r["source"], "LLM row lost its source pointer")
                self.assertNotIn("not json", r["instruction"])
                self.assertIn("<|im_start|>", r["instruction"] if "<|im_start|>" in r["instruction"]
                              else r["text"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_llm_pass_does_not_wipe_existing_rows(self):
        tmp = tempfile.mkdtemp(prefix="dsb_llm2_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out, exist_ok=True)
            write(os.path.join(out, "corpus.json"),
                  json.dumps({"Math/groups.txt": "A group is a monoid with inverses. " * 20}))
            write(os.path.join(out, "atoms.jsonl"), json.dumps({
                "kind": "definition", "concept": "group",
                "statement": "A group is a monoid in which every element has an inverse.",
                "subject": "Math", "source": "Math/groups.txt"}) + "\n")
            cfg = os.path.join(tmp, "cfg.json")
            write(cfg, json.dumps({
                "llm": {"enabled": True, "endpoint": f"http://127.0.0.1:{self.port}/v1",
                        "api_key": "test", "model": "mock", "workers": 1, "timeout": 30},
                "qa": {"target": 100, "per_source_cap": 50, "per_atom": 2,
                       "max_per_answer": 2, "cloze_every": 3},
            }))
            subprocess.run([sys.executable, os.path.join(SCRIPTS, "qa.py"),
                            "--output", out, "--config", cfg, "--llm"],
                           capture_output=True, text=True, timeout=300)
            before = open(os.path.join(out, "qa_all.jsonl"), encoding="utf-8").read()
            self.assertIn("llm", before)
            # rerun WITHOUT the llm flag: previously generated LLM rows must survive
            subprocess.run([sys.executable, os.path.join(SCRIPTS, "qa.py"),
                            "--output", out, "--config", cfg],
                           capture_output=True, text=True, timeout=300)
            after = [json.loads(l) for l in
                     open(os.path.join(out, "qa_all.jsonl"), encoding="utf-8") if l.strip()]
            self.assertTrue(any(r.get("origin") == "llm" for r in after),
                            "rerun without --llm wiped the earlier LLM rows")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_llm_endpoint_failure_is_not_fatal(self):
        tmp = tempfile.mkdtemp(prefix="dsb_llm3_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out, exist_ok=True)
            write(os.path.join(out, "corpus.json"),
                  json.dumps({"Math/groups.txt": "A group is a monoid with inverses. " * 20}))
            write(os.path.join(out, "atoms.jsonl"), json.dumps({
                "kind": "definition", "concept": "group",
                "statement": "A group is a monoid in which every element has an inverse.",
                "subject": "Math", "source": "Math/groups.txt"}) + "\n")
            cfg = os.path.join(tmp, "cfg.json")
            write(cfg, json.dumps({
                "llm": {"enabled": True, "endpoint": "http://127.0.0.1:9/v1",
                        "api_key": "x", "model": "mock", "workers": 1, "timeout": 5},
                "qa": {"target": 100, "per_source_cap": 50, "per_atom": 2,
                       "max_per_answer": 2, "cloze_every": 3},
            }))
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "qa.py"),
                                "--output", out, "--config", cfg, "--llm"],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            rows = [json.loads(l) for l in
                    open(os.path.join(out, "qa_all.jsonl"), encoding="utf-8") if l.strip()]
            self.assertTrue(len(rows) >= 1, "deterministic rows must survive an LLM outage")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
