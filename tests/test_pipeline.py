#!/usr/bin/env python3
"""End-to-end + unit tests for the instruction-dataset-builder pipeline.

Run:  python tests/test_pipeline.py
Exits non-zero on the first failure summary.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

from fixtures import make_corpus, write  # noqa: E402

PY = sys.executable


def run(script, *args, expect_ok=True):
    cmd = [PY, os.path.join(SCRIPTS, script)] + [str(a) for a in args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"{script} failed ({p.returncode})\n"
                             f"STDOUT:\n{p.stdout[-2500:]}\nSTDERR:\n{p.stderr[-2500:]}")
    return p


class PipelineE2E(unittest.TestCase):
    """Full pipeline over a synthetic corpus with edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dsb_e2e_")
        cls.src = os.path.join(cls.tmp, "docs")
        cls.out = os.path.join(cls.tmp, "out")
        make_corpus(cls.src)
        cfg = os.path.join(cls.tmp, "cfg.json")
        write(cfg, json.dumps({
            "subjects": ["Math", "Science", "Social Science"],
            "auto_subject": True,
            "subject_map": {},
            "qa": {"target": 500, "per_source_cap": 50, "per_atom": 2,
                   "max_per_answer": 2, "cloze_every": 3},
            "split": {"eval_frac": 0.2, "max_overlap": 0.10, "shingle_n": 5,
                      "subject_caps": {}},
        }))
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_extract(self):
        p = run("extract.py", "--input", self.src, "--output", self.out,
                "--pages", "2", "--workers", "2")
        self.assertIn("[extract]", p.stdout)
        corpus = json.load(open(os.path.join(self.out, "corpus.json")))
        self.assertGreaterEqual(len(corpus), 5)
        # BOM + CRLF must be handled
        joined = " ".join(corpus.values())
        self.assertNotIn("\ufeff", joined)
        self.assertNotIn("\r", joined)
        # watermark noise must be stripped
        self.assertNotIn("www.example.com", joined)
        # unicode preserved
        self.assertTrue(any("Ω" in v or "français" in v for v in corpus.values()))

    def test_02_extract_is_resumable(self):
        before = os.path.getsize(os.path.join(self.out, "corpus.json"))
        p = run("extract.py", "--input", self.src, "--output", self.out,
                "--pages", "2", "--workers", "2")
        after = os.path.getsize(os.path.join(self.out, "corpus.json"))
        self.assertIn("already complete", p.stdout)
        self.assertEqual(before, after)

    def test_02b_dedupe_removes_near_duplicate_documents(self):
        p = run("dedupe.py", "--output", self.out)
        self.assertIn("duplicate", p.stdout)
        corpus = json.load(open(os.path.join(self.out, "corpus.json")))
        self.assertFalse(any("algebra_copy" in k for k in corpus),
                         "near-duplicate document survived dedupe")
        # the original must be kept
        self.assertTrue(any(k.endswith("algebra.txt") for k in corpus))

    def test_03_atoms(self):
        p = run("atoms.py", "--output", self.out, "--config", self.cfg)
        self.assertIn("[atoms]", p.stdout)
        atoms = [json.loads(l) for l in open(os.path.join(self.out, "atoms.jsonl")) if l.strip()]
        self.assertGreater(len(atoms), 5)
        kinds = {a["kind"] for a in atoms}
        self.assertIn("definition", kinds)
        # no fragment-style statements
        for a in atoms:
            self.assertFalse(a["statement"].rstrip().endswith("?"))
            self.assertGreaterEqual(len(a["statement"]), 25)

    def test_04_subject_autodetect(self):
        atoms = [json.loads(l) for l in open(os.path.join(self.out, "atoms.jsonl")) if l.strip()]
        subs = {a["subject"] for a in atoms}
        self.assertTrue(subs & {"Math", "Science", "Social Science"},
                        f"no subject detected, got {subs}")

    def test_05_qa(self):
        p = run("qa.py", "--output", self.out, "--config", self.cfg, "--target", "500")
        rows = [json.loads(l) for l in open(os.path.join(self.out, "qa_all.jsonl")) if l.strip()]
        self.assertGreater(len(rows), 5)
        for r in rows:
            self.assertTrue(r["instruction"].strip())
            self.assertTrue(r["output"].strip())
            self.assertNotIn("{c}", r["output"], "unsubstituted template variable")
            self.assertNotIn("{c}", r["instruction"], "unsubstituted template variable")
            self.assertNotIn("{s}", r["instruction"], "unsubstituted template variable")
            self.assertNotIn("{t}", r["instruction"], "unsubstituted template variable")

    def test_06_assemble(self):
        p = run("assemble.py", "--output", self.out, "--config", self.cfg)
        self.assertIn("[assemble]", p.stdout)
        train = [json.loads(l) for l in open(os.path.join(self.out, "train.jsonl")) if l.strip()]
        ev = [json.loads(l) for l in open(os.path.join(self.out, "eval.jsonl")) if l.strip()]
        self.assertGreater(len(train), 0)
        self.assertGreater(len(ev), 0)

    def test_07_validate(self):
        p = run("validate.py", "--output", self.out, "--config", self.cfg)
        self.assertIn("All checks passed", p.stdout)
        self.assertNotIn("ERRORS", p.stdout)

    def test_08_report(self):
        p = run("report.py", "--output", self.out)
        self.assertIn("Dataset Report", p.stdout)
        self.assertIn("Split health", p.stdout)

    def test_09_chatml_and_csv_parity(self):
        import csv as csvmod
        for split in ("train", "eval"):
            jl = [json.loads(l) for l in open(os.path.join(self.out, f"{split}.jsonl")) if l.strip()]
            with open(os.path.join(self.out, f"{split}.csv"), encoding="utf-8") as f:
                rows = list(csvmod.reader(f))
            self.assertEqual(rows[0], ["instruction", "input", "output", "subject"])
            self.assertEqual(len(rows) - 1, len(jl), f"{split} csv/jsonl mismatch")
            for o in jl:
                t = o["text"]
                self.assertEqual(t.count("<|im_start|>"), 2)
                self.assertEqual(t.count("<|im_end|>"), 2)
                self.assertIn("<|im_start|>user", t)
                self.assertIn("<|im_start|>assistant", t)

    def test_10_no_leakage(self):
        from common import shingles
        train = [json.loads(l) for l in open(os.path.join(self.out, "train.jsonl")) if l.strip()]
        ev = [json.loads(l) for l in open(os.path.join(self.out, "eval.jsonl")) if l.strip()]
        tri = {r["instruction"].lower().strip() for r in train}
        for r in ev:
            self.assertNotIn(r["instruction"].lower().strip(), tri)
        tr = set()
        for r in train:
            tr |= shingles(r["output"])
        for r in ev:
            s = shingles(r["output"])
            if s:
                self.assertLessEqual(len(s & tr) / len(s), 0.10)


class AssembleSafetyNet(unittest.TestCase):
    """assemble.py must catch duplicate documents even if dedupe.py was skipped."""

    def test_assemble_drops_duplicate_documents(self):
        tmp = tempfile.mkdtemp(prefix="dsb_dup_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out, exist_ok=True)
            rows = []
            for src, suffix in (("a/book.txt", "a"), ("a/book_copy.txt", "b")):
                for i in range(6):
                    rows.append({
                        "text": f"<|im_start|>user\nWhat is concept {i}?\n<|im_end|>\n"
                                f"<|im_start|>assistant\nConcept {i} is defined in the "
                                f"source document with a full explanatory sentence here."
                                f"<|im_end|>",
                        "subject": "Math", "instruction": f"What is concept {i}?",
                        "input": "", "output":
                            f"Concept {i} is defined in the source document with a full "
                            f"explanatory sentence here.",
                        "source": src, "origin": "atoms", "kind": "fact"})
            write(os.path.join(out, "qa_all.jsonl"),
                  "".join(json.dumps(r) + "\n" for r in rows))
            p = run("assemble.py", "--output", out, "--eval-frac", "0.2")
            self.assertIn("near-duplicate", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class EdgeCases(unittest.TestCase):
    """Each stage must fail loudly or degrade gracefully on bad input."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dsb_edge_")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.out, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_atoms_without_corpus_fails_cleanly(self):
        p = run("atoms.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("corpus.json", (p.stderr + p.stdout))

    def test_qa_without_atoms_fails_cleanly(self):
        p = run("qa.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)

    def test_assemble_without_qa_fails_cleanly(self):
        p = run("assemble.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)

    def test_validate_reports_missing_files(self):
        p = run("validate.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ERRORS", p.stdout)

    def test_validate_detects_broken_chatml(self):
        write(os.path.join(self.out, "train.jsonl"),
              json.dumps({"text": "<|im_start|>user\nhi"}) + "\n")
        write(os.path.join(self.out, "eval.jsonl"),
              json.dumps({"text": "<|im_start|>user\nhi<|im_end|>\n"
                                  "<|im_start|>assistant\nyo<|im_end|>"}) + "\n")
        p = run("validate.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ChatML", p.stdout)

    def test_validate_detects_watermark(self):
        write(os.path.join(self.out, "train.jsonl"), json.dumps({
            "text": "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
                    "Prashant Kirad made this<|im_end|>",
            "subject": "X", "instruction": "q", "input": "", "output": "x"}) + "\n")
        write(os.path.join(self.out, "eval.jsonl"), "")
        p = run("validate.py", "--output", self.out, expect_ok=False)
        self.assertIn("watermark", p.stdout)

    def test_empty_corpus_produces_empty_but_valid_output(self):
        src = os.path.join(self.tmp, "empty")
        os.makedirs(src, exist_ok=True)
        run("extract.py", "--input", src, "--output", self.out)
        p = run("atoms.py", "--output", self.out, expect_ok=False)
        self.assertNotEqual(p.returncode, 0)

    def test_single_document_corpus_does_not_crash_split(self):
        src = os.path.join(self.tmp, "one")
        write(os.path.join(src, "a.txt"),
              "A monoid is a set with an associative binary operation and an identity element.\n"
              "A group is a monoid in which every element has an inverse.\n" * 3)
        run("extract.py", "--input", src, "--output", self.out, "--workers", "1")
        run("atoms.py", "--output", self.out)
        p = run("qa.py", "--output", self.out, "--target", "50")
        p = run("assemble.py", "--output", self.out, "--eval-frac", "0.2")
        self.assertIn("[assemble]", p.stdout)
        train = open(os.path.join(self.out, "train.jsonl")).read().strip()
        ev = open(os.path.join(self.out, "eval.jsonl")).read().strip()
        self.assertTrue(train or ev, "both splits empty for a 1-doc corpus")


class UnitTests(unittest.TestCase):
    """Direct unit tests of the shared helpers."""

    def test_clean_text_strips_noise(self):
        from common import clean_text
        t = clean_text("Prashant Kirad\nPage 3\nHello world.\nwww.x.com")
        self.assertNotIn("Prashant", t)
        self.assertNotIn("Page 3", t)
        self.assertNotIn("www.x.com", t)
        self.assertIn("Hello world.", t)

    def test_clean_text_splits_glued_words(self):
        from common import clean_text
        self.assertIn(", ", clean_text("red,green seeded"))

    def test_unwrap_joins_wrapped_sentences(self):
        from common import unwrap
        text = "The Fundamental Theorem of Arithmetic says that every composite\nnumber can be factorised as a product of primes."
        out = unwrap(text)
        self.assertIn("every composite number can be factorised", out)

    def test_unwrap_keeps_bullets_separate(self):
        from common import unwrap
        text = "Some intro line here that is long enough to be a wrapped line\n- bullet one\n- bullet two"
        out = unwrap(text)
        self.assertIn("- bullet one", out)
        self.assertIn("- bullet two", out)

    def test_unwrap_joins_dangling_lines(self):
        from common import unwrap
        text = "The distance of a point from the y-axis is called its\ny-coordinate."
        out = unwrap(text)
        self.assertIn("called its y-coordinate", out)

    def test_is_bad_answer_rejects_fragments(self):
        from common import is_bad_answer
        self.assertTrue(is_bad_answer("Short."))
        self.assertTrue(is_bad_answer("a" * 800))
        self.assertFalse(is_bad_answer(
            "A polynomial in one variable x is an expression of the form "
            "p(x) = a_n x^n plus a constant term a_0."))

    def test_topic_of_ignores_questions(self):
        from qa import topic_of
        self.assertNotIn("can you", topic_of("Can you guess how heights are found?").lower())
        self.assertTrue(len(topic_of("The tangent to a circle is a special case").split()) <= 9)

    def test_shingles_overlap(self):
        from common import shingles
        a = "the quick brown fox jumps over the lazy dog again and again"
        b = "the quick brown fox jumps over the lazy cat again and again"
        c = "completely different content about polynomials and their degree"
        # one changed word invalidates up to n shingles of an n-gram set
        self.assertGreater(len(shingles(a) & shingles(b)) / len(shingles(a)), 0.4)
        self.assertEqual(len(shingles(a) & shingles(c)), 0)

    def test_detect_subject(self):
        from common import detect_subject
        math = ("polynomial equation triangle circle radius coordinate probability "
                "theorem surface area volume algebra geometry quadratic factor")
        sci = ("atom electron chemical reaction acid salt photosynthesis respiration "
               "magnetic electric current voltage resistance ozone ecosystem")
        self.assertEqual(detect_subject(math, ["Math", "Science"]), "Math")
        self.assertEqual(detect_subject(sci, ["Math", "Science"]), "Science")
        self.assertEqual(detect_subject("zzz qqq", ["Math"]), "General")

    def test_config_defaults_and_overrides(self):
        from common import load_config
        cfg = load_config(None)
        self.assertIn("qa", cfg)
        self.assertEqual(cfg["qa"]["per_atom"], 2)
        cfg2 = load_config(None, {"qa": {"target": 5}})
        self.assertEqual(cfg2["qa"]["target"], 5)
        self.assertEqual(cfg2["qa"]["per_atom"], 2, "override must not wipe siblings")

    def test_config_file_overrides(self):
        from common import load_config
        p = os.path.join(tempfile.mkdtemp(), "c.json")
        write(p, json.dumps({"qa": {"target": 42}}))
        cfg = load_config(p)
        self.assertEqual(cfg["qa"]["target"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
