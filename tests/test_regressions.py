#!/usr/bin/env python3
"""Regression tests for defects found while hardening the pipeline.

Each test names the behaviour that was broken and pins it down, so a future
change cannot silently reintroduce the bug.
"""
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

PY = sys.executable


def run(script, *args):
    cmd = [PY, os.path.join(SCRIPTS, script)] + [str(a) for a in args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise AssertionError(f"{script} failed ({p.returncode})\n"
                             f"STDOUT:\n{p.stdout[-2500:]}\nSTDERR:\n{p.stderr[-2500:]}")
    return p


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_row(source, subject, i):
    seed = hash((source, subject, i)) % 997
    flavour = ["measured", "derived", "observed", "recorded", "defined",
               "compared", "summarised", "explained", "traced"][seed % 9]
    thing = f"idea{seed % 89} aspect{seed % 31}"
    instr = f"Question {i} on {flavour} {thing} in {subject}"
    out = (f"In {subject}, {flavour} {thing} is presented as a point that "
           f"carries its own meaning within {source}, and it is worth "
           f"noting because it connects to {flavour} {thing} elsewhere.")
    return {"text": f"<|im_start|>user\n{instr}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n{out}<|im_end|>",
            "subject": subject, "instruction": instr, "input": "",
            "output": out, "source": source, "origin": "atoms", "kind": "fact"}


class SubjectResolution(unittest.TestCase):
    """The most specific path prefix must win, and detection must be possible."""

    def test_longest_prefix_wins(self):
        from common import subject_for
        smap = {"Math/": "Math", "Math/NCERT/": "Math Advanced",
                "Science/": "Science"}
        self.assertEqual(subject_for("Math/NCERT/ch1.pdf", smap), "Math Advanced")
        self.assertEqual(subject_for("Math/NCERT/x/y.pdf", smap), "Math Advanced")
        self.assertEqual(subject_for("Math/other.pdf", smap), "Math")
        self.assertEqual(subject_for("Science/notes.pdf", smap), "Science")

    def test_prefix_does_not_match_inside_a_word(self):
        from common import subject_for
        smap = {"Math": "Math"}
        # "AppliedMathematics" starts with "Math" but is a different folder
        self.assertEqual(subject_for("AppliedMathematics/x.pdf", smap), "General")
        self.assertEqual(subject_for("Math/x.pdf", smap), "Math")

    def test_detection_can_actually_return_a_label(self):
        """A pool of only the default label must not silence detection."""
        from common import detect_subject
        text = ("polynomial equation theorem triangle circle radius factor "
                "divisor quadratic algebra geometry trigonometry")
        # a pool containing real labels works
        self.assertEqual(detect_subject(text, ["Math", "Science"], "General"), "Math")
        # a pool with no real label must not crash or invent one
        self.assertEqual(detect_subject(text, ["General"], "General"), "General")
        # an empty pool falls back to the built-in table
        self.assertEqual(detect_subject(text, None, "General"), "Math")


class LeakageMeasure(unittest.TestCase):
    """Overlap must be IDF-weighted, symmetric, and free of boilerplate bias."""

    def test_universal_boilerplate_carries_no_weight(self):
        from common import build_idf, shingles
        # the same boilerplate in every answer, one distinctive word each
        texts = [f"The value is defined as the special quantity called term{i}."
                 for i in range(20)]
        idf = build_idf(texts, 5)
        boiler = next(g for g in shingles(texts[0], 5)
                      if g in idf and idf[g] == 0.0)
        self.assertEqual(idf[boiler], 0.0,
                         "a shingle present in every answer must weigh nothing")

    def test_boilerplate_does_not_fake_leakage(self):
        from common import (build_idf, build_overlap_index, max_overlap,
                            shingles)
        texts = [f"The value is defined as the special quantity called term{i}."
                 for i in range(20)]
        idf = build_idf(texts, 5)
        a, b = shingles(texts[0], 5), shingles(texts[1], 5)
        ov = max_overlap(a, build_overlap_index([b]), [b], idf, symmetric=True)[0]
        self.assertLess(ov, 0.10,
                        "shared boilerplate must not read as leakage")

    def test_overlap_is_symmetric(self):
        from common import (build_idf, build_overlap_index, max_overlap,
                            shingles)
        # A long answer and a shorter one that shares its distinctive head.
        # Measured short-to-long the overlap is near zero; measured the other
        # way it is near one. The symmetric measure must take the worse side.
        long_a = ("the mitochondrion releases energy from glucose molecules "
                  "inside every living cell of the organism during respiration")
        short_b = ("the mitochondrion releases energy from glucose molecules "
                   "during respiration in living cells")
        texts = [long_a, short_b, "an unrelated sentence about the postal service"]
        idf = build_idf(texts, 5)
        sa, sb = shingles(long_a, 5), shingles(short_b, 5)
        inv = build_overlap_index([sb])
        one_way = max_overlap(sa, inv, [sb], idf, symmetric=False)[0]
        other_way = max_overlap(sb, build_overlap_index([sa]), [sa], idf,
                                 symmetric=False)[0]
        sym = max_overlap(sa, inv, [sb], idf, symmetric=True)[0]
        self.assertNotAlmostEqual(one_way, other_way, places=2,
                                   msg="the fixture must be asymmetric")
        self.assertAlmostEqual(sym, max(one_way, other_way), places=6)
        self.assertGreater(sym, 0.10,
                           "a one-directional leak must still be caught")

    def test_distinctive_content_is_detected(self):
        from common import (build_idf, build_overlap_index, max_overlap,
                            shingles)
        # Two answers that state the same fact, so they share a run of
        # distinctive content rather than mere boilerplate.
        a = ("the enzyme amylase breaks down starch into simple sugars in "
            "the small intestine during the digestion of food")
        b = ("the enzyme amylase breaks down starch into simple sugars in "
            "the small intestine during digestion")
        texts = [a, b, "an unrelated sentence about the postal service"]
        idf = build_idf(texts, 5)
        sa, sb = shingles(a, 5), shingles(b, 5)
        self.assertTrue(sa & sb, "the fixture must share a 5-gram run")
        ov = max_overlap(sa, build_overlap_index([sb]), [sb], idf, symmetric=True)[0]
        self.assertGreater(ov, 0.10, "genuinely similar content must register")


class SplitIntegrity(unittest.TestCase):
    """No row may be lost, and the two scripts must agree on the overlap."""

    def _rows(self, source, subject, i):
        return make_row(source, subject, i)

    def test_no_document_is_lost_between_splits(self):
        """No source document may vanish from both splits.

        Individual rows may legitimately be dropped - a train row that
        duplicates eval material is exactly what the leak filter is for -
        but a whole document disappearing is silent data loss. The eval set
        must also cover every subject.
        """
        tmp = tempfile.mkdtemp(prefix="dsb_lost_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            rows = []
            for subject in ("Math", "Science", "History"):
                for d in range(4):
                    for i in range(12):
                        rows.append(self._rows(f"{subject}/doc{d}.txt", subject, i))
            write_jsonl(os.path.join(out, "qa_all.jsonl"), rows)
            run("assemble.py", "--output", out, "--eval-frac", "0.1")
            tr = read_jsonl(os.path.join(out, "train.jsonl"))
            ev = read_jsonl(os.path.join(out, "eval.jsonl"))
            all_docs = {r["source"] for r in rows}
            kept_docs = {r.get("source") for r in tr} | {r.get("source") for r in ev}
            missing = all_docs - kept_docs
            self.assertEqual(missing, set(),
                             f"these documents vanished entirely: {missing}")
            self.assertEqual({r["subject"] for r in ev},
                             {"Math", "Science", "History"},
                             "the eval set must cover every subject")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_subject_with_one_document_still_reaches_train(self):
        """A single-document subject must not vanish from the training set."""
        tmp = tempfile.mkdtemp(prefix="dsb_solo_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            rows = []
            for subject in ("Math", "Science", "Economics"):
                for d in range(3):
                    for i in range(10):
                        rows.append(self._rows(f"{subject}/doc{d}.txt", subject, i))
            write_jsonl(os.path.join(out, "qa_all.jsonl"), rows)
            run("assemble.py", "--output", out, "--eval-frac", "0.15")
            tr = read_jsonl(os.path.join(out, "train.jsonl"))
            ev = read_jsonl(os.path.join(out, "eval.jsonl"))
            self.assertIn("Economics", {r["subject"] for r in tr},
                          "a subject lost its only document and vanished")
            self.assertIn("Economics", {r["subject"] for r in ev},
                          "a subject must also appear in the eval set")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_assemble_and_validate_agree_on_the_overlap(self):
        """The splitter may not ship a dataset its own validator rejects."""
        import re
        tmp = tempfile.mkdtemp(prefix="dsb_agree_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            rows = []
            for subject in ("Math", "Science", "History"):
                for d in range(4):
                    for i in range(12):
                        rows.append(self._rows(f"{subject}/doc{d}.txt", subject, i))
            write_jsonl(os.path.join(out, "qa_all.jsonl"), rows)
            a = run("assemble.py", "--output", out, "--eval-frac", "0.1")
            v = run("validate.py", "--output", out)
            self.assertIn("All checks passed", v.stdout,
                          f"validate rejected what assemble shipped\n"
                          f"{a.stdout[-1500:]}\n{v.stdout[-1500:]}")
            am = re.search(r"max eval/train answer overlap: ([0-9.]+)", a.stdout)
            vm = re.search(r"max answer overlap ([0-9.]+)", v.stdout)
            self.assertIsNotNone(am)
            self.assertIsNotNone(vm)
            self.assertAlmostEqual(float(am.group(1)), float(vm.group(1)),
                                   places=3,
                                   msg="assemble and validate report different overlaps")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Provenance(unittest.TestCase):
    """Rows must keep the pointer back to their source document."""

    def test_rows_keep_their_source(self):
        tmp = tempfile.mkdtemp(prefix="dsb_prov_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            rows = []
            for d in range(4):
                for i in range(10):
                    rows.append(make_row(f"Math/doc{d}.txt", "Math", i))
            write_jsonl(os.path.join(out, "qa_all.jsonl"), rows)
            run("assemble.py", "--output", out, "--eval-frac", "0.2")
            for split in ("train", "eval"):
                for r in read_jsonl(os.path.join(out, f"{split}.jsonl")):
                    self.assertTrue(r.get("source"),
                                    f"{split} row lost its source pointer")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Grounding(unittest.TestCase):
    """Answers must be supported by the source text."""

    def test_a_verbatim_answer_scores_full_marks(self):
        from ground import build_index, score_row
        corpus = {"a.txt": ("the enzyme amylase breaks down starch into "
                            "simple sugars inside the small intestine")}
        per_doc, whole = build_index(corpus)
        row = {"output": ("amylase breaks down starch into simple sugars "
                          "inside the small intestine"), "source": "a.txt"}
        own, anyw, n = score_row(row, per_doc, whole)
        self.assertEqual(own, 1.0)
        self.assertEqual(anyw, 1.0)
        self.assertGreater(n, 0)

    def test_a_fabricated_answer_scores_low(self):
        from ground import build_index, score_row
        corpus = {"a.txt": ("the enzyme amylase breaks down starch into "
                            "simple sugars inside the small intestine")}
        per_doc, whole = build_index(corpus)
        row = {"output": ("photosynthesis converts light energy into chemical "
                          "energy stored inside chloroplasts of green leaves"),
               "source": "a.txt"}
        own, anyw, n = score_row(row, per_doc, whole)
        # a shared word like "energy" may still appear, but the answer as a
        # whole must read as ungrounded
        self.assertLess(own, 0.35, "a fabricated answer must not look grounded")
        self.assertLess(anyw, 0.35)

    def test_own_document_is_scored_strictly(self):
        """A word found only in ANOTHER document must not count as grounded."""
        from ground import build_index, score_row
        corpus = {
            "a.txt": "the mitochondrion releases energy from glucose molecules",
            "b.txt": "chlorophyll absorbs light energy in the chloroplast",
        }
        per_doc, whole = build_index(corpus)
        row = {"output": "chlorophyll absorbs light energy in the chloroplast",
               "source": "a.txt"}
        own, anyw, n = score_row(row, per_doc, whole)
        self.assertLess(own, 0.35, "wrong-document words must not count")
        self.assertGreater(anyw, own,
                           "the corpus fallback should still find the words")

    def test_ground_stage_runs_on_a_dataset(self):
        tmp = tempfile.mkdtemp(prefix="dsb_ground_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            with open(os.path.join(out, "corpus.json"), "w", encoding="utf-8") as f:
                json.dump({"a.txt": "the small intestine digests the food"}, f)
            row = make_row("a.txt", "Biology", 0)
            row["output"] = "the small intestine digests the food"
            write_jsonl(os.path.join(out, "train.jsonl"), [row])
            write_jsonl(os.path.join(out, "eval.jsonl"), [row])
            p = run("ground.py", "--output", out)
            self.assertIn("own-document grounding", p.stdout)
            self.assertIn("high (>=0.85)", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class QuestionQuality(unittest.TestCase):
    """A concept must be a term, not a fragment of a sentence."""

    def test_a_quantifier_lead_is_not_a_concept(self):
        from atoms import classify
        cfg = {"atoms": {"min_sentence": 30, "max_sentence": 400,
                         "min_words": 5, "require_complete": True}}
        # "Many of our cities ..." is a sentence, not a term
        kind, concept, stmt = classify(
            "Many of our cities have such resistant crops and dry farming "
            "techniques in practice today.", cfg)
        self.assertNotEqual(kind, "definition",
                            "a quantifier lead must not become a concept")
        self.assertNotEqual(concept.lower(), "many of our cities")

    def test_a_real_term_is_still_a_definition(self):
        from atoms import classify
        cfg = {"atoms": {"min_sentence": 30, "max_sentence": 400,
                         "min_words": 5, "require_complete": True}}
        kind, concept, stmt = classify(
            "A polynomial is an expression of the form p(x) = a_n x^n plus "
            "a constant term written in one variable.", cfg)
        self.assertEqual(kind, "definition")
        self.assertEqual(concept, "A polynomial")

    def test_a_truncated_phrase_is_not_a_concept(self):
        from atoms import classify
        cfg = {"atoms": {"min_sentence": 30, "max_sentence": 400,
                         "min_words": 5, "require_complete": True}}
        kind, concept, stmt = classify(
            "The three main sectors of the economy of the nation are the "
            "primary, secondary and tertiary sectors respectively.", cfg)
        self.assertNotEqual(kind, "definition",
                            "a phrase ending in a preposition is not a term")

    def test_a_fragment_is_not_a_topic(self):
        from qa import topic_of
        # the topic must not be a quantifier-led fragment
        t = topic_of("Many of our cities have resistant crops and dry "
                     "farming techniques in practice today.")
        self.assertFalse(t.lower().startswith("many of our cities"),
                         f"fragment used as a topic: {t!r}")


class DegenerateCorpora(unittest.TestCase):
    """A tiny corpus must not crash the pipeline or empty the training set."""

    def test_validate_survives_an_empty_training_set(self):
        tmp = tempfile.mkdtemp(prefix="dsb_empty_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            write_jsonl(os.path.join(out, "train.jsonl"), [])
            write_jsonl(os.path.join(out, "eval.jsonl"), [make_row("a.txt", "Math", 0)])
            p = run("validate.py", "--output", out)
            self.assertIn("All checks passed", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_single_row_lands_in_train_not_eval(self):
        tmp = tempfile.mkdtemp(prefix="dsb_one_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            write_jsonl(os.path.join(out, "qa_all.jsonl"), [make_row("a.txt", "Math", 0)])
            run("assemble.py", "--output", out, "--eval-frac", "0.07")
            tr = read_jsonl(os.path.join(out, "train.jsonl"))
            ev = read_jsonl(os.path.join(out, "eval.jsonl"))
            self.assertEqual(len(tr), 1, "the only row must stay in train")
            self.assertEqual(len(ev), 0, "a single row cannot also be eval")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_training_set_is_never_emptied(self):
        tmp = tempfile.mkdtemp(prefix="dsb_notrain_")
        try:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            write_jsonl(os.path.join(out, "qa_all.jsonl"),
                        [make_row("a.txt", "Math", 0)])
            run("assemble.py", "--output", out, "--eval-frac", "0.5")
            tr = read_jsonl(os.path.join(out, "train.jsonl"))
            self.assertTrue(tr, "assemble produced no training rows at all")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
