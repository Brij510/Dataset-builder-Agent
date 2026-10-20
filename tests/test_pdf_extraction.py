#!/usr/bin/env python3
"""Tests for the block-aware PDF extraction."""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

from fixtures import write  # noqa: E402


class BlockExtraction(unittest.TestCase):
    def test_blocks_keep_a_boxed_sidebar_intact(self):
        """A margin example must not be spliced onto the main paragraph."""
        from pdfcols import _blocks_from_lines

        page_width = 600.0

        def line(text, top, x0, w=200):
            return {"text": text, "top": top, "bottom": top + 12,
                    "x0": x0, "x1": x0 + w}

        # main body on the left, boxed example offset to the right
        lines = [
            line("The two lines intersect at the point (1, 0).", 100, 40),
            line("So x = 1, y = 0 is the required solution", 112, 40),
            line("Solution: Let us denote the number of pants by x and the number", 130, 330),
            line("of skirts by y. Then the equations formed are", 142, 330),
        ]
        blocks = _blocks_from_lines(lines, page_width)
        joined = " ".join(" ".join(l["text"] for l in b) for b in blocks)
        for b in blocks:
            text = " ".join(l["text"] for l in b)
            self.assertNotIn("required solution Solution:", text)
        self.assertIn("Solution: Let us denote the number of pants by x", joined)

    def test_blocks_split_when_a_box_starts_far_away(self):
        """A boxed example offset to the right must form its own block."""
        from pdfcols import _blocks_from_lines
        page_width = 600.0

        def line(text, top, x0):
            return {"text": text, "top": top, "bottom": top + 12,
                    "x0": x0, "x1": x0 + 200}

        lines = [
            line("main body sentence one", 100, 40),
            line("main body sentence two", 112, 40),
            line("boxed example line one", 104, 340),
            line("boxed example line two", 116, 340),
        ]
        blocks = _blocks_from_lines(lines, page_width)
        self.assertGreaterEqual(len(blocks), 2, "sidebar and body were merged")
        for b in blocks:
            text = " ".join(l["text"] for l in b)
            self.assertFalse("main body" in text and "boxed example" in text,
                             "body and sidebar ended up in one block")

    def test_two_column_page_is_detected(self):
        from pdfcols import _blocks_from_lines, _is_two_column
        page_width = 600.0

        def line(text, top, x0):
            return {"text": text, "top": top, "bottom": top + 12, "x0": x0, "x1": x0 + 180}

        lines = []
        for i in range(6):
            lines.append(line(f"left column line {i}", 100 + i * 14, 40))
        for i in range(6):
            lines.append(line(f"right column line {i}", 100 + i * 14, 330))
        blocks = _blocks_from_lines(lines, page_width)
        self.assertTrue(_is_two_column(blocks, page_width))

    def test_single_column_page_is_not_split(self):
        from pdfcols import _blocks_from_lines, _is_two_column
        page_width = 600.0

        def line(text, top, x0):
            return {"text": text, "top": top, "bottom": top + 12, "x0": x0, "x1": x0 + 400}

        lines = [line(f"body line {i}", 100 + i * 14, 60) for i in range(10)]
        blocks = _blocks_from_lines(lines, page_width)
        self.assertFalse(_is_two_column(blocks, page_width))

    def test_furniture_lines_are_dropped(self):
        from pdfcols import FURNITURE
        for junk in ["12", "Page 4", "Table 3.3", "Fig. 3.2", "EXERCISE 3.1", "(i)"]:
            self.assertTrue(FURNITURE.match(junk), junk)
        for real in ["The two lines intersect at the point (1, 0).",
                     "Plot the points and draw the lines passing through them."]:
            self.assertFalse(FURNITURE.match(real), real)


class ExtractIntegration(unittest.TestCase):
    def test_extract_handles_a_missing_and_a_corrupt_file(self):
        tmp = tempfile.mkdtemp(prefix="dsb_pdf_")
        try:
            src = os.path.join(tmp, "docs")
            out = os.path.join(tmp, "out")
            write(os.path.join(src, "a.txt"),
                  "A polynomial in one variable is an expression of a certain form.\n")
            with open(os.path.join(src, "broken.pdf"), "wb") as f:
                f.write(b"not a pdf")
            import subprocess
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "extract.py"),
                                "--input", src, "--output", out, "--workers", "1"],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            import json
            corpus = json.load(open(os.path.join(out, "corpus.json")))
            self.assertTrue(any("polynomial" in v.lower() for v in corpus.values()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
