"""Shared test helpers: build a synthetic corpus with known edge cases."""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))


def write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding, errors="ignore") as f:
        f.write(text)
    return path


def make_corpus(root):
    """Create a small corpus covering the edge cases the pipeline must survive."""
    write(os.path.join(root, "Math", "algebra.txt"), """
Chapter 1: Algebra
A polynomial in one variable x is an expression of the form p(x) = a_n x^n + ... + a_0
where the coefficients are real numbers and n is a non-negative integer.
The highest power of x in p(x) is called the degree of the polynomial.
The Fundamental Theorem of Arithmetic states that every composite number can be
factorised as a product of prime numbers.
A line which is parallel to the x-axis is called a horizontal line.
The slope of a line is the ratio of the vertical change to the horizontal change.
""")
    write(os.path.join(root, "Science", "acids.txt"), """
Acids, Bases and Salts
An acid is a substance that gives hydrogen ions in aqueous solution.
A base is a substance that accepts hydrogen ions in aqueous solution.
Neutralisation is the reaction of an acid with a base to form a salt and water.
Sodium hydroxide reacts with hydrochloric acid to produce sodium chloride and water.
The pH of an acidic solution is less than seven.
Red litmus paper turns blue in the presence of a base.
""")
    write(os.path.join(root, "Social Science", "democracy.txt"), """
Democracy and Power Sharing
Democracy is a form of government in which the rulers are elected by the people.
Power sharing is a characteristic feature of the political system in India.
The Indian Constitution provides for equality before the law for all citizens.
There are three tiers of Panchayati Raj in rural India.
Money and credit are the instruments which help in the growth of the economy.
Globalisation has led to the integration of the Indian economy with the world.
""")
    # --- edge case: empty file
    write(os.path.join(root, "Math", "empty.txt"), "")
    # --- edge case: CRLF line endings + BOM
    write(os.path.join(root, "Math", "crlf.txt"),
          "\ufeffA triangle is a polygon with three sides.\r\n"
          "The area of a triangle is half the base multiplied by the height.\r\n",
          encoding="utf-8-sig")
    # --- edge case: unicode + emoji
    write(os.path.join(root, "Science", "unicode.txt"),
          "Le français est une langue officielle de plusieurs pays. Ω ≈ 3.14 ∑ ∫ √ "
          "A resistor obeys Ohm's law and dissipates energy as heat. 二人行 ")
    # --- edge case: very long single line
    write(os.path.join(root, "Math", "longline.txt"),
          "The " + "very " * 200 + "long sentence about polynomials and their degree.\n")
    # --- edge case: no trailing newline
    write(os.path.join(root, "Math", "notrail.txt"),
          "A circle is a closed curve whose every point is at a fixed distance from a centre.")
    # --- edge case: near-duplicate of the algebra file (tests dedupe)
    write(os.path.join(root, "Math", "algebra_copy.txt"), """
Chapter 1: Algebra
A polynomial in one variable x is an expression of the form p(x) = a_n x^n + ... + a_0
where the coefficients are real numbers and n is a non-negative integer.
The highest power of x in p(x) is called the degree of the polynomial.
The Fundamental Theorem of Arithmetic states that every composite number can be
factorised as a product of prime numbers.
A line which is parallel to the x-axis is called a horizontal line.
The slope of a line is the ratio of the vertical change to the horizontal change.
""")
    # --- edge case: only noise
    write(os.path.join(root, "Science", "noise.txt"),
          "Page 1\nPage 2\nwww.example.com\n© 2026\n\n\n")
    return root


def tempdir():
    d = tempfile.mkdtemp(prefix="dsb_test_")
    return d


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)
