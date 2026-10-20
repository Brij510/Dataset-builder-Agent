# Design Notes

Why the pipeline is shaped the way it is. Each decision below was forced by a
measured failure, not by preference.

## The pipeline is six stages, not one

```
raw docs -> extract -> dedupe -> atoms -> qa -> split -> validate
```

Splitting it this way buys three things:

- **Resumability.** Extraction is the slow stage (minutes per thousand pages).
  It caches per file, so re-running after adding documents only parses the new
  ones. Everything downstream is seconds.
- **Inspectability.** Each stage has a human-readable artifact. When the output
  is bad you can look at `corpus.json`, then `atoms.jsonl`, then `qa_all.jsonl`
  and see exactly where it went wrong.
- **Testability.** Every stage can be tested against a synthetic corpus in
  isolation, which is how the 45 tests in `tests/` are organised.

## Processes, not threads

`pdfplumber` is pure Python and holds the GIL. A `ThreadPoolExecutor` over PDF
parsing gives **no speedup** - eight threads on four cores parsed a 794-document
corpus at the same rate as one. `ProcessPoolExecutor` gives near-linear scaling
on independent files.

This was measured, not assumed: switching to processes cut a 25-minute
extraction to a few minutes.

`maxtasksperchild` and a sequential fallback matter in practice, because a
worker that dies on a pathological PDF breaks the whole pool.

## Text blocks are reconstructed before anything else

Textbooks put worked examples in boxed sidebars beside the main paragraph.
`page.extract_text()` reads the page in visual order and splices the two
together:

> 'of the pair of linear equations, i.e., the number of pants she purchased is 1'

pdfplumber's `use_text_flow=True` and `layout=True` produce the same result -
the interleaving is in the page's character ordering, not in the extraction
mode.

`scripts/pdfcols.py` reconstructs the page's *text blocks*: groups of lines
that sit close together vertically and start at roughly the same x. Each block
is emitted whole, so a sidebar stays in one piece. Genuine two-column pages
are detected (left and right blocks at overlapping heights) and read column by
column.

Measured on a 66-document NCERT slice:

| | candidate sentences | atoms | yield |
|---|---|---|---|
| `page.extract_text()` | 3,928 | 1,114 | 28.4% |
| block-aware | 2,780 | 1,165 | 41.9% |
| block-aware + enumerator strip | 2,780 | 1,311 | **47.2%** |

The block-aware version removes 1,148 spliced-garbage sentences and lifts the
atom yield by half. The gain is not only yield: those spliced lines were
*silently wrong* text that no filter could have repaired.

## Line unwrapping comes before sentence splitting

PDF extraction emits one *visual* line per row. A single sentence is usually
split across three or four lines. Splitting into sentences without first
re-joining the wrapped lines produces fragments like:

> "The distance of a point from the y-axis is called its"

...which then become training questions. `common.unwrap()` re-joins lines
before splitting, using three signals:

- a line ending in sentence punctuation closes a unit
- a bullet or numbered item always starts a new unit (lists must not collapse
  into one blob)
- a line ending in a *dangling* word (`its`, `called`, `which`, `of`) is
  grammatically unfinished, so the next line continues it even if that line is
  short

The third rule is what fixed the fragment above.

## Knowledge atoms, not sentences

A sentence becomes a typed atom (`definition`, `formula`, `compare`, `cause`,
`process`, `temporal`, `example`, `rule`, `purpose`, `list`, `fact`), and the
type decides which question templates are generated from it. This is what turns
a corpus into a *curriculum* rather than a transcript.

Two filters keep the atoms clean:

- `looks_complete()` rejects fragments: lowercase start, dangling conjunction,
  no verb, table-like comma density.
- Source sentences that are themselves questions are dropped - a question is
  not a fact.

## Questions ask about a topic, not quote the answer

The first version embedded the whole source statement in the question:

> "State the following fact. <entire statement>"

Measured on a real corpus, **79% of training rows were echo rows** - the
question already contained the answer. A model trained on that learns to
repeat the prompt, not to answer.

The generator now extracts the *subject noun phrase* of the statement and asks
about that ("Explain the factors affecting..."). Echo rows fell from 79% to
**5.8%** on a 66-document NCERT slice.

A small fraction of rows still quote the statement on purpose - they teach
paraphrase. Those are tagged `kind: "restate"` and excluded from the echo
metric so the number stays meaningful.

## Jaccard, not containment, for duplicate detection

Containment (`inter / min`) fires on any two documents that share a lot of
boilerplate. On a real corpus of question papers it flagged **2,372 pairs**,
almost all false positives: two different papers share headers and formatting
but not content.

Jaccard (`inter / union`) flagged **148** - the real duplicates (a textbook
and its scan, a "notes (2).pdf" next to "notes.pdf"). Both `dedupe.py` and the
safety net in `assemble.py` use Jaccard.

Short documents are excluded from comparison entirely, since their shingle
sets are too small to be meaningful.

## Leakage is measured by distinctive content, not raw shingles

A raw 5-gram overlap counts boilerplate as leakage. Two answers that both
say "is a unit of measurement" share five grams, and a corpus full of
textbook phrasing produces overlap numbers that look alarming while the
actual content is different.

The filter is therefore IDF-weighted: each shingle is weighted by how rare it
is across the corpus, and only distinctive content can push the overlap over
the limit. On a 66-document NCERT slice this moved the reported overlap from
0.080 (raw) to 0.041 (weighted) - the difference was boilerplate, not
leakage.

## The eval set is subject-stratified

Holding out documents at random can leave the eval set dominated by whichever
subject happens to have small documents, which makes the score meaningless.
Each subject now gets a quota proportional to its share of the rows, and
whole documents are taken until the quota is met. Smallest-fitting documents
are preferred so a single large document cannot dominate.

Two safeguards fall out of the document-holdout semantics:

- a held-out document joins the eval set **whole** - truncating it would drop
  the leftover rows, since the whole document is excluded from train
- if a subject's only document would be held out, it is handed back, so no
  subject disappears from training

## The split is document-level, then shingle-filtered

Holding out whole *documents* is the strongest available guarantee that the
model has not memorised the evaluation material. But it is not sufficient on
its own - a fact can be restated in several different books - so a symmetric
5-gram shingle filter removes any train row that overlaps an eval answer by
more than `max_overlap`, measured in **both** directions (a long eval answer
and a short train answer must not slip through).

The printed `max eval/train answer overlap` must be at or below the limit.

## `max_per_answer` is 1 by default

Allowing two questions to share one answer doubles the row count, but both
questions are about the same topic, so they are near-duplicates by
construction. Measured: `max_per_answer: 2` produced **256** near-duplicate
question pairs against **81** at a setting of 1.

If you need more rows, grow the corpus (more pages per document, or a wider
`require_complete` setting) rather than re-asking the same fact. `validate.py`
now separates the two causes so the warning is actionable.

## Subject auto-detection, but overridable

A folder of mixed documents has no reliable path-prefix mapping. Keyword
scoring over document content gives sensible labels with no configuration, and
`subject_map` still wins when present. `subject_keywords` in the config adds
your own labels without editing the code.

## Determinism

The same input tree produces byte-identical `train.jsonl` and `eval.jsonl` on
every run. The seed is fixed, every iteration order is sorted, and the
process-pool results are written into a dict keyed by path. There is a test
for this, because a dataset that changes between runs makes every comparison
afterwards meaningless.

## Source text can contain ChatML markers

A document that itself shows ChatML examples (a tutorial, a spec, a prompt
guide) contains the literal string `<|im_end|>`. Written into a row verbatim,
that injects a stray marker and breaks the format for the whole training
file - one bad row is enough.

`common.chatml()` strips literal `<|im_start|>` / `<|im_end|>` sequences from
both the instruction and the answer before writing. This was found by the
validator on a real corpus, not by inspection.

## The validator and the splitter must measure the same thing

Leakage is reported by two scripts. If they use different measures, one will
pass while the other fails, and the disagreement looks like a bug in the
dataset rather than in the tooling. Both now:

- weight shingles by IDF over the same basis (the final train+eval sets)
- use the same `max_overlap` limit

so `max eval/train answer overlap` means the same number in both.

## Quality gates

`validate.py` fails on: invalid JSON, broken ChatML, missing keys, CSV/JSONL
row-count mismatch, watermark noise, duplicate instructions, eval/train
instruction overlap, answer-overlap above the limit.

It warns on: echo rows, near-duplicate questions, low template diversity,
short answers. `--strict` turns warnings into failures, which is what CI should
use.

## What is deliberately not here

- **No OCR.** Scanned PDFs produce no text. Run `ocrmypdf` first.
- **No LLM-written answers.** The LLM backend may rewrite questions; answers
  always come from the extracted text. This is what keeps the dataset grounded.
- **No synthetic templates per domain.** The archetype set is generic. A
  medical corpus would want a `contraindication` kind - add it in `atoms.py`
  and `qa.py`.
