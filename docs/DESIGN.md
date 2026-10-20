# Design Notes

Why the pipeline is shaped the way it is. Each decision below was forced by a
measured failure, not by preference.

## The pipeline is seven stages, not one

```
raw docs -> extract -> dedupe -> atoms -> qa -> split -> validate -> ground
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
the limit.

### The IDF floor must reach zero

The first implementation used `log(N / df) + 1`. That "+1" is fatal. A shingle
present in *every* answer still carries a full unit of weight, and a 5-gram
window over a shared template produces many such shingles. In a 144-row
synthetic corpus, 12 of the 16 shingles shared by two unrelated answers had
document frequency 144 - they were pure phrasing - and the measure reported
0.17 overlap between two answers that share no content at all.

Switching to the smooth form `log((N+1) / (df+1))` gives a universally common
shingle a weight of exactly **zero**, so boilerplate cannot contribute to the
numerator or the denominator. The same pair then scored 0.016, and genuinely
similar answers still scored above 0.10. A shingle that appears in every
answer proves nothing, so it must be worth nothing.

The fallback matters too: when a row's shingles are *all* universal, the
IDF-weighted denominator is zero. Reporting 0.0 there would hide a genuine
duplicate behind a corpus made entirely of boilerplate, so the raw shingle
ratio is used instead.

## Leakage is measured in both directions, against one row

A one-directional measure misses leaks. A short train answer sitting inside a
long eval answer scores near 1.0 measured train-to-eval, but near 0.0
measured eval-to-train. The measure is now the **worse** of the two
directions.

The comparison is also made against the single row that actually conflicts,
not against the union of all train shingles. The union overstates overlap:
two train rows that each share a different fragment with an eval row can push
the union measure over the limit even though no single train row really
duplicates it.

## A leak is resolved by dropping the train row

The train set is abundant and the eval set is scarce. When a train row and an
eval row conflict, the train row is dropped. Dropping the eval row instead
loses the only copy of that document's material, and when it is the last
document of a subject, that subject disappears from the eval set entirely.

The earlier implementation did the opposite - it dropped eval rows to satisfy
the limit, and then handed back documents whose subjects had vanished from
training. The result was a dataset that `assemble.py` reported as
`max eval/train answer overlap: 0.192 (limit 0.1)` while `validate.py`
rejected the same files with "9 eval rows exceed the overlap limit". The two
scripts measured the same thing differently and disagreed.

Both scripts now share one implementation (`common.symmetric_overlap`,
`common.max_overlap`) and one IDF basis, so they cannot disagree. A
regression test asserts that the number `assemble.py` prints is the number
`validate.py` prints, and that the validator accepts what the splitter
shipped.

## A document must never vanish from both splits

When a subject's only document was held out for eval, the hand-back logic
removed it from eval - but the leak index had been built before that
removal, so the handed-back rows were still judged against their own stale
index entries and dropped from train as well. The document ended up in
neither split, and the subject disappeared from the dataset entirely. The
index is now rebuilt from the current eval set every time it is consulted.

A safety net catches any row that still matches neither split. It is offered
back to train, but it must clear the same leak filter as every other train
row - returning a row that duplicates eval material would undo the guarantee
the script just made. A row that cannot clear the filter is a genuine
duplicate and is reported and dropped.

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

### The detection pool must contain real labels

The detector scores each candidate label from a pool and returns the best one
above a threshold. If the pool contains only the default label, the detector
can never return anything else - it is not "no match", it is "no candidate".

The pool is built from the config's `subjects` list, with the default label
removed, falling back to the built-in table when that list is empty or
unusable. Detection also used to be skipped entirely whenever a `subject_map`
was configured, which meant a corpus with a *partial* map got `General` for
everything the map did not cover. A path prefix is authoritative; detection
fills the gaps left by an unmatched prefix, so it runs either way.

On the 66-document NCERT slice this moved the atom mix from
`{'General': 538}` to `{'Math': 229, 'Social Science': 237, 'Economics': 34,
'Science': 33}`.

### The most specific path prefix wins

`subject_map` used to be scanned in dictionary order and the first prefix
found won, so `{"Math/": "Math", "Math/NCERT/": "Math Advanced"}` classified
everything under `Math/NCERT/` as plain `Math`. The longest matching prefix
wins now.

The match is also anchored to a path boundary. A bare `Math` prefix must not
match `AppliedMathematics/x.pdf`, which is a different subject entirely.
Prefixes that already end in `/` are unambiguous and need no boundary check.

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
- measure symmetrically, against the single row that actually conflicts
- use the same `max_overlap` limit

so `max eval/train answer overlap` means the same number in both. A
regression test asserts the two numbers are equal.

## Every row keeps its source

A row that cannot be traced back to the document it came from cannot be
audited, cannot be corrected, and cannot be removed when a source turns out
to be wrong. The `source` field was dropped when the final rows were written,
even though the split logic depended on it - the pointer was there and then
gone by the time it reached the file.

Rows now carry `source` and `origin` into `train.jsonl` / `eval.jsonl`, and
the CSV gained a matching `source` column. The validator treats a missing
source as a failure.

## Grounding is measured, not assumed

The pipeline never writes an answer that is not in the extracted text, so
every answer is grounded by construction. That is a guarantee about the
*code*, not about the *data*: a lossy PDF extraction can produce an answer
that reads as gibberish, and a hand-written `gold.jsonl` can contain
anything.

`ground.py` checks it independently, scoring every row by the share of its
content words that actually appear in the text it was extracted from. It
scores against the row's own document strictly, and against the whole corpus
as a weaker fallback - a fact restated in another book is still grounded, it
just is not attributable to this document.

Scoring rather than enforcing is deliberate. A hard threshold depends on the
corpus: a textbook restates its own phrasing, so a genuinely-grounded answer
can still score low when the extraction was lossy. The score is reported, and
`--drop-below` removes the weakest rows when the user wants a clean set.

## Quality gates

`validate.py` fails on: invalid JSON, broken ChatML, missing keys, CSV/JSONL
row-count mismatch, watermark noise, duplicate instructions, eval/train
instruction overlap, answer-overlap above the limit.

It warns on: echo rows, near-duplicate questions, low template diversity,
short answers. `--strict` turns warnings into failures, which is what CI should
use.

## What is deliberately not here

- **No OCR.** Scanned PDFs produce no text. Run `ocrmypdf` first. The
  extractor reports which documents yielded no usable text, because a silently
  skipped chapter is a silently missing chapter.
- **No LLM-written answers.** The LLM backend may rewrite questions; answers
  always come from the extracted text. This is what keeps the dataset grounded.
- **No synthetic templates per domain.** The archetype set is generic. A
  medical corpus would want a `contraindication` kind - add it in `atoms.py`
  and `qa.py`.
