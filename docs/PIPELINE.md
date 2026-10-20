# Pipeline Reference

Stage contracts, data formats, and implementation notes.

## Stage contracts

| Stage | Script | Reads | Writes | Resumable |
|---|---|---|---|---|
| 1 | `extract.py` | raw docs tree | `corpus.json` | yes (per-file cache) |
| 2 | `dedupe.py` | `corpus.json` | `corpus.json` (pruned) | yes (idempotent) |
| 3 | `atoms.py` | `corpus.json` | `atoms.jsonl` | yes (recompute, cheap) |
| 4 | `qa.py` | `atoms.jsonl`, `corpus.json` | `qa_all.jsonl` | merges with the existing file |
| 5 | `assemble.py` | `qa_all.jsonl`, `gold.jsonl` | `train.*`, `eval.*` | yes (deterministic) |
| 6 | `validate.py` | `train.*`, `eval.*` | - | n/a |
| 7 | `ground.py` | `train.*`, `eval.*`, `corpus.json` | `*_grounded.jsonl` (optional) | n/a |

The whole chain is **deterministic**: the same input tree produces byte-identical
`train.jsonl` and `eval.jsonl` on every run. There is a test for this.

## Data formats

### `corpus.json`

```json
{
  "Math/NCERT/CH-2 POLYNOMIALS.pdf": "cleaned plain text ...",
  "Science/Notes/Electricity.pdf": "cleaned plain text ..."
}
```

Keys are paths relative to `--input`. Cleaning removes watermarks, page numbers,
bullet artifacts and de-hyphenates words broken across lines.

### `atoms.jsonl`

```json
{"kind": "definition", "concept": "Polynomials", "statement": "the expression p(x) = ...",
 "subject": "Math", "source": "Math/NCERT/CH-2 POLYNOMIALS.pdf"}
```

`kind` drives question selection. See the table in SKILL.md.

### `qa_all.jsonl`

```json
{"instruction": "Define Polynomials.", "input": "", "output": "Polynomials are ...",
 "subject": "Math", "source": "Math/NCERT/CH-2 POLYNOMIALS.pdf",
 "kind": "definition", "origin": "atoms"}
```

`origin` controls merge priority: `gold` > `llm` > `atoms` > `manual`/`v1`.

### `train.jsonl` / `eval.jsonl` (final output)

```json
{"text": "<|im_start|>user\nDefine Polynomials.<|im_end|>\n<|im_start|>assistant\nPolynomials are ...<|im_end|>",
 "subject": "Math", "instruction": "Define Polynomials.", "input": "", "output": "Polynomials are ...",
 "kind": "definition", "source": "Math/NCERT/CH-2 POLYNOMIALS.pdf", "origin": "atoms"}
```

The `text` field is ChatML, which is what Qwen3 / Llama 3 / Mistral expect natively.
`source` points back at the document the row came from and is never dropped - a row
that cannot be traced to its source cannot be audited or corrected.

The CSV twin has columns `instruction,input,output,subject,source`.

## Text blocks

`scripts/pdfcols.py` rebuilds each page's text blocks before the text is
cleaned. Textbook sidebars (worked examples printed in a box beside the main
paragraph) and two-column pages are kept intact instead of being spliced into
each other. The block grouping is:

- a new block starts when a line sits more than 14pt below the previous one
  **and** starts more than 18% of the page width away from it
- a page is treated as two-column when left and right blocks overlap in
  height for more than half of the shorter column

Furniture lines (bare page numbers, "Table 3.3", "Fig. 3.2", "EXERCISE 3.1")
are dropped when they are short.

## Near-duplicate documents

A textbook and its scan carry the same content under different paths. Two consequences:

1. Generation effort is spent twice on identical sentences.
2. Worse, the document-holdout split becomes dishonest: the model sees the same material in
   both halves even though the files differ.

`dedupe.py` compares every pair of documents by Jaccard similarity over 5-gram shingles and
drops the later copy of any pair above 75% overlap. Jaccard, not containment: containment
flags any two documents that share boilerplate (a 794-document corpus produced 2,372 false
pairs that way, versus 148 true duplicates with Jaccard). Documents under 500 characters or
with fewer than 30 distinct shingles are excluded from comparison. It rewrites `corpus.json` in place and keeps
`corpus.json.pre-dedupe` as a backup. `assemble.py` runs the same check at row level as a
safety net for corpora that skipped stage 2.

## Atom classification rules

Ordered - the first match wins:

1. **definition** - `^<Capitalised phrase> (is|are|was|were|refers to|means|denotes|is defined as|is called|is known as|is the|are the) <clause>`
   Concept must be 1-7 words and must not start with a pronoun/filler.
2. **list** - `^<Heading>: <comma separated items>`
3. **formula** - contains `=`, `<=`, `>=`, `~`, or the words formula/theorem/identity
4. **compare** - contains unlike / whereas / in contrast / different from
5. **cause** - contains because / since / due to / leads to / results in
6. **process** - contains process / procedure / steps / method
7. **temporal** - contains a year or full date *and* a past-tense verb
8. **example** - contains for example / e.g. / such as
9. **rule** - contains must / cannot / never / always / is required
10. **purpose** - contains used to / in order to / helps to / ensures
11. **fact** - fallback

Tune the regexes in `scripts/atoms.py` for your domain. A medical corpus, for
example, will benefit from a stronger `definition` pattern and a dedicated
`contraindication`/`dosage` kind.

## Line unwrapping and completeness

PDF extraction emits one visual line per visual row, so a sentence is usually
split across several lines. `common.unwrap()` re-joins them before sentence
splitting. It treats short lines as headings, always starts a new unit at a
bullet or numbered item, and joins a line to the previous one only when the
previous line looks like a wrapped sentence (long enough, not ending in
punctuation).

`atoms.looks_complete()` then rejects fragments: a line that starts lowercase,
ends on a dangling conjunction, has no verb, or has a comma density typical of
a table row.

Both are configurable. Set `atoms.require_complete: false` to trade quality for
roughly 2-3x the row count, and `atoms.min_words` to 3-4 to keep short factual
sentences.

## Question archetypes

Each kind maps to a template list. Templates are picked round-robin so the
dataset does not repeat one phrasing. `per_atom` controls how many QA pairs are
produced from a single atom, and `cloze_every` adds a fill-in-the-blank item
for every Nth atom.

To add a new kind: add an entry to `TEMPLATES` in `scripts/qa.py` and a matching
branch in `build_for_atom`. Keep the answer verbatim from the source.

## LLM backend

`qa.py --llm` sends ~1000-character chunks to an OpenAI-compatible endpoint and
asks for natural exam-style questions. Constraints baked into the prompt:

- answers must be grounded only in the supplied chunk
- output must be JSON lines in ChatML format
- no mention of "source text" or "notes" in the question

The endpoint is called with 2-3 concurrent workers; a single request typically
takes 15-60s depending on the model. For 5,000 extra rows budget roughly
45-90 minutes. If the endpoint is slow, run the deterministic generator first,
start training on it, and add LLM rows in a later pass.

If the LLM returns non-JSON, `qa.py` falls back to line-by-line extraction and
skips malformed lines rather than failing the run.

## Near-duplicate questions

Exact instruction dedupe in the generator cannot catch two questions that mean the same thing
with different wording ("Define the term polynomial ... class ten" vs "... class 10").
`validate.py` compares question 4-grams across the training set and reports the number of pairs
above 80% overlap. These rows waste training signal and make the eval set look easier than it
is, so treat a high count as a signal to tighten `qa.max_per_answer` or the per-source cap.

## Leakage control

The split is the part most pipelines get wrong. This implementation:

1. holds out **whole source documents** for eval - the strongest guarantee that
   the model has not memorised the material
2. computes 5-gram shingles of every answer, weighted by inverse document frequency so
   that boilerplate shared across the corpus cannot by itself trigger the limit.
   A shingle present in every answer weighs exactly zero
3. drops any train row whose answer shares more than `max_overlap` (default
   10%) of shingles with any eval answer, measured **in both directions** so
   that a long eval answer and a short train answer cannot slip through, and
   against the single row that actually conflicts rather than the union of all
   train shingles
4. drops any remaining leaky **train** rows - a leak is always resolved by
   dropping the train row, never the eval row, because the eval set is the
   scarce resource and must keep covering every subject
5. prints the worst surviving overlap - it must be at or below the limit

Always read that number, and check that `validate.py` prints the same one. If it is
above the limit, the corpus contains near-duplicate documents (a textbook and its scan,
for example) and the fix is to exclude one copy of each duplicate pair, or to hold out
more source documents.
