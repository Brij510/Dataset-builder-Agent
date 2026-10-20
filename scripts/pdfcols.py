"""Block-aware PDF text extraction.

Textbooks often place worked examples in a boxed sidebar beside the main
paragraph. `page.extract_text()` reads the page in visual order and splices the
two together, producing a line that is two unrelated sentences glued into one:

    'of the pair of linear equations, i.e., the number of pants she purchased is 1'

pdfplumber's own `use_text_flow` and `layout` modes produce the same result.
The fix is to reconstruct the page's text *blocks* - groups of lines that share
a horizontal band and are close together vertically - and emit each block
whole, before moving to the next.

Two things are handled here:
  * boxed sidebars and margin examples stay in one piece
  * genuine two-column pages are read column by column
"""
import re

FURNITURE = re.compile(
    r"^\s*(\d{1,4}\s*$|Page\s+\d+|Chapter\s+\d+|Fig(ure)?\.?\s*\d+.*|"
    r"Table\s+\d+.*|EXERCISE.*|[ivx]+\.?$|\(?[a-z]\)?$)")

# A line belongs to the same block as the previous one when it sits close
# above it and starts roughly where it starts.
V_GAP = 14.0          # points
X_TOLERANCE = 0.18    # fraction of page width


def _blocks_from_lines(lines, page_width):
    """Group extracted lines into visual blocks."""
    if not lines:
        return []
    lines = sorted(lines, key=lambda l: (round(l["top"]), l["x0"]))
    blocks = []
    cur = [lines[0]]
    for prev, ln in zip(lines, lines[1:]):
        same_band = (ln["top"] - prev["bottom"]) <= V_GAP
        aligned = abs(ln["x0"] - prev["x0"]) <= page_width * X_TOLERANCE
        if same_band and aligned:
            cur.append(ln)
        else:
            blocks.append(cur)
            cur = [ln]
    blocks.append(cur)
    return blocks


def _is_two_column(blocks, page_width):
    """True when the page clearly has two side-by-side text bands."""
    if len(blocks) < 4:
        return False
    left = [b for b in blocks if b[0]["x0"] < page_width * 0.5]
    right = [b for b in blocks if b[0]["x0"] >= page_width * 0.5]
    if len(left) < 2 or len(right) < 2:
        return False
    # a real two-column page has left and right blocks at overlapping heights
    lo = [b[0]["top"] for b in left]
    ro = [b[0]["top"] for b in right]
    lo_span = max(lo) - min(lo)
    ro_span = max(ro) - min(ro)
    overlap = min(max(lo), max(ro)) - max(min(lo), min(ro))
    shortest = min(lo_span, ro_span) or 1
    return overlap / shortest > 0.5


def extract_page_text(page, drop_furniture=True):
    """Extract one page, keeping text blocks intact."""
    width = float(page.width or 612)
    try:
        lines = page.extract_text_lines(layout=False, strip=True)
    except Exception:
        lines = []
    if not lines:
        try:
            return page.extract_text() or ""
        except Exception:
            return ""

    blocks = _blocks_from_lines(lines, width)
    if _is_two_column(blocks, width):
        left = [b for b in blocks if b[0]["x0"] < width * 0.5]
        right = [b for b in blocks if b[0]["x0"] >= width * 0.5]
        ordered = sorted(left, key=lambda b: b[0]["top"]) + \
            sorted(right, key=lambda b: b[0]["top"])
    else:
        ordered = sorted(blocks, key=lambda b: (b[0]["top"], b[0]["x0"]))

    out = []
    for block in ordered:
        block_lines = [ln["text"] for ln in
                       sorted(block, key=lambda l: (round(l["top"]), l["x0"]))]
        if drop_furniture:
            block_lines = [t for t in block_lines
                           if not (FURNITURE.match(t.strip()) and len(t.strip()) < 60)]
        if block_lines:
            out.append(" ".join(block_lines))
    return "\n".join(out)
