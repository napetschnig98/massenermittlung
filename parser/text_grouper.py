"""
text_grouper.py
---------------
Groups text from a pdfplumber page into TextBlocks, handling:
  - Horizontal text (room labels, door labels, window labels)
  - Rotated text 90° CCW / CW (window annotations in some AutoCAD exports)

Key design: horizontal text is clustered SPATIALLY (X+Y proximity) via Union-Find,
not globally by Y-coordinate alone. This correctly separates text in different rooms
even when they share the same Y-height on a large floor plan.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Spatial clustering for horizontal words (from page.extract_words())
MAX_X_GAP  = 22.0   # pt – max horizontal gap between words in the same block
MAX_Y_GAP  = 14.0   # pt – max vertical gap between words in the same block
LINE_TOL   = 3.0    # pt – two words are on the same line if |top diff| ≤ this

# Rotated text (char-level, for truly rotated annotations)
ROTATION_THRESHOLD = 0.3
COL_TOLERANCE_H    = 5.0    # pt – x0 tolerance for same vertical column
WORD_GAP_V         = 6.0    # pt – vertical gap that splits lines within a column
BLOCK_GAP_H        = 14.0   # pt – x0 gap that splits rotated text blocks

# Page filtering
TITLE_BLOCK_RATIO  = 0.15   # skip bottom 15% of page (Schriftfeld)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    lines: list[str]
    x0: float
    top: float
    x1: float
    bottom: float
    rotation: str  # "horizontal" | "ccw90" | "cw90"

    def __repr__(self) -> str:
        return f"TextBlock({self.rotation!r}, lines={self.lines!r})"


# ---------------------------------------------------------------------------
# Spatial clustering (Union-Find) for horizontal words
# ---------------------------------------------------------------------------

def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _cluster_words_spatially(words: list[dict]) -> list[list[dict]]:
    """
    Cluster pdfplumber words into spatial groups.
    Two words belong to the same cluster if they are within MAX_X_GAP
    horizontally AND MAX_Y_GAP vertically of each other.

    Returns list of clusters, each cluster is a list of word dicts.
    """
    if not words:
        return []

    # Sort by x0 so we can break early in inner loop
    words = sorted(words, key=lambda w: (w["x0"], w["top"]))
    n = len(words)
    parent = list(range(n))

    for i in range(n):
        w1 = words[i]
        for j in range(i + 1, n):
            w2 = words[j]
            # Since sorted by x0, once x-gap exceeds MAX_X_GAP, all further words too
            if w2["x0"] - w1["x1"] > MAX_X_GAP:
                break
            # Check vertical proximity (gap between bounding boxes, not centers)
            y_gap = max(0.0, max(w1["top"], w2["top"]) - min(w1["bottom"], w2["bottom"]))
            if y_gap <= MAX_Y_GAP:
                parent[_find(parent, i)] = _find(parent, j)

    groups: dict[int, list[dict]] = defaultdict(list)
    for i, w in enumerate(words):
        groups[_find(parent, i)].append(w)

    return list(groups.values())


def _words_to_textblock(word_group: list[dict], rotation: str = "horizontal") -> TextBlock | None:
    """Convert a cluster of words into a TextBlock with sorted lines."""
    if not word_group:
        return None

    # Sort words by (top, x0) so we can group into lines
    word_group = sorted(word_group, key=lambda w: (w["top"], w["x0"]))

    lines: list[str] = []
    current_line_words: list[dict] = []
    current_top: float | None = None

    for word in word_group:
        if current_top is None or abs(word["top"] - current_top) <= LINE_TOL:
            current_line_words.append(word)
            current_top = current_top if current_top is not None else word["top"]
        else:
            line_text = " ".join(w["text"] for w in sorted(current_line_words, key=lambda w: w["x0"]))
            if line_text.strip():
                lines.append(line_text.strip())
            current_line_words = [word]
            current_top = word["top"]

    if current_line_words:
        line_text = " ".join(w["text"] for w in sorted(current_line_words, key=lambda w: w["x0"]))
        if line_text.strip():
            lines.append(line_text.strip())

    if not lines:
        return None

    x0 = min(w["x0"] for w in word_group)
    top = min(w["top"] for w in word_group)
    x1 = max(w["x1"] for w in word_group)
    bottom = max(w["bottom"] for w in word_group)

    return TextBlock(lines=lines, x0=x0, top=top, x1=x1, bottom=bottom, rotation=rotation)


# ---------------------------------------------------------------------------
# Rotated text: char-level grouping (for truly rotated annotations)
# ---------------------------------------------------------------------------

def _classify_rotation(char: dict) -> str:
    matrix = char.get("matrix")
    if matrix is None:
        return "horizontal" if char.get("upright", True) else "ccw90"

    a, b, c, d = matrix[0], matrix[1], matrix[2], matrix[3]

    if abs(a) > (1 - ROTATION_THRESHOLD) and abs(d) > (1 - ROTATION_THRESHOLD):
        return "horizontal"

    if abs(a) < ROTATION_THRESHOLD and abs(d) < ROTATION_THRESHOLD:
        if b > ROTATION_THRESHOLD and c < -ROTATION_THRESHOLD:
            return "ccw90"
        if b < -ROTATION_THRESHOLD and c > ROTATION_THRESHOLD:
            return "cw90"

    return "horizontal" if char.get("upright", True) else "ccw90"


def _group_rotated_chars(chars: list[dict], rotation: str) -> list[TextBlock]:
    """Group truly-rotated characters into TextBlocks."""
    if not chars:
        return []

    chars = sorted(chars, key=lambda c: (c["x0"], c["top"]))

    # Group chars into columns by x0
    columns: list[list[dict]] = []
    current_col: list[dict] = []
    last_x0: float | None = None

    for char in chars:
        if last_x0 is None or abs(char["x0"] - last_x0) <= COL_TOLERANCE_H:
            current_col.append(char)
            last_x0 = char["x0"] if last_x0 is None else last_x0
        else:
            if current_col:
                columns.append(current_col)
            current_col = [char]
            last_x0 = char["x0"]
    if current_col:
        columns.append(current_col)

    # Each column → one or more line strings
    col_entries: list[tuple[str, float, float, float, float]] = []  # (text, x0, top, x1, bottom)

    for col_chars in columns:
        col_chars = sorted(col_chars, key=lambda c: c["top"])
        lines_in_col: list[list[dict]] = []
        curr: list[dict] = []

        for char in col_chars:
            if not curr:
                curr.append(char)
            else:
                gap = char["top"] - curr[-1]["bottom"]
                if gap > WORD_GAP_V:
                    lines_in_col.append(curr)
                    curr = [char]
                else:
                    curr.append(char)
        if curr:
            lines_in_col.append(curr)

        for line_chars in lines_in_col:
            text = "".join(c["text"] for c in line_chars).strip()
            if not text:
                continue
            x0 = min(c["x0"] for c in line_chars)
            x1 = max(c["x1"] for c in line_chars)
            top = min(c["top"] for c in line_chars)
            bottom = max(c["bottom"] for c in line_chars)
            col_entries.append((text, x0, top, x1, bottom))

    # Group columns into blocks by x0 gap
    col_entries = sorted(col_entries, key=lambda t: (t[1], t[2]))

    blocks: list[TextBlock] = []
    block_lines: list[str] = []
    block_meta: list[tuple] = []  # (x0, top, x1, bottom)

    for text, x0, top, x1, bottom in col_entries:
        if not block_lines:
            block_lines.append(text)
            block_meta.append((x0, top, x1, bottom))
        else:
            prev_x1 = block_meta[-1][2]
            if x0 - prev_x1 > BLOCK_GAP_H:
                block = _make_rotated_block(block_lines, block_meta, rotation)
                if block:
                    blocks.append(block)
                block_lines = [text]
                block_meta = [(x0, top, x1, bottom)]
            else:
                block_lines.append(text)
                block_meta.append((x0, top, x1, bottom))

    if block_lines:
        block = _make_rotated_block(block_lines, block_meta, rotation)
        if block:
            blocks.append(block)

    return blocks


def _make_rotated_block(lines: list[str], meta: list[tuple], rotation: str) -> TextBlock | None:
    if not lines:
        return None
    x0 = min(m[0] for m in meta)
    top = min(m[1] for m in meta)
    x1 = max(m[2] for m in meta)
    bottom = max(m[3] for m in meta)
    return TextBlock(lines=lines, x0=x0, top=top, x1=x1, bottom=bottom, rotation=rotation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def group_page_text(page) -> tuple[list[TextBlock], list[TextBlock]]:
    """
    Extract and group text from a pdfplumber Page object.

    Returns:
        (horizontal_blocks, rotated_blocks)

    horizontal_blocks: for room, door, AND window matching
                       (windows may be upright text in AutoCAD exports)
    rotated_blocks:    for window matching of truly-rotated annotations
    """
    skip_below = float(page.height) * (1 - TITLE_BLOCK_RATIO)

    # --- Horizontal words via extract_words() ---
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )

    # Filter title block zone
    words = [w for w in words if w["top"] < skip_below]

    # Spatial clustering → TextBlocks
    clusters = _cluster_words_spatially(words)
    h_blocks: list[TextBlock] = []
    for cluster in clusters:
        block = _words_to_textblock(cluster, "horizontal")
        if block:
            h_blocks.append(block)

    # --- Rotated chars (for truly-rotated annotations) ---
    chars = page.chars
    ccw90_chars = []
    cw90_chars = []

    for char in chars:
        if not char.get("text", "").strip():
            continue
        rot = _classify_rotation(char)
        if rot == "ccw90":
            ccw90_chars.append(char)
        elif rot == "cw90":
            cw90_chars.append(char)

    r_blocks = (
        _group_rotated_chars(ccw90_chars, "ccw90")
        + _group_rotated_chars(cw90_chars, "cw90")
    )

    return h_blocks, r_blocks
