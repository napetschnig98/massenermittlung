"""
pdf_parser.py
-------------
Orchestrator: opens the PDF, iterates pages, calls text_grouper and
pattern_matcher, deduplicates results, and returns the final JSON-ready dict.
"""

from __future__ import annotations
import io
import re

import pdfplumber

from .text_grouper import group_page_text
from .pattern_matcher import (
    Raum, Fenster, Tuer,
    match_room, match_window, match_door,
    compute_confidence,
)

# Pattern to detect that a horizontal block contains more than one room
# (split them before matching)
_AREA_PAT = re.compile(r"[\d.,]+\s*m[²2]", re.IGNORECASE)


def _split_merged_blocks(blocks):
    """
    If a horizontal block contains more than one area pattern (e.g. two rooms
    merged by the grouper), split it at the second area match.
    """
    from .text_grouper import TextBlock

    result = []
    for block in blocks:
        area_indices = [i for i, line in enumerate(block.lines) if _AREA_PAT.search(line)]
        if len(area_indices) <= 1:
            result.append(block)
            continue

        # Split into sub-blocks: one per area match, starting from the line
        # FLOOR_KEYWORD_SEARCH_START lines before each area index.
        split_points = [0] + area_indices[1:]
        for i, start in enumerate(split_points):
            end = split_points[i + 1] if i + 1 < len(split_points) else len(block.lines)
            # Walk back from area index to find the likely name line
            # (usually 2 lines before the area line in a room block)
            if start > 0:
                look_back = max(0, area_indices[i] - 2)
                start = look_back
            sub_lines = block.lines[start:end]
            if sub_lines:
                result.append(TextBlock(
                    lines=sub_lines,
                    x0=block.x0, top=block.top,
                    x1=block.x1, bottom=block.bottom,
                    rotation=block.rotation,
                ))

    return result


def parse_pdf(pdf_bytes: bytes) -> dict:
    """
    Parse a PDF floor plan using Vision API (primary) with regex fallback.

    Returns a dict with keys: raeume, fenster, tueren, konfidenz, methode.
    """
    try:
        from .vision_parser import parse_plan_with_vision
        return parse_plan_with_vision(pdf_bytes)
    except Exception as vision_exc:
        result = _parse_pdf_regex(pdf_bytes)
        result["methode"] = "regex_fallback"
        result["vision_error"] = str(vision_exc)
        return result


def _parse_pdf_regex(pdf_bytes: bytes) -> dict:
    """
    Fallback regex-based parser using pdfplumber.
    Used when Vision API is unavailable (missing API key, network error, etc.).
    """
    total_chars = 0
    raeume: list[Raum] = []
    fenster_dict: dict[str, Fenster] = {}   # deduplicate by bezeichnung
    tueren_dict: dict[str, Tuer] = {}       # deduplicate by bezeichnung

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            total_chars += len(page.chars)

            if not page.chars:
                continue

            h_blocks, r_blocks = group_page_text(page)

            # Split any accidentally merged room blocks
            h_blocks = _split_merged_blocks(h_blocks)

            # Priority order: door > window > room
            # Windows are tried on horizontal blocks too because AutoCAD PDFs often
            # store annotation text as upright (horizontal matrix) even when visually
            # placed vertically next to a window symbol.
            for block in h_blocks:
                door = match_door(block)
                if door:
                    tueren_dict[door.bezeichnung] = door
                    continue
                window = match_window(block)
                if window:
                    fenster_dict[window.bezeichnung] = window
                    continue
                room = match_room(block)
                if room:
                    raeume.append(room)

            # Also check rotated blocks for windows and doors
            for block in r_blocks:
                window = match_window(block)
                if window:
                    fenster_dict[window.bezeichnung] = window
                else:
                    door = match_door(block)
                    if door:
                        tueren_dict[door.bezeichnung] = door

    if total_chars == 0:
        raise ValueError("no_text_layer")

    fenster_list = list(fenster_dict.values())
    tueren_list = list(tueren_dict.values())

    return {
        "raeume": [r.to_dict() for r in raeume],
        "fenster": [f.to_dict() for f in fenster_list],
        "tueren": [t.to_dict() for t in tueren_list],
        "konfidenz": {
            "raeume": compute_confidence(raeume),
            "fenster": compute_confidence(fenster_list),
            "tueren": compute_confidence(tueren_list),
        },
        "methode": "regex",
    }
