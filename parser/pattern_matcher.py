"""
pattern_matcher.py
------------------
Extracts rooms (Räume), windows (Fenster), and doors (Türen) from TextBlocks.
Handles Austrian decimal notation (comma as separator) and both m² / m2 variants.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict

from .text_grouper import TextBlock


# ---------------------------------------------------------------------------
# Austrian number parsing
# ---------------------------------------------------------------------------

def parse_number(s: str) -> float | None:
    """
    Parse a number string in Austrian/German format:
      "26,37"      → 26.37
      "1.234,56"   → 1234.56
      "312.5"      → 312.5  (already English decimal, pass through)
      "312"        → 312.0
    Returns None on failure.
    """
    s = s.strip()
    if not s:
        return None
    try:
        if "," in s and "." in s:
            # German thousands separator: "1.234,56"
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        return float(s)
    except ValueError:
        return None


def _extract_number(text: str) -> float | None:
    """Extract first numeric token from a string and parse it."""
    m = re.search(r"[+-]?[\d.,]+", text)
    if m:
        return parse_number(m.group(0))
    return None


# ---------------------------------------------------------------------------
# Known room/floor keywords
# ---------------------------------------------------------------------------

FLOOR_KEYWORDS = re.compile(
    r"^(Parkett|Fliesen|Feinsteinzeug|Betonplatten|Betonsteinpflaster|"
    r"Teppich|Laminat|Estrich|Kies|Wiese|Wassergebundene\s+Wegedecke)$",
    re.IGNORECASE,
)

# Lines that look like window/door annotation data – not room names
_NOT_ROOM_NAME = re.compile(
    r"^(FE_\d|ALUTÜ_|RPH|FPH|AL\d|RB\d|RL\d|DL\s*\d|U:|H:|WNF|TOP\s|"
    r"Loggia|Betonplatten|Betonsteinpflaster|Kies|Wiese)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Raum:
    name: str
    belag: str | None
    flaeche: float | None
    umfang: float | None
    hoehe: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    def confidence_score(self) -> float:
        required = [self.name, self.flaeche]
        optional = [self.belag, self.umfang, self.hoehe]
        req_score = sum(1 for v in required if v is not None) / len(required)
        opt_score = sum(1 for v in optional if v is not None) / len(optional)
        return round(req_score * 0.8 + opt_score * 0.2, 4)


@dataclass
class Fenster:
    bezeichnung: str
    rph: float | None = None
    fph: float | None = None
    al_breite: float | None = None
    al_hoehe: float | None = None
    rb_breite: float | None = None
    rb_hoehe: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def confidence_score(self) -> float:
        required = [self.bezeichnung, self.al_breite, self.al_hoehe, self.rb_breite, self.rb_hoehe]
        optional = [self.rph, self.fph]
        req_score = sum(1 for v in required if v is not None) / len(required)
        opt_score = sum(1 for v in optional if v is not None) / len(optional) if optional else 1.0
        return round(req_score * 0.8 + opt_score * 0.2, 4)


@dataclass
class Tuer:
    bezeichnung: str
    dl_breite: float | None = None
    dl_hoehe: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def confidence_score(self) -> float:
        fields = [self.bezeichnung, self.dl_breite, self.dl_hoehe]
        return round(sum(1 for v in fields if v is not None) / len(fields), 4)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Room fields
_AREA_PAT     = re.compile(r"([\d.,]+)\s*m[²2]", re.IGNORECASE)
_UMFANG_PAT   = re.compile(r"U:\s*([\d.,]+)\s*m", re.IGNORECASE)
_HOEHE_PAT    = re.compile(r"H:\s*([\d.,]+)\s*m", re.IGNORECASE)

# Window fields
_WINDOW_ID    = re.compile(r"^FE_\d+$")
_RPH_PAT      = re.compile(r"RPH\s*([+-]?\d+(?:[.,]\d+)?)")
_FPH_PAT      = re.compile(r"FPH\s*([+-]?\d+(?:[.,]\d+)?)")
_AL_PAT       = re.compile(r"^AL(\d+(?:[.,]\d+)?)$")
_RB_PAT       = re.compile(r"^RB(\d+(?:[.,]\d+)?)$")

# Door fields
_DOOR_ID      = re.compile(r"ALUTÜ_\d+", re.IGNORECASE)
_RL_PAT       = re.compile(r"^RL(\d+(?:[.,]\d+)?)$")
_DL_PAT       = re.compile(r"^DL\s*(\d+(?:[.,]\d+)?)$")


# ---------------------------------------------------------------------------
# Room matching
# ---------------------------------------------------------------------------

def match_room(block: TextBlock) -> Raum | None:
    """
    Try to extract a Raum from a horizontal TextBlock.
    Returns None if the block doesn't look like a room label.
    """
    lines = [l.strip() for l in block.lines if l.strip()]
    if len(lines) < 2:
        return None

    flaeche: float | None = None
    umfang: float | None = None
    hoehe: float | None = None
    belag: str | None = None
    name: str | None = None

    for line in lines:
        # Area
        m = _AREA_PAT.search(line)
        if m and flaeche is None:
            flaeche = parse_number(m.group(1))
            continue

        # Perimeter
        m = _UMFANG_PAT.search(line)
        if m:
            umfang = parse_number(m.group(1))
            continue

        # Height
        m = _HOEHE_PAT.search(line)
        if m:
            hoehe = parse_number(m.group(1))
            continue

        # Floor material
        if FLOOR_KEYWORDS.match(line) and belag is None:
            belag = line
            continue

    # Must have area to qualify as a room
    if flaeche is None:
        return None

    # Find name: first line that isn't a keyword or numeric
    for line in lines:
        if _AREA_PAT.search(line):
            continue
        if _UMFANG_PAT.search(line):
            continue
        if _HOEHE_PAT.search(line):
            continue
        if FLOOR_KEYWORDS.match(line):
            continue
        if line == belag:
            continue
        # Skip obviously non-name lines
        if re.match(r"^[\d.,\s\+\-]+$", line):
            continue
        if _NOT_ROOM_NAME.match(line):
            continue
        name = line
        break

    if not name:
        # Fallback: use first line
        name = lines[0]

    # Reject names that are single characters (grid reference letters like "U", "V")
    # or contain no actual letters at all
    if not name or len(name.strip()) <= 1:
        return None
    if not re.search(r"[a-zA-ZäöüÄÖÜß]", name):
        return None

    return Raum(name=name, belag=belag, flaeche=flaeche, umfang=umfang, hoehe=hoehe)


# ---------------------------------------------------------------------------
# Window matching
# ---------------------------------------------------------------------------

def match_window(block: TextBlock) -> Fenster | None:
    """
    Try to extract a Fenster from a (rotated) TextBlock.
    Returns None if no window identifier found.
    """
    lines = [l.strip() for l in block.lines if l.strip()]

    bezeichnung: str | None = None
    rph: float | None = None
    fph: float | None = None
    al_values: list[float] = []
    rb_values: list[float] = []

    for line in lines:
        if _WINDOW_ID.match(line):
            bezeichnung = line
            continue

        m = _RPH_PAT.search(line)
        if m:
            rph = parse_number(m.group(1))
            continue

        m = _FPH_PAT.search(line)
        if m:
            fph = parse_number(m.group(1))
            continue

        m = _AL_PAT.match(line)
        if m:
            v = parse_number(m.group(1))
            if v is not None:
                al_values.append(v)
            continue

        m = _RB_PAT.match(line)
        if m:
            v = parse_number(m.group(1))
            if v is not None:
                rb_values.append(v)
            continue

    if not bezeichnung:
        return None

    return Fenster(
        bezeichnung=bezeichnung,
        rph=rph,
        fph=fph,
        al_breite=al_values[0] if len(al_values) >= 1 else None,
        al_hoehe=al_values[1] if len(al_values) >= 2 else None,
        rb_breite=rb_values[0] if len(rb_values) >= 1 else None,
        rb_hoehe=rb_values[1] if len(rb_values) >= 2 else None,
    )


# ---------------------------------------------------------------------------
# Door matching
# ---------------------------------------------------------------------------

def match_door(block: TextBlock) -> Tuer | None:
    """
    Try to extract a Tuer from a TextBlock.
    Returns None if no door identifier found.
    """
    lines = [l.strip() for l in block.lines if l.strip()]
    full_text = " ".join(lines)

    m = _DOOR_ID.search(full_text)
    if not m:
        return None

    bezeichnung = m.group(0)
    dl_values: list[float] = []

    for line in lines:
        m2 = _DL_PAT.match(line)
        if m2:
            v = parse_number(m2.group(1))
            if v is not None:
                dl_values.append(v)

    return Tuer(
        bezeichnung=bezeichnung,
        dl_breite=dl_values[0] if len(dl_values) >= 1 else None,
        dl_hoehe=dl_values[1] if len(dl_values) >= 2 else None,
    )


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(items: list) -> float:
    """
    Average confidence score across all items.
    Each item must implement .confidence_score() -> float.
    """
    if not items:
        return 0.0
    return round(sum(item.confidence_score() for item in items) / len(items), 2)
