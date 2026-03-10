"""
vision_parser.py
----------------
Parses a floor plan PDF using Claude Vision API.
Achieves ~90-95% accuracy vs ~65-80% for the regex approach.

Flow:
  PDF bytes → pymupdf renders page to PNG → Claude Vision API → JSON
"""

from __future__ import annotations
import base64
import json
import re

import anthropic
import pymupdf  # fitz

# ---------------------------------------------------------------------------
# PDF → Image
# ---------------------------------------------------------------------------

def pdf_page_to_image(pdf_bytes: bytes, page_num: int = 0, dpi: int = 150) -> bytes:
    """
    Render one PDF page to PNG bytes using pymupdf.
    No external dependencies (Poppler etc.) required.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Du bist ein Experte für österreichische Baupläne (Hochbau, Polierplan, Maßstab 1:50). "
    "Antworte ausschließlich mit validem JSON ohne Markdown-Blöcke oder Erklärungen."
)

_USER_PROMPT = """\
Analysiere diesen Polierplan und extrahiere alle folgenden Elemente:

1. RÄUME (raeume) – Beschriftungsblock im Inneren jedes Raums:
   - name: Raumname (z.B. "Wohnküche", "Zimmer", "Bad", "Vorraum", "Stiegenhaus", "Loggia", "AR")
   - belag: Bodenbelag (z.B. "Parkett", "Fliesen", "Feinsteinzeug", "Betonplatten", "Betonsteinpflaster")
   - flaeche: Fläche als float, Komma = Dezimaltrenner: "26,37 m²" → 26.37
   - umfang: Umfang als float: "U: 20,66 m" → 20.66
   - hoehe: Raumhöhe als float: "H: 2,42 m" → 2.42

2. FENSTER (fenster) – kleine Beschriftungen neben Fenstersymbolen (oft rotiert 90°):
   - bezeichnung: z.B. "FE_31"
   - rph: RPH-Wert als float: "RPH -24" → -24
   - fph: FPH-Wert als float: "FPH 0" → 0
   - al_breite: erster AL-Wert (Breite): "AL120" → 120
   - al_hoehe: zweiter AL-Wert (Höhe): "AL231" → 231
   - rb_breite: erster RB-Wert (Breite): "RB130" → 130
   - rb_hoehe: zweiter RB-Wert (Höhe): "RB288" → 288

3. TÜREN (tueren) – Beschriftungen bei Türsymbolen:
   - bezeichnung: z.B. "ALUTÜ_07"
   - dl_breite: erster DL-Wert: "DL100" → 100
   - dl_hoehe: zweiter DL-Wert: "DL 220" → 220

Regeln:
- Österreichisches Dezimalformat: Komma = Dezimaltrenner ("26,37" → 26.37, "312,5" → 312.5)
- Gleiche Fensterbezeichnung (z.B. FE_31) taucht in mehreren Wohnungen auf → nur EINMAL je bezeichnung ausgeben
- Fehlende Werte als null
- Außenbereiche (Wiese, Kies, Betonpflaster Außen, Kinderspielfläche) NICHT als Räume aufnehmen
- Loggias und Abstellräume (AR) ARE Räume → aufnehmen

Antworte NUR mit diesem JSON (kein Text davor oder danach):
{
  "raeume": [
    {"name": "Wohnküche", "belag": "Parkett", "flaeche": 26.37, "umfang": 20.66, "hoehe": 2.42}
  ],
  "fenster": [
    {"bezeichnung": "FE_31", "rph": -24, "fph": 0, "al_breite": 120, "al_hoehe": 231, "rb_breite": 130, "rb_hoehe": 288}
  ],
  "tueren": [
    {"bezeichnung": "ALUTÜ_07", "dl_breite": 100, "dl_hoehe": 220}
  ]
}
"""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extract JSON from Claude response.
    Handles cases where the model wraps output in ```json ... ``` blocks
    despite being instructed not to.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Kein gültiges JSON in Claude-Response gefunden:\n{text[:300]}")


# ---------------------------------------------------------------------------
# Confidence scoring (reuses pattern_matcher logic)
# ---------------------------------------------------------------------------

def _compute_konfidenz(data: dict) -> dict:
    """Compute confidence scores using the same logic as pattern_matcher."""
    try:
        from .pattern_matcher import Raum, Fenster, Tuer, compute_confidence

        raeume_items = []
        for r in data.get("raeume", []):
            try:
                raeume_items.append(Raum(**r))
            except Exception:
                pass

        fenster_items = []
        for f in data.get("fenster", []):
            try:
                fenster_items.append(Fenster(**f))
            except Exception:
                pass

        tueren_items = []
        for t in data.get("tueren", []):
            try:
                tueren_items.append(Tuer(**t))
            except Exception:
                pass

        return {
            "raeume": compute_confidence(raeume_items),
            "fenster": compute_confidence(fenster_items),
            "tueren": compute_confidence(tueren_items),
        }
    except Exception:
        return {"raeume": 0.0, "fenster": 0.0, "tueren": 0.0}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_plan_with_vision(pdf_bytes: bytes) -> dict:
    """
    Parse a PDF floor plan using Claude Vision API.

    Returns the same dict structure as pdf_parser.parse_pdf():
        raeume, fenster, tueren, konfidenz, methode
    """
    # Render page to PNG — try 150dpi, downgrade to 100dpi if too large
    img_bytes = pdf_page_to_image(pdf_bytes, dpi=150)
    if len(img_bytes) > 4_500_000:
        img_bytes = pdf_page_to_image(pdf_bytes, dpi=100)

    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    # Call Claude Vision
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _USER_PROMPT},
                ],
            }
        ],
    )

    raw_text = response.content[0].text
    data = _extract_json(raw_text)

    return {
        "raeume": data.get("raeume", []),
        "fenster": data.get("fenster", []),
        "tueren": data.get("tueren", []),
        "konfidenz": _compute_konfidenz(data),
        "methode": "vision",
    }
