"""
main.py
-------
FastAPI application with a single endpoint:
  POST /api/parse-plan  – accepts a PDF, returns structured JSON
"""

import io

from dotenv import load_dotenv

load_dotenv()  # lädt ANTHROPIC_API_KEY aus .env – muss VOR anderen Imports stehen

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from parser.pdf_parser import parse_pdf
from parser.text_grouper import group_page_text, _classify_rotation

app = FastAPI(
    title="Massenermittlung – Bauplan Parser",
    description="Extracts rooms, windows, and doors from Austrian AutoCAD/ArchiCAD PDF floor plans.",
    version="1.0.0",
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post(
    "/api/parse-plan",
    summary="Parse a PDF floor plan",
    response_description="Extracted rooms, windows, doors and confidence scores",
)
async def parse_plan(file: UploadFile = File(...)):
    """
    Upload a vector PDF floor plan (AutoCAD / ArchiCAD export).
    Returns structured JSON with:
    - **raeume**: list of rooms with name, floor material, area, perimeter, height
    - **fenster**: list of windows with designation, RPH, FPH, AL/RB dimensions
    - **tueren**: list of doors with designation and door-leaf dimensions
    - **konfidenz**: confidence scores per category (0.0 – 1.0)
    """
    # Validate file type
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Nur PDF-Dateien werden akzeptiert.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Die hochgeladene Datei ist leer.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Datei überschreitet das Limit von 50 MB.")

    try:
        result = parse_pdf(pdf_bytes)
    except ValueError as exc:
        if "no_text_layer" in str(exc):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Das PDF enthält keine extrahierbaren Textelemente. "
                    "Handelt es sich um einen eingescannten Plan (Rasterbild)?"
                ),
            )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parsing fehlgeschlagen: {exc}")

    return JSONResponse(content=result)


@app.post(
    "/api/debug-text",
    summary="Debug: zeigt raw Text-Struktur der ersten Seite",
    include_in_schema=True,
)
async def debug_text(file: UploadFile = File(...)):
    """
    Hilfreich für Diagnose: zeigt was pdfplumber aus dem Plan liest.
    Gibt Rotations-Statistiken, die ersten 80 Wörter und die ersten 15 Text-Blöcke zurück.
    """
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Leere Datei.")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[0]
            chars = page.chars

            # Rotation statistics
            rot_stats: dict[str, int] = {"horizontal": 0, "ccw90": 0, "cw90": 0, "other": 0}
            for c in chars:
                if c.get("text", "").strip():
                    rot_stats[_classify_rotation(c)] = rot_stats.get(_classify_rotation(c), 0) + 1

            # Sample raw words
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            words_sample = [
                {"text": w["text"], "x0": round(w["x0"], 1), "top": round(w["top"], 1)}
                for w in words[:80]
            ]

            # Sample text blocks
            h_blocks, r_blocks = group_page_text(page)
            h_sample = [
                {"lines": b.lines, "x0": round(b.x0, 1), "top": round(b.top, 1)}
                for b in h_blocks[:15]
            ]
            r_sample = [
                {"lines": b.lines, "x0": round(b.x0, 1), "top": round(b.top, 1)}
                for b in r_blocks[:10]
            ]

        return JSONResponse({
            "page_1": {
                "total_chars": len(chars),
                "rotation_stats": rot_stats,
                "total_words": len(words),
                "words_sample": words_sample,
                "h_blocks_count": len(h_blocks),
                "r_blocks_count": len(r_blocks),
                "h_blocks_sample": h_sample,
                "r_blocks_sample": r_sample,
            }
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}
