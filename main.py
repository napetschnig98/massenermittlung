"""
main.py
-------
FastAPI application:
  GET  /                    – Frontend (static/index.html)
  POST /api/parse-plan      – PDF hochladen, parsen, in Supabase speichern
  GET  /api/results         – Letzte 20 Ergebnisse aus Supabase
  GET  /api/results/{id}    – Einzelnes Ergebnis
  PUT  /api/results/{id}    – Ergebnis bearbeiten
  GET  /health              – Railway/Vercel Healthcheck
"""

import io
import os

from dotenv import load_dotenv

load_dotenv()  # lädt .env lokal – muss VOR anderen Imports stehen

import pdfplumber
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from parser.pdf_parser import parse_pdf
from parser.text_grouper import group_page_text, _classify_rotation

app = FastAPI(
    title="Massenermittlung – Bauplan Parser",
    description="Extracts rooms, windows, and doors from Austrian AutoCAD/ArchiCAD PDF floor plans.",
    version="2.0.0",
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Serve static frontend (optional – only if aiofiles is installed)
_static_ok = False
try:
    from fastapi.staticfiles import StaticFiles
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        _static_ok = True
except Exception as _e:
    print(f"[WARN] StaticFiles not available: {_e}")


@app.get("/", include_in_schema=False)
async def root():
    if _static_ok and os.path.isfile("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Massenermittlung API läuft. <a href='/docs'>Swagger UI</a></h2>")


# ---------------------------------------------------------------------------
# Parse endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/parse-plan",
    summary="Parse a PDF floor plan",
    response_description="Extracted rooms, windows, doors and confidence scores",
)
async def parse_plan(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Nur PDF-Dateien werden akzeptiert.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Die hochgeladene Datei ist leer.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Datei überschreitet das Limit von 50 MB.")

    # Fetch Anthropic API key from Supabase (falls konfiguriert)
    api_key: str | None = None
    try:
        from supabase_client import get_anthropic_key
        api_key = get_anthropic_key()
    except Exception as key_exc:
        print(f"[INFO] Supabase key fetch failed ({key_exc}), using env var fallback")
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY nicht konfiguriert. Bitte in Railway Variables oder Supabase config-Tabelle setzen.")

    try:
        result = parse_pdf(pdf_bytes, api_key=api_key)
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

    # Ergebnis in Supabase speichern
    try:
        from supabase_client import save_result
        result_id = save_result(file.filename or "unbekannt", result)
        result["id"] = result_id
    except Exception:
        result["id"] = None  # Supabase nicht konfiguriert – kein Fehler

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Results CRUD
# ---------------------------------------------------------------------------

@app.get("/api/results", summary="Liste aller gespeicherten Ergebnisse")
async def list_results():
    try:
        from supabase_client import get_results
        return JSONResponse(content=get_results())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/results/{result_id}", summary="Einzelnes Ergebnis laden")
async def get_result(result_id: str):
    try:
        from supabase_client import get_result
        data = get_result(result_id)
        if not data:
            raise HTTPException(status_code=404, detail="Nicht gefunden.")
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/results/{result_id}", summary="Ergebnis aktualisieren (nach Bearbeitung)")
async def update_result(result_id: str, body: dict = Body(...)):
    try:
        from supabase_client import update_result as _update
        _update(result_id, body)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Debug + Health
# ---------------------------------------------------------------------------

@app.post(
    "/api/debug-text",
    summary="Debug: zeigt raw Text-Struktur der ersten Seite",
    include_in_schema=True,
)
async def debug_text(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Leere Datei.")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[0]
            chars = page.chars

            rot_stats: dict[str, int] = {"horizontal": 0, "ccw90": 0, "cw90": 0, "other": 0}
            for c in chars:
                if c.get("text", "").strip():
                    rot_stats[_classify_rotation(c)] = rot_stats.get(_classify_rotation(c), 0) + 1

            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            words_sample = [
                {"text": w["text"], "x0": round(w["x0"], 1), "top": round(w["top"], 1)}
                for w in words[:80]
            ]

            h_blocks, r_blocks = group_page_text(page)
            h_sample = [
                {"lines": b.lines, "x0": round(b.x0, 1), "top": round(b.top, 1)}
                for b in h_blocks[:15]
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
            }
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}
