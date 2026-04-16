from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, shutil, os, re, fitz

from classifier      import classify_document
from page_classifier import analyze_document_pages
from extractor       import parse_financial_data

app = FastAPI(title="Financial Extractor API", version="3.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://finalversionv2.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fixes CORS headers being stripped on 500 crashes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
        headers=headers,
    )

# ── Helper ────────────────────────────────────────────────────────────────────
def _save(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


# ── 1. Full pipeline ──────────────────────────────────────────────────────────
# Returns shape that matches fsa-2.js expectations:
# result.mapped.companyname / result.mapped.allperiods / result.mapped.data
# result.raw / result.confidence
@app.post("/analyze-pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    path = _save(file)
    try:
        doc_info  = classify_document(path)
        if doc_info.get("document_type") == "unknown":
            return {"status": "error", "message": "Unknown document type."}

        page_info = analyze_document_pages(path, doc_type=doc_info["document_type"])
        extracted = parse_financial_data(path, page_info)

        # extracted must contain: mapped{}, raw, confidence{}
        # parse_financial_data should return this shape already.
        # We pass it through directly so fsa-2.js can read result.mapped etc.
        return {
            "status": "success",
            "document_classification": doc_info,
            "page_summary": {
                "total_pages":    page_info["total_pages"],
                "relevant_pages": page_info["relevant_pages_count"],
            },
            # These three keys are what fsa-2.js reads directly on `result`
            "mapped":     extracted.get("mapped", {}),
            "raw":        extracted.get("raw", ""),
            "confidence": extracted.get("confidence", {}),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── 2. Page classification only (no LLM cost — good for debugging routing) ───
@app.post("/classify-pages")
async def classify_pages(file: UploadFile = File(...)):
    path = _save(file)
    try:
        doc_info  = classify_document(path)
        page_info = analyze_document_pages(path, doc_type=doc_info["document_type"])
        return {
            "status": "success",
            "document_classification": doc_info,
            "page_analysis": page_info,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── 3. X-ray debugger (inspect raw text on any page) ─────────────────────────
@app.post("/debug-page")
async def debug_page(page_number: int, file: UploadFile = File(...)):
    path = _save(file)
    try:
        doc = fitz.open(path)
        if page_number < 1 or page_number > len(doc):
            return {"status": "error", "message": f"Document has {len(doc)} pages."}
        page     = doc[page_number - 1]
        raw_text = page.get_text()
        is_scan  = len(raw_text.strip()) < 100
        return {
            "status":       "success",
            "page":         page_number,
            "is_scanned":   is_scan,
            "char_count":   len(raw_text),
            "number_count": len(re.findall(r"\d+", raw_text)),   # BUG3 fixed: was r"\\d+"
            "text_preview": raw_text[:1500],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "version": "3.0"}
