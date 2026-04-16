import fitz
import re
import base64
import os
from groq import Groq

client       = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

VALID_TYPES = {"balance_sheet", "profit_and_loss", "cash_flow", "transactions"}

PAGE_PROFILES = {
    "financial_statement": {
        "balance_sheet":   ["balance sheet","equity","liabilities","assets",
                            "shareholder","current assets","non-current","total equity"],
        "profit_and_loss": ["profit and loss","revenue from operations","total revenue",
                            "expenses","profit before tax","other income","total expenses",
                            "statement of profit"],
        "cash_flow":       ["cash flow","operating activities","investing activities",
                            "financing activities","net increase in cash"],
    },
    "bank_statement": {
        "transactions": ["date","particulars","debit","credit",
                         "balance","withdrawal","deposit"],
    },
}

NOISE_MARKERS = [
    "independent auditor", "we have audited", "auditor\u2019s report",
    "to the members", "key audit matters", "basis of preparation",
]


# ─── Vision helper ────────────────────────────────────────────────────────────
def _page_to_b64(page, dpi=120):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _vision_page_type(page):
    """
    Ask Groq Vision to classify a scanned page in one word.
    Returns a valid VALID_TYPES key or None.
    """
    img = _page_to_b64(page)
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "What type of financial document page is this?\n"
                        "Reply with ONLY ONE of these exact words:\n"
                        "balance_sheet\nprofit_and_loss\ncash_flow\nnotes\nauditor_report\nother"
                    )},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img}"}},
                ],
            }],
            temperature=0,
            max_tokens=10,
        )
        result = resp.choices[0].message.content.strip().lower().replace(" ", "_")
        return result if result in VALID_TYPES else None
    except Exception as e:
        print(f"[Vision page classify] {e}")
        return None


# ─── Text helper ──────────────────────────────────────────────────────────────
def _is_data_heavy(text, min_numbers=12):
    return len(re.findall(r"\d+", text)) >= min_numbers


# ─── Main ─────────────────────────────────────────────────────────────────────
def analyze_document_pages(pdf_path, doc_type="financial_statement", max_pages=None):
    doc      = fitz.open(pdf_path)
    results  = []

    if doc_type == "unknown":
        return {"total_pages": len(doc), "relevant_pages_count": 0, "pages": []}

    profiles = PAGE_PROFILES.get(doc_type, {})
    limit    = min(len(doc), max_pages) if max_pages else len(doc)

    for page_num in range(limit):
        page       = doc[page_num]
        text       = page.get_text().lower()
        is_scanned = len(text.strip()) < 100

        if is_scanned:
            # ── Scanned page: ask Groq Vision (1 call, returns 1 word) ────────
            page_type = _vision_page_type(page)
            if page_type:
                results.append({
                    "page_number":   page_num + 1,
                    "type":          page_type,
                    "score":         5,      # vision = highest confidence
                    "scanned_image": True,
                })
        else:
            # ── Native text page: keyword scoring ─────────────────────────────
            if not _is_data_heavy(text):
                continue

            best_type  = "noise"
            best_score = 0

            for p_type, keywords in profiles.items():
                score      = sum(1 for kw in keywords if kw in text)
                noise_hits = sum(1 for nm in NOISE_MARKERS if nm in text)
                score      = max(0, score - noise_hits)
                if score > best_score:
                    best_score = score
                    best_type  = p_type

            if best_score >= 2:
                results.append({
                    "page_number":   page_num + 1,
                    "type":          best_type,
                    "score":         best_score,
                    "scanned_image": False,
                })

    return {
        "total_pages":          len(doc),
        "relevant_pages_count": len(results),
        "pages":                results,
    }
