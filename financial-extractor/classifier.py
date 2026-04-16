import fitz
import base64
import os
from groq import Groq

client      = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

FINANCIAL_KEYWORDS = [
    "balance sheet", "profit and loss", "assets", "liabilities",
    "equity", "cash flow", "statement of profit", "total income",
]
BANK_KEYWORDS = [
    "account statement", "transaction", "debit", "credit",
    "withdrawal", "deposit", "bank statement",
]
TAX_KEYWORDS = [
    "income tax return", "assessment year", "pan",
    "total income", "tax payable", "itr",
]


def _page_to_b64(page, dpi=100):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _vision_classify(pdf_path):
    """Groq Vision fallback for scanned/image-based PDFs."""
    doc = fitz.open(pdf_path)
    img = _page_to_b64(doc[0])
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "What type of document is this? "
                        "Reply with ONLY ONE of these exact words:\n"
                        "financial_statement\nbank_statement\ntax_document\nunknown"
                    )},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img}"}},
                ],
            }],
            temperature=0,
            max_tokens=10,
        )
        result = resp.choices[0].message.content.strip().lower().replace(" ", "_")
        return result if result in {"financial_statement","bank_statement","tax_document"} \
               else "financial_statement"   # safe default for this use case
    except Exception as e:
        print(f"[Vision classify error] {e}")
        return "financial_statement"


def extract_text_first_pages(pdf_path, max_pages=5):
    doc = fitz.open(pdf_path)
    return "".join(doc[i].get_text().lower() for i in range(min(len(doc), max_pages)))


def score_text(text, keywords):
    return sum(1 for kw in keywords if kw in text)


def classify_document(pdf_path):
    text   = extract_text_first_pages(pdf_path)
    scores = {
        "financial_statement": score_text(text, FINANCIAL_KEYWORDS),
        "bank_statement":      score_text(text, BANK_KEYWORDS),
        "tax_document":        score_text(text, TAX_KEYWORDS),
    }
    total     = sum(scores.values()) or 1
    max_score = max(scores.values())

    # All scores zero = scanned PDF, native text empty → use Groq Vision
    if max_score == 0:
        doc_type = _vision_classify(pdf_path)
        return {
            "document_type": doc_type,
            "confidence":    0.8,
            "scores":        scores,
            "method":        "vision",
        }

    doc_type = max(scores, key=scores.get)
    return {
        "document_type": doc_type,
        "confidence":    round(scores[doc_type] / total, 2),
        "scores":        scores,
        "method":        "text",
    }