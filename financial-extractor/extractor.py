import fitz
import pdfplumber
import re
import json
import base64
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

# ─── Field maps for pdfplumber path (digital PDFs) ───────────────────────────
PNL_FIELD_MAP = {
    "revenue": [
        "revenue from operations", "total revenue", "sales",
        "net sales", "turnover", "operating income"
    ],
    "other_income": ["other income", "non operating income"],
    "total_income": ["total income", "total revenue including"],
    "cost_of_goods_sold": [
        "cost of goods sold", "cost of materials",
        "contractual expenses", "purchases of stock"
    ],
    "employee_expenses": [
        "employee benefit expense", "staff cost",
        "salaries and employee wages", "salaries and wages"
    ],
    "depreciation": ["depreciation and amortization", "depreciation"],
    "finance_costs": ["finance costs", "interest & charges", "interest expense"],
    "other_expenses": ["other expenses", "operating expense", "administrative"],
    "total_expenses": [
        "total expenses", "total expenditure",
        "total for operating expense"
    ],
    "profit_before_tax": [
        "profit before tax", "profit before exceptional",
        "operating profit"
    ],
    "tax_expense": ["tax expense", "current tax", "income tax"],
    "net_profit": [
        "profit for the year", "profit/(loss) for",
        "net profit/loss", "net profit", "profit after tax"
    ],
}

BS_FIELD_MAP = {
    "share_capital": ["share capital", "equity share capital"],
    "reserves_and_surplus": [
        "reserves and surplus", "other equity",
        "retained earnings", "current year earnings"
    ],
    "total_equity": ["total equity", "total shareholders", "total equities"],
    "long_term_borrowings": ["long-term borrowings", "non-current borrowings"],
    "short_term_borrowings": [
        "short-term borrowings", "current borrowings",
        "sbi od account"
    ],
    "trade_payables": ["trade payables", "accounts payable"],
    "other_current_liabilities": ["other current liabilities", "salary payable"],
    "total_current_liabilities": ["total current liabilities"],
    "total_liabilities": [
        "total equity and liabilities",
        "total liabilities & equities",
        "total liabilities and equity"
    ],
    "fixed_assets": ["property plant", "fixed assets", "tangible assets"],
    "inventories": ["inventories", "stock"],
    "trade_receivables": ["trade receivables", "accounts receivable"],
    "cash_and_equivalents": [
        "cash and cash equivalents", "cash at bank",
        "cash in hand"
    ],
    "total_current_assets": ["total current assets"],
    "total_assets": ["total assets"],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _clean_number(val):
    if not val or not isinstance(val, str):
        return None
    val = val.strip().replace("\n", "").replace(" ", "")
    negative = val.startswith("(") and val.endswith(")")
    val = val.strip("()")
    val = re.sub(r"[^\d.]", "", val)
    if not val:
        return None
    try:
        result = float(val)
        return -result if negative else result
    except ValueError:
        return None


def _detect_year_cols(header_row):
    year_cols = {}
    if not header_row:
        return year_cols
    for i, cell in enumerate(header_row):
        if not cell:
            continue
        s = str(cell).lower()
        m = re.search(r"march[,\s]+(\d{4})", s) \
            or re.search(r"f\.?y\.?\s*(\d{4})", s) \
            or re.search(r"\b(20\d{2})\b", s)
        if m:
            year_cols[i] = f"FY{m.group(1)}"
    return year_cols


def _match_field(label, field_map):
    if not label:
        return None
    lc = re.sub(r"\s+", " ", str(label).lower().strip())
    lc = re.sub(r"^\d+[.\)]\s*", "", lc)
    lc = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", lc)
    for field_key, patterns in field_map.items():
        for p in patterns:
            if p in lc or lc in p:
                return field_key
    return None


# ─── Path A: pdfplumber (digital pages) ──────────────────────────────────────
def _extract_pdfplumber(pdf_path, page_number, stmt_type):
    result = {}
    field_map = PNL_FIELD_MAP if stmt_type == "profit_and_loss" else BS_FIELD_MAP
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if not tables:
                return None
            for table in tables:
                if len(table) < 3:
                    continue
                year_cols = {}
                for row in table[:3]:
                    year_cols = _detect_year_cols(row)
                    if year_cols:
                        break
                for row in table:
                    if not row or not row[0]:
                        continue
                    field = _match_field(row[0], field_map)
                    if not field:
                        continue
                    if year_cols:
                        for col_idx, fy in year_cols.items():
                            if col_idx < len(row) and row[col_idx]:
                                v = _clean_number(str(row[col_idx]))
                                if v is not None:
                                    result.setdefault(field, {})[fy] = v
                    else:
                        nums = [
                            _clean_number(str(c)) for c in row[1:]
                            if c and _clean_number(str(c)) is not None
                        ]
                        if nums:
                            result.setdefault(field, {})
                            if len(nums) >= 2:
                                result[field]["FY_current"] = nums[0]
                                result[field]["FY_previous"] = nums[1]
                            else:
                                result[field]["FY_current"] = nums[0]
            return result if len(result) >= 3 else None
    except Exception as e:
        print(f"[pdfplumber] {e}")
        return None


# ─── Path B: Groq Vision OCR (scanned pages) ─────────────────────────────────
def _page_to_b64(pdf_path, page_number, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def _vision_ocr(image_b64, stmt_type):
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Read this scanned {stmt_type.replace('_',' ')} from an Indian "
                        "financial statement. Extract ALL text exactly as shown, preserving "
                        "every row and column. Keep numbers exactly as printed. Output raw text only."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                },
            ],
        }],
        temperature=0,
        max_tokens=2000,
    )
    return resp.choices[0].message.content


# ─── Path C: Groq LLM structuring — FULL LINE ITEM EXTRACTION ────────────────
_PNL_EXAMPLE = """{
  "revenue": {"FY2025": 3130606.05},
  "cost_of_goods_sold": {"FY2025": 2316301.20},
  "gross_profit": {"FY2025": 814304.85},
  "operating_expenses": {
    "FY2025": {
      "bank_fees_and_charges": 634.10,
      "cod_commission": 72530.69,
      "credit_card_charges": 2113.00,
      "interest_and_charges": 72585.00,
      "website_and_seo": 44314.69,
      "meals_and_entertainment": 9832.00,
      "miscellaneous": 7944.44,
      "office_supplies": 2009.00,
      "photoshoot_expenses": 11789.29,
      "commission": 21642.79,
      "printing_and_stationery": 26504.67,
      "repairs_and_maintenance": 1074.90,
      "salaries_and_wages": 317000.00,
      "travel_expense": 26285.88,
      "total": 626467.85
    }
  },
  "operating_profit": {"FY2025": 187837.00},
  "net_profit": {"FY2025": 187837.00}
}"""

_BS_EXAMPLE = """{
  "equity": {
    "FY2025": {
      "opening_balance": 150159.00,
      "current_year_earnings": 187837.00,
      "drawings": -99337.94,
      "total": 238658.06
    }
  },
  "current_liabilities": {
    "FY2025": {
      "loan_from_sangeetha": 30000.00,
      "salary_payable": 26000.00,
      "sbi_od_account": 149237.40,
      "total": 205237.40
    }
  },
  "current_assets": {
    "FY2025": {
      "inventories": 93817.00,
      "accounts_receivable": 31500.00,
      "input_igst": 161957.00,
      "input_cgst": 23802.00,
      "input_sgst": 23802.00,
      "tds_receivable": 5795.12,
      "cash_in_hand": 7355.00,
      "cash_at_bank": 95867.34,
      "total": 443895.46
    }
  },
  "total_assets": {"FY2025": 443895.46},
  "total_liabilities_and_equity": {"FY2025": 443895.46}
}"""

_CF_EXAMPLE = """{
  "operating_activities": {
    "FY2024": {
      "net_profit_before_tax": 26.19,
      "depreciation": 4.47,
      "finance_costs": 17.22,
      "trade_receivables_change": 199.89,
      "trade_payables_change": -18.63,
      "net_cash": -123.10
    }
  },
  "investing_activities": {
    "FY2024": {
      "purchase_of_fixed_assets": -16.41,
      "net_cash": 204.88
    }
  },
  "financing_activities": {
    "FY2024": {
      "long_term_borrowings": -188.86,
      "share_capital": 0.05,
      "net_cash": -93.54
    }
  },
  "net_change_in_cash": {"FY2024": -11.76},
  "closing_cash_balance": {"FY2024": 9.09}
}"""


def _llm_structure(raw_text, stmt_type):
    if stmt_type == "profit_and_loss":
        example = _PNL_EXAMPLE
    elif stmt_type == "balance_sheet":
        example = _BS_EXAMPLE
    else:
        example = _CF_EXAMPLE

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a senior financial analyst extracting data from "
                    f"Indian {stmt_type.replace('_',' ')} statements.\n\n"
                    "EXTRACTION RULES:\n"
                    "1. Extract EVERY line item — do not skip sub-items or collapse them into totals.\n"
                    "2. For expense sections (Operating Expenses, Other Expenses etc.), "
                    "create a nested object with each individual line item as a key, "
                    "plus a 'total' key for the section total.\n"
                    "3. For Balance Sheet sections (Equity, Current Liabilities, "
                    "Current Assets, Non-Current Assets, Non-Current Liabilities), "
                    "create a nested object with each sub-line item, plus 'total'.\n"
                    "4. Extract for EVERY fiscal year present. Use format FY20XX.\n"
                    "5. Values are plain floats. Brackets = negative numbers.\n"
                    "6. NEVER skip a line item. If unsure of the section, put it under 'other'.\n"
                    "7. Ignore note reference numbers (small integers beside line items).\n"
                    "8. Output ONLY valid JSON — no markdown, no explanation.\n\n"
                    f"Example output format:\n{example}"
                )
            },
            {
                "role": "user",
                "content": f"Extract ALL line items from:\n\n{raw_text[:6000]}"
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ─── Native text aggregator ───────────────────────────────────────────────────
def _native_text(pdf_path, page_list, top_n=2):
    doc = fitz.open(pdf_path)
    text = ""
    for p in page_list[:top_n]:
        text += f"\n\n--- Page {p['page_number']} ---\n"
        text += doc[p["page_number"] - 1].get_text()
    return text.strip()


# ─── Main entry point ─────────────────────────────────────────────────────────
def parse_financial_data(pdf_path, page_analysis):
    pages = page_analysis.get("pages", [])
    if not pages:
        return {"status": "error", "message": "No relevant pages found."}

    def by_type(t):
        return sorted(
            [p for p in pages if p["type"] == t],
            key=lambda x: x["score"],
            reverse=True
        )

    def extract(page_list, stmt_type):
        if not page_list:
            return None
        top = page_list[0]

        # ① Digital page → pdfplumber (zero LLM cost)
        if not top["scanned_image"]:
            result = _extract_pdfplumber(pdf_path, top["page_number"], stmt_type)
            if result:
                return {"method": "pdfplumber", "data": result}

        # ② Scanned → Groq Vision OCR
        if top["scanned_image"]:
            ocr_text = _vision_ocr(_page_to_b64(pdf_path, top["page_number"]), stmt_type)
        else:
            ocr_text = _native_text(pdf_path, page_list, top_n=2)

        # ③ Text → Groq LLM → full hierarchical JSON
        return {"method": "groq_llm", "data": _llm_structure(ocr_text, stmt_type)}

    results = {}
    if by_type("profit_and_loss"):
        results["profit_and_loss"] = extract(by_type("profit_and_loss"), "profit_and_loss")
    if by_type("balance_sheet"):
        results["balance_sheet"] = extract(by_type("balance_sheet"), "balance_sheet")
    if by_type("cash_flow"):
        results["cash_flow"] = extract(by_type("cash_flow"), "cash_flow")

    return {"status": "success", "data": results}