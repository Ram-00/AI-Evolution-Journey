import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

from PIL import Image
import pytesseract
# On Windows, set if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dateutil import parser as dateparser

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

@dataclass
class CFG:
    image_path: str = "sample_invoice.png"  # replace with your image
    llm_model: str = "llama3"
    use_cuda_if_available: bool = True
    currency_default: str = "USD"

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def caption_image(img: Image.Image, use_cuda=True) -> str:
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    cap = processor.decode(out[0], skip_special_tokens=True)
    return cap.strip()

def ocr_text(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img, lang="eng")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_fields_with_llm(ocr: str, caption: str, currency_default="USD", model_name="llama3") -> Dict[str, Any]:
    """
    Ask the LLM to propose a structured extraction, but force Unknown for missing fields.
    """
    llm = OllamaLLM(model=model_name)
    template = PromptTemplate.from_template(
        "You are a careful information extraction system. Use ONLY the OCR text to extract fields; "
        "you may use the image caption for context, but never invent values not present in OCR.\n\n"
        "OCR Text:\n{ocr}\n\n"
        "Image Caption:\n{caption}\n\n"
        "Return a JSON object (and nothing else) with keys:\n"
        "{{\n"
        "  \"vendor\": string or \"Unknown\",\n"
        "  \"invoice_date\": string (YYYY-MM-DD) or \"Unknown\",\n"
        "  \"invoice_id\": string or \"Unknown\",\n"
        "  \"currency\": 3-letter code (default {currency}) or \"Unknown\",\n"
        "  \"subtotal\": number or null,\n"
        "  \"tax\": number or null,\n"
        "  \"total\": number or null,\n"
        "  \"notes\": string (short) or \"\"\n"
        "}}\n"
        "Rules:\n"
        "- Prefer exact strings as they appear in OCR for vendor and invoice_id.\n"
        "- If date is present, normalize to YYYY-MM-DD; otherwise \"Unknown\".\n"
        "- If currency symbol like $, €, ₹ appears, map to USD/EUR/INR respectively; else use {currency}.\n"
        "- If any numeric field is ambiguous or missing, set to null.\n"
        "- Do not add extra fields; return compact JSON.\n"
    )
    chain = template | llm
    result = chain.invoke({"ocr": ocr, "caption": caption, "currency": currency_default})
    if isinstance(result, str):
        # Try to locate JSON (robust to minor formatting)
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
    try:
        data = json.loads(result)
    except Exception:
        # Fallback minimal structure if parsing fails
        data = {
            "vendor": "Unknown",
            "invoice_date": "Unknown",
            "invoice_id": "Unknown",
            "currency": currency_default,
            "subtotal": None,
            "tax": None,
            "total": None,
            "notes": ""
        }
    return data

def normalize_currency(raw: Optional[str], default_code="USD") -> str:
    if not raw or raw == "Unknown":
        return default_code
    raw = raw.upper().strip()
    # Map common symbols if present inside OCR-derived context
    if raw in ["USD", "EUR", "INR", "GBP", "JPY", "AUD", "CAD"]:
        return raw
    symbol_map = {"$": "USD", "€": "EUR", "₹": "INR", "£": "GBP", "¥": "JPY"}
    if raw in symbol_map:
        return symbol_map[raw]
    return default_code

def normalize_date(raw: str) -> str:
    if not raw or raw == "Unknown":
        return "Unknown"
    try:
        dt = dateparser.parse(raw, dayfirst=False, yearfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "Unknown"

def to_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        # Strip common formatting
        s = re.sub(r"[,$₹€£ ]", "", str(x))
        return float(s)
    except Exception:
        return None

def validate_invoice(data: Dict[str, Any]) -> Dict[str, Any]:
    report = {"ok": True, "errors": [], "warnings": []}
    # Required-ish fields
    if not data.get("vendor") or data["vendor"] == "Unknown":
        report["warnings"].append("Vendor is Unknown.")
    # Date normalization
    data["invoice_date"] = normalize_date(data.get("invoice_date", "Unknown"))
    if data["invoice_date"] == "Unknown":
        report["warnings"].append("Invoice date not detected or unparseable.")
    # Currency normalization
    data["currency"] = normalize_currency(data.get("currency", "Unknown"))
    # Numbers
    for k in ["subtotal", "tax", "total"]:
        data[k] = to_number(data.get(k))
        if data[k] is not None and data[k] < 0:
            report["errors"].append(f"{k} is negative.")
    # Arithmetic check (if fields present)
    if data["subtotal"] is not None and data["tax"] is not None and data["total"] is not None:
        if abs((data["subtotal"] + data["tax"]) - data["total"]) > 0.02:
            report["errors"].append("subtotal + tax does not equal total (tolerance 0.02).")
    # ID quick sanity
    inv_id = str(data.get("invoice_id", "Unknown"))
    if inv_id == "Unknown" or len(inv_id) < 3:
        report["warnings"].append("Invoice ID appears missing or too short.")
    # Outcome
    if report["errors"]:
        report["ok"] = False
    return {"data": data, "report": report}

def main():
    cfg = CFG()
    img = load_image(cfg.image_path)
    print("[1/4] Captioning...")
    cap = caption_image(img, use_cuda=cfg.use_cuda_if_available)
    print("Caption:", cap)

    print("[2/4] OCR...")
    ocr = ocr_text(img)
    print("OCR snippet:", (ocr[:200] + "...") if len(ocr) > 200 else ocr)

    print("[3/4] LLM field extraction...")
    fields = extract_fields_with_llm(ocr, cap, currency_default=cfg.currency_default, model_name=cfg.llm_model)
    print("Raw extracted fields:", json.dumps(fields, indent=2))

    print("[4/4] Validation...")
    validated = validate_invoice(fields)
    print("\n=== FINAL JSON (Normalized) ===")
    print(json.dumps(validated["data"], indent=2))
    print("\n=== VALIDATION REPORT ===")
    print(json.dumps(validated["report"], indent=2))

    # Save artifacts
    base = os.path.splitext(os.path.basename(cfg.image_path))[0]
    with open(f"{base}_extracted.json", "w", encoding="utf-8") as f:
        json.dump(validated["data"], f, indent=2)
    with open(f"{base}_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(validated["report"], f, indent=2)
    print(f"\nSaved: {base}_extracted.json and {base}_validation_report.json")

if __name__ == "__main__":
    main()
