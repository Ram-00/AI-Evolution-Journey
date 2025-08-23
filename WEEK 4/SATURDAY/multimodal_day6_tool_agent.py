import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

from PIL import Image
import pytesseract
# Set Tesseract path on Windows if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dateutil import parser as dateparser

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ===================== Config =====================
@dataclass
class CFG:
    image_path: str = "sample_doc.png"  # replace with your screenshot/receipt
    llm_model: str = "llama3"
    use_cuda_if_available: bool = True
    dry_run: bool = True  # True = no file overwrites; safe preview
    out_dir: str = "agent_outputs"

SAFE_TASKS = {"SUMMARIZE", "EXTRACT_JSON", "FLAG_ANOMALY", "APPEND_CSV", "ASK_CLARIFY"}

UNSAFE_KEYWORDS = (
    "self-harm", "suicide", "kill myself", "how to hack", "illegal", "violence",
    "credit card number", "password", "private key"
)

# ===================== Perception =====================
def load_image(path: str) -> Image.Image:
    if not os.path.exists(path): raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

def caption_image(img: Image.Image, use_cuda=True) -> str:
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip()

def ocr_text(img: Image.Image) -> str:
    txt = pytesseract.image_to_string(img, lang="eng")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ===================== Safety Filters =====================
def is_unsafe(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in UNSAFE_KEYWORDS)

def redact_preview(text: str) -> str:
    # Mask emails, phones, and long numbers for preview logs
    txt = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[REDACTED_EMAIL]", text, flags=re.I)
    txt = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b", "[REDACTED_PHONE]", txt)
    txt = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_NUM]", txt)
    return txt

# ===================== Reasoning (Task Policy) =====================
POLICY_PROMPT = PromptTemplate.from_template(
    "You are a cautious multimodal agent policy. Decide ONE task for the tools below.\n"
    "Use OCR text as factual source; use caption for context only.\n\n"
    "OCR:\n{ocr}\n\nCaption:\n{cap}\n\n"
    "Available tasks:\n"
    "- SUMMARIZE: Write a concise .md summary (title, key points, next steps).\n"
    "- EXTRACT_JSON: Output minimal JSON fields if this looks like a receipt/invoice/ticket.\n"
    "- FLAG_ANOMALY: If there are obvious inconsistencies or anomalies, flag with reasons.\n"
    "- APPEND_CSV: If this is a transaction/record, append a one-line CSV with core fields.\n"
    "- ASK_CLARIFY: If evidence is weak or ambiguous, request a clearer image or missing info.\n\n"
    "Rules:\n"
    "- Prefer EXTRACT_JSON for receipts/invoices/tickets; else SUMMARIZE.\n"
    "- Use FLAG_ANOMALY only if there is clear issue.\n"
    "- Use APPEND_CSV only for simple records with date+amount or clear key.\n"
    "- If unsure, use ASK_CLARIFY.\n"
    "Return strictly JSON with keys: task (one of the above), rationale (short), fields (object; can be empty).\n"
)

def decide_task(ocr: str, cap: str, llm_model: str) -> Dict[str, Any]:
    llm = OllamaLLM(model=llm_model)
    out = llm.invoke(POLICY_PROMPT.format(ocr=ocr, cap=cap))
    # Extract JSON robustly
    m = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not m:
        return {"task": "ASK_CLARIFY", "rationale": "No parseable policy output", "fields": {}}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"task": "ASK_CLARIFY", "rationale": "JSON parse error", "fields": {}}

# ===================== Tools =====================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_write(path: str, content: str, dry_run: bool) -> str:
    if dry_run:
        return f"[DRY-RUN] Would write: {path}\n---\n{content[:400]}{'...' if len(content)>400 else ''}"
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        path = f"{base}_new{ext}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote: {path}"

def tool_summarize(ocr: str, cap: str, out_dir: str, base_name: str, dry_run: bool) -> str:
    title = f"# Summary for {base_name}\n\n"
    body = f"Caption hint: {cap}\n\nKey points (OCR-derived):\n- " + "\n- ".join(ocr.splitlines()[:5])
    return safe_write(os.path.join(out_dir, f"{base_name}.md"), title + body, dry_run)

def tool_extract_json(ocr: str, cap: str, llm_model: str, out_dir: str, base_name: str, dry_run: bool) -> str:
    llm = OllamaLLM(model=llm_model)
    tmpl = PromptTemplate.from_template(
        "Extract minimal fields from OCR (no hallucination). If missing, use null or \"Unknown\".\n"
        "Return ONLY JSON with keys: vendor, date(YYYY-MM-DD or Unknown), amount(number or null), id(string or Unknown), notes.\n"
        "OCR:\n{ocr}\n\nCaption:\n{cap}\n"
    )
    out = llm.invoke(tmpl.format(ocr=ocr, cap=cap))
    m = re.search(r"\{.*\}", out, re.DOTALL)
    js = m.group(0) if m else '{"vendor":"Unknown","date":"Unknown","amount":null,"id":"Unknown","notes":""}'
    try:
        data = json.loads(js)
    except Exception:
        data = {"vendor":"Unknown","date":"Unknown","amount":None,"id":"Unknown","notes":""}
    content = json.dumps(data, indent=2)
    return safe_write(os.path.join(out_dir, f"{base_name}.json"), content, dry_run)

def tool_flag_anomaly(ocr: str, cap: str, out_dir: str, base_name: str, dry_run: bool) -> str:
    # Heuristic: flag if both "total" and "subtotal" appear without "tax"
    low = ocr.lower()
    issues = []
    if "total" in low and "subtotal" in low and "tax" not in low:
        issues.append("Subtotal and Total present without explicit Tax.")
    if not issues:
        issues.append("No clear anomaly rules triggered; manual review advised.")
    content = "Anomaly Report\n\n- " + "\n- ".join(issues)
    return safe_write(os.path.join(out_dir, f"{base_name}_anomaly.txt"), content, dry_run)

def tool_append_csv(ocr: str, out_dir: str, base_name: str, dry_run: bool) -> str:
    # Very simple pickers: date and amount
    date_m = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/](20\d{2}))\b", ocr)
    amount_m = re.search(r"(\$|₹|€|£)?\s?(\d{1,3}(?:[,.]\d{3})*|\d+)(?:[,.]\d{2})?", ocr)
    date_raw = date_m.group(0) if date_m else "Unknown"
    try:
        date_norm = dateparser.parse(date_raw).strftime("%Y-%m-%d") if date_raw!="Unknown" else "Unknown"
    except Exception:
        date_norm = "Unknown"
    amount = amount_m.group(0).strip() if amount_m else ""
    line = f"{base_name},{date_norm},{amount}\n"
    path = os.path.join(out_dir, "records.csv")
    if dry_run:
        return f"[DRY-RUN] Would append to {path}: {line.strip()}"
    header = "file,date,amount\n"
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header: f.write(header)
        f.write(line)
    return f"Appended line to {path}: {line.strip()}"

# ===================== Orchestrator =====================
def run_agent(cfg: CFG) -> Dict[str, Any]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    img = load_image(cfg.image_path)
    ocr_raw = ocr_text(img)
    cap = caption_image(img, cfg.use_cuda_if_available)

    # Safety gate
    if is_unsafe(ocr_raw) or is_unsafe(cap):
        return {
            "status": "refused",
            "reason": "Unsafe content detected",
            "preview": redact_preview(ocr_raw)[:400]
        }

    # Decide task
    policy = decide_task(ocr_raw, cap, cfg.llm_model)
    task = policy.get("task", "ASK_CLARIFY")
    rationale = policy.get("rationale", "")
    if task not in SAFE_TASKS:
        task = "ASK_CLARIFY"
        rationale += " | Fallback to ASK_CLARIFY (unknown task)."

    base = os.path.splitext(os.path.basename(cfg.image_path))[0]
    ensure_outdir(cfg.out_dir)

    # Execute tool
    if task == "SUMMARIZE":
        result = tool_summarize(ocr_raw, cap, cfg.out_dir, base, cfg.dry_run)
    elif task == "EXTRACT_JSON":
        result = tool_extract_json(ocr_raw, cap, cfg.llm_model, cfg.out_dir, base, cfg.dry_run)
    elif task == "FLAG_ANOMALY":
        result = tool_flag_anomaly(ocr_raw, cap, cfg.out_dir, base, cfg.dry_run)
    elif task == "APPEND_CSV":
        result = tool_append_csv(ocr_raw, cfg.out_dir, base, cfg.dry_run)
    else:
        result = "ASK_CLARIFY: Please provide a clearer image or specify the goal."

    # Decision log (redacted preview)
    log = {
        "input_image": cfg.image_path,
        "task": task,
        "rationale": rationale,
        "dry_run": cfg.dry_run,
        "caption": cap,
        "ocr_preview": redact_preview(ocr_raw)[:800],
        "tool_result": result
    }

    # Save log
    log_path = os.path.join(cfg.out_dir, f"{base}_decision_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    return {"status": "ok", "log_path": log_path, "summary": result}

if __name__ == "__main__":
    cfg = CFG()
    out = run_agent(cfg)
    print(json.dumps(out, indent=2))
