import os
import re
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from PIL import Image
import pytesseract
# On Windows, set if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dateutil import parser as dateparser

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------# Config# ----------------------------
@dataclass
class CFG:
    image_path: str = "sample_invoice.png"  # replace with your file
    out_dir: str = "capstone_outputs"
    collection_name: str = "mm_capstone"
    llm_model: str = "llama3"
    embed_model: str = "llama3"
    device_vision: str = "auto"
    chunk_size: int = 380
    chunk_overlap: int = 40
    dry_run: bool = True  # safe by default

UNSAFE_KEYWORDS = (
    "self-harm","suicide","kill myself","how to hack","illegal","violence",
    "credit card number","password","private key"
)

# ----------------------------# Perception# ----------------------------
def load_image(path: str) -> Image.Image:
    if not os.path.exists(path): raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    txt = pytesseract.image_to_string(img, lang="eng")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def caption_image(img: Image.Image, device_hint: str) -> str:
    # Resolve device to a valid PyTorch device
    if device_hint == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_hint
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip()

# ----------------------------# Redaction# ----------------------------
def redact_preview(text: str) -> str:
    t = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[REDACTED_EMAIL]", text, flags=re.I)
    t = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b", "[REDACTED_PHONE]", t)
    t = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_NUM]", t)
    return t

def is_unsafe_text(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in UNSAFE_KEYWORDS)

# ----------------------------# Chunking & Index# ----------------------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += max(1, size - overlap)
    return [c for c in out if c.strip()]

def build_index(chunks: List[str], embed_model: str, collection: str, source: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=embed_model)
    metas = [{"chunk_id": f"{source}#C{i+1}", "source": source, "type": "ocr"} for i in range(len(chunks))]
    vs = Chroma.from_texts(chunks, embeddings, metadatas=metas, collection_name=collection)
    return vs

def retrieve(vs: Chroma, query: str, k: int = 5) -> List[Dict]:
    docs = vs.similarity_search(query, k=k)
    return [{"id": d.metadata.get("chunk_id","?"), "source": d.metadata.get("source","?"), "text": d.page_content} for d in docs]

def format_evidence(items: List[Dict], max_len=350) -> str:
    lines=[]
    for x in items:
        t = x["text"][:max_len] + ("..." if len(x["text"])>max_len else "")
        lines.append(f"{x['id']} ({x['source']}): {t}")
    return "\n".join(lines)

# ----------------------------# Grounded QA with Citations# ----------------------------
def answer_with_citations(question: str, retrieved: List[Dict], llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model)
    block = format_evidence(retrieved)
    prompt = f"""
Use ONLY the retrieved OCR evidence to answer. Cite chunk IDs in parentheses, e.g., (file.png#C2).
If evidence is insufficient, ask a clarifying question.

Retrieved Evidence:
{block}

Question: {question}

Answer (with citations):
"""
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out)

# ----------------------------# Safe Actions (Do)# ----------------------------
def safe_write(path: str, content: str, dry_run: bool) -> str:
    if dry_run:
        return f"[DRY-RUN] Would write: {path}\n---\n{content[:400]}{'...' if len(content)>400 else ''}"
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        path = f"{base}_new{ext}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote: {path}"

def action_summary_md(base: str, ocr: str, caption: str, out_dir: str, dry_run: bool) -> str:
    title = f"# Summary: {base}\n\n"
    body = f"Caption: {caption}\n\nKey OCR lines:\n- " + "\n- ".join([ln for ln in ocr.splitlines() if ln.strip()][:5])
    return safe_write(os.path.join(out_dir, f"{base}.md"), title+body, dry_run)

def action_extract_json(base: str, ocr: str, caption: str, llm_model: str, out_dir: str, dry_run: bool) -> str:
    llm = OllamaLLM(model=llm_model)
    tmpl = (
      "Extract minimal fields as JSON only: vendor, date(YYYY-MM-DD or Unknown), amount(number or null), id(string or Unknown), notes.\n"
      "Use OCR as ground truth; do not invent values. Caption is context only.\n"
      f"OCR:\n{{ocr}}\n\nCaption:\n{{caption}}\n"
    )
    out = llm.invoke(tmpl.format(ocr=ocr, caption=caption))
    m = re.search(r"\{.*\}", out, re.DOTALL)
    js = m.group(0) if m else '{"vendor":"Unknown","date":"Unknown","amount":null,"id":"Unknown","notes":""}'
    try:
        data = json.loads(js)
    except Exception:
        data = {"vendor":"Unknown","date":"Unknown","amount":None,"id":"Unknown","notes":""}
    return safe_write(os.path.join(out_dir, f"{base}.json"), json.dumps(data, indent=2), dry_run)

# ----------------------------# Orchestration# ----------------------------
def main():
    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(cfg.image_path))[0]

    # Perception
    img = load_image(cfg.image_path)
    ocr = ocr_text(img)
    caption = caption_image(img, cfg.device_vision)

    # Safety gate
    if is_unsafe_text(ocr) or is_unsafe_text(caption):
        log = {"status":"refused","reason":"unsafe content","ocr_preview":redact_preview(ocr)[:600]}
        with open(os.path.join(cfg.out_dir, f"{base}_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(json.dumps(log, indent=2))
        return

    # Index & retrieve
    chunks = chunk_text(ocr, cfg.chunk_size, cfg.chunk_overlap)
    vs = build_index(chunks, cfg.embed_model, cfg.collection_name, base)
    question = "Summarize key entities (names, dates, totals) and cite the exact chunks."
    retrieved = retrieve(vs, question, k=5)
    answer = answer_with_citations(question, retrieved, cfg.llm_model)

    # Choose action: if it looks like a receipt/invoice, export JSON, else write a summary
    looks_structured = any(w in ocr.lower() for w in ["invoice","receipt","total","amount","bill"])
    if looks_structured:
        action_result = action_extract_json(base, ocr, caption, cfg.llm_model, cfg.out_dir, cfg.dry_run)
        action_name = "EXTRACT_JSON"
    else:
        action_result = action_summary_md(base, ocr, caption, cfg.out_dir, cfg.dry_run)
        action_name = "SUMMARIZE"

    # Log everything (with redacted previews)
    log = {
        "status": "ok",
        "image": cfg.image_path,
        "question": question,
        "retrieved_ids": [r["id"] for r in retrieved],
        "answer": answer,
        "action": action_name,
        "action_result": action_result,
        "ocr_preview": redact_preview(ocr)[:800],
        "caption": caption,
        "dry_run": cfg.dry_run
    }
    with open(os.path.join(cfg.out_dir, f"{base}_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\n=== CAPSTONE OUTPUT ===")
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
