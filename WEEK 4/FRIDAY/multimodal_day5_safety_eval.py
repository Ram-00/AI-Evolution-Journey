import os
import re
import json
import regex as reg
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from PIL import Image
import pytesseract
# On Windows, set if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------# Config# ----------------------------
@dataclass
class CFG:
    image_path: str = "sample_doc.png"
    collection_name: str = "mm_safe_eval"
    llm_model: str = "llama3"
    embed_model: str = "llama3"
    device_vision: str = "auto"
    chunk_size: int = 380
    chunk_overlap: int = 40
    redact_preview_token: str = "[REDACTED]"
    unsafe_keywords: Tuple[str, ...] = (
        "self-harm", "suicide", "kill myself", "how to hack", "credit card number", "illegal", "violence"
    )

# ----------------------------# Perception# ----------------------------
def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    txt = pytesseract.image_to_string(img, lang="eng")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def caption_image(img: Image.Image, device_hint: str) -> str:
    # Handle device selection properly
    if device_hint == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_hint
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    cap = processor.decode(out[0], skip_special_tokens=True)
    return cap.strip()

# ----------------------------# Redaction# ----------------------------
PII_PATTERNS = {
    "email": reg.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", reg.I),
    "phone": reg.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b"),
    "invoice_id": reg.compile(r"\b(?:INV[- ]?\d{3,}|Invoice\s*#?\s*\w{3,})\b", reg.I),
    "cc_hint": reg.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # crude credit card pattern
}

def redact_pii(text: str, token: str = "[REDACTED]") -> Tuple[str, Dict[str, int]]:
    counts = {}
    redacted = text
    for name, patt in PII_PATTERNS.items():
        matches = patt.findall(redacted)
        counts[name] = len(matches)
        if matches:
            redacted = patt.sub(token, redacted)
    return redacted, counts

# ----------------------------# Chunking & Index# ----------------------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += max(1, size - overlap)
    return [c for c in out if c.strip()]

def build_index(chunks: List[str], embed_model: str, collection_name: str, source: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=embed_model)
    metas = [{"chunk_id": f"{source}#C{i+1}", "source": source, "type": "ocr"} for i in range(len(chunks))]
    vs = Chroma.from_texts(chunks, embeddings, metadatas=metas, collection_name=collection_name)
    return vs

# ----------------------------# Safety Filters# ----------------------------
def is_unsafe_query(q: str, unsafe_keywords: Tuple[str, ...]) -> bool:
    q_low = q.lower()
    return any(kw in q_low for kw in unsafe_keywords)

# ----------------------------# Retrieval & Answering# ----------------------------
def retrieve(vs: Chroma, query: str, k: int = 5) -> List[Dict]:
    docs = vs.similarity_search(query, k=k)
    results = []
    for d in docs:
        results.append({
            "id": d.metadata.get("chunk_id", "?"),
            "source": d.metadata.get("source", "?"),
            "type": d.metadata.get("type", "?"),
            "text": d.page_content
        })
    return results

def format_evidence(items: List[Dict], max_len=350) -> str:
    lines = []
    for x in items:
        t = x["text"][:max_len] + ("..." if len(x["text"]) > max_len else "")
        lines.append(f"{x['id']} ({x['source']}|{x['type']}): {t}")
    return "\n".join(lines)

def answer_with_citations(question: str, retrieved: List[Dict], llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model)
    block = format_evidence(retrieved)
    prompt = f"""
You are a grounded assistant. Use ONLY the retrieved evidence to answer.
Rules:
- Cite chunk IDs in parentheses, e.g., (FILE#C2).
- If evidence is insufficient, say what is missing and ask a clarifying question.

Retrieved Evidence:
{block}

Question: {question}

Answer (with citations):
"""
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out)

# ----------------------------# Evaluation: Citation & Grounding Checks# ----------------------------
SENT_SPLIT = reg.compile(r"(?<=[.!?])\s+")

def extract_citations(answer: str) -> List[str]:
    # IDs look like NAME#C<number>
    return re.findall(r"\(([^()]*?#C\d+)\)", answer)

def check_citation_validity(answer: str, retrieved_ids: List[str]) -> Dict[str, float]:
    cited = extract_citations(answer)
    if not cited:
        return {"valid_rate": 0.0, "total_citations": 0}
    valid = sum(1 for c in cited if any(cid in c for cid in retrieved_ids))
    return {"valid_rate": valid / len(cited), "total_citations": len(cited)}

def sentence_grounding_coverage(answer: str) -> Dict[str, float]:
    sents = [s for s in SENT_SPLIT.split(answer.strip()) if s.strip()]
    if not sents:
        return {"coverage": 0.0, "total_sentences": 0}
    with_cite = sum(1 for s in sents if re.search(r"\(#?\w+.*?C\d+\)", s))
    return {"coverage": with_cite / len(sents), "total_sentences": len(sents)}

# ----------------------------# Main Orchestration# ----------------------------
def main():
    cfg = CFG()
    img = load_image(cfg.image_path)
    print("[1] OCR...")
    raw_ocr = ocr_text(img)
    print("OCR chars:", len(raw_ocr))

    print(" Redaction...")
    red_ocr, pii_counts = redact_pii(raw_ocr, token=cfg.redact_preview_token)
    print("PII counts:", pii_counts)

    print(" Caption (contextual, not for facts)...")
    cap = caption_image(img, cfg.device_vision)
    print("Caption:", cap)

    print(" Chunk & Index...")
    chunks = chunk_text(red_ocr, cfg.chunk_size, cfg.chunk_overlap)
    vs = build_index(chunks, cfg.embed_model, cfg.collection_name, os.path.basename(cfg.image_path))
    print("Chunks indexed:", len(chunks))

    # Example questions (replace with your own)
    questions = [
        "Summarize key entities (names, dates, totals) present in the document.",
        "List any invoice IDs or references and their context.",
        "Are there inconsistencies between amounts mentioned? Explain."
    ]

    for q in questions:
        print("\n=== QUESTION ===")
        print(q)

        # Safety check
        if is_unsafe_query(q, cfg.unsafe_keywords):
            print("Refusal: Question flagged by safety filter.")
            continue

        print(" Retrieve evidence...")
        top = retrieve(vs, q, k=5)
        top_ids = [t["id"] for t in top]
        print("Top IDs:", ", ".join(top_ids))

        print(" Answer with citations...")
        ans = answer_with_citations(q, top, cfg.llm_model)
        print("\n--- ANSWER ---")
        print(ans)

        print("\n Evaluate grounding & citations...")
        cite_stats = check_citation_validity(ans, top_ids)
        cov_stats = sentence_grounding_coverage(ans)
        report = {
            "pii_counts": pii_counts,
            "citation_validity": cite_stats,
            "sentence_grounding": cov_stats,
            "retrieved_ids": top_ids
        }
        print(json.dumps(report, indent=2))

        # Optional: Save artifacts per question
        safe_q = re.sub(r"[^a-zA-Z0-9]+", "_", q.lower())[:40]
        with open(f"report_{safe_q}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
