import os
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from PIL import Image
import pytesseract

# If needed on Windows:# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

@dataclass
class CFG:
    image_path: str = "sample_doc.png"   # replace with your image
    model_llm: str = "llama3"
    model_embed: str = "llama3"          # OllamaEmbeddings wrapper will call Ollama embedding endpoint
    device_vision: str = "auto"          # "auto" -> cuda if available, else cpu
    chunk_size: int = 400
    chunk_overlap: int = 40
    collection_name: str = "doc_chunks"

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img, lang="eng")
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def caption_image(img: Image.Image, device_hint: str) -> str:
    device = "cuda" if (device_hint == "auto" and torch.cuda.is_available()) else device_hint
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    cap = processor.decode(out[0], skip_special_tokens=True)
    return cap.strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return [c.strip() for c in chunks if c.strip()]

def build_vector_index(chunks: List[str], embed_model: str, collection_name: str):
    embeddings = OllamaEmbeddings(model=embed_model)
    # Store chunk IDs like C1, C2,...
    metadatas = [{"chunk_id": f"C{i+1}"} for i in range(len(chunks))]
    vs = Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas, collection_name=collection_name)
    return vs

def retrieve(vs: Chroma, query: str, k: int = 4) -> List[Dict]:
    docs = vs.similarity_search(query, k=k)
    # Each doc has page_content and metadata with chunk_id
    out = []
    for d in docs:
        out.append({"id": d.metadata.get("chunk_id", "?"), "text": d.page_content})
    return out

def format_citations(chunks: List[Dict]) -> str:
    # Provide a compact, numbered citation block for the LLM
    lines = []
    for c in chunks:
        short = c["text"]
        if len(short) > 350:
            short = short[:350] + "..."
        lines.append(f"{c['id']}: {short}")
    return "\n".join(lines)

def answer_with_citations(question: str, caption: str, retrieved: List[Dict], llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model)
    citations_block = format_citations(retrieved)
    prompt = f"""
You are a careful multimodal assistant. Use the retrieved OCR chunks as primary evidence and the image caption as context. 
Rules:
- Prefer evidence from the chunks over the caption if they disagree.
- Answer concisely and include inline citations to chunk IDs in parentheses, e.g., (C2).
- If evidence is insufficient, ask a clarifying question instead of guessing.

Image Caption:
{caption}

Retrieved Evidence Chunks:
{citations_block}

Question: {question}

Answer (with citations):
"""
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out)

def main():
    cfg = CFG()
    img = load_image(cfg.image_path)

    print("[Step 1] Running OCR...")
    text = ocr_text(img)
    print(f"OCR length: {len(text)} chars")

    print("[Step 2] Captioning image...")
    caption = caption_image(img, device_hint=cfg.device_vision)
    print(f"Caption: {caption}")

    print("[Step 3] Chunking OCR and indexing...")
    chunks = chunk_text(text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print(f"Chunks: {len(chunks)}")
    vs = build_vector_index(chunks, cfg.model_embed, cfg.collection_name)

    print("[Step 4] Ask question and retrieve evidence...")
    question = "Extract key entities (names, dates, totals) and summarize the document. Flag any inconsistencies."
    top = retrieve(vs, question, k=4)

    print("[Step 5] Compose grounded answer with citations...")
    answer = answer_with_citations(question, caption, top, cfg.model_llm)

    print("\n=== GROUNDED ANSWER ===")
    print(answer)

    print("\n=== EVIDENCE USED (IDs and excerpts) ===")
    for t in top:
        preview = t["text"][:250] + ("..." if len(t["text"]) > 250 else "")
        print(f"{t['id']}: {preview}")

if __name__ == "__main__":
    main()
