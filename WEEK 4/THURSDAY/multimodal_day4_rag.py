import os
import re
import glob
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Optional Windows path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

@dataclass
class CFG:
    data_dir: str = "multimodal_docs"  # folder with PNG/JPG and/or PDF files
    collection_name: str = "mm_rag_index"
    chunk_size: int = 380
    chunk_overlap: int = 40
    llm_model: str = "llama3"
    embed_model: str = "llama3"
    device_vision: str = "auto"  # "auto" -> cuda if available else cpu
    max_caption_tokens: int = 40
    # Optional: path to Poppler binaries on Windows for pdf2image
    poppler_path: Optional[str] = None

def list_media(data_dir: str) -> Tuple[List[str], List[str]]:
    exts_img = ("*.png", "*.jpg", "*.jpeg")
    images = []
    for p in exts_img:
        images.extend(glob.glob(os.path.join(data_dir, p)))
    pdfs = glob.glob(os.path.join(data_dir, "*.pdf"))
    return sorted(images), sorted(pdfs)

def pdf_to_images(pdf_path: str, poppler_path: Optional[str] = None) -> List[Image.Image]:
    # Convert each PDF page to RGB image
    pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    out = [p.convert("RGB") for p in pages]
    return out

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img, lang="eng")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def caption_image(img: Image.Image, device_hint: str, max_new_tokens=40) -> str:
    # Resolve device string: map "auto" → "cuda" if available else "cpu"
    if device_hint == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_hint
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
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

def build_index(cfg: CFG) -> Chroma:
    images, pdfs = list_media(cfg.data_dir)
    print(f"Found {len(images)} images and {len(pdfs)} PDFs.")
    embeddings = OllamaEmbeddings(model=cfg.embed_model)

    all_texts = []
    all_metas = []

    # Process images
    for img_path in images:
        img = load_image(img_path)
        text = ocr_text(img)
        cap = caption_image(img, device_hint=cfg.device_vision, max_new_tokens=cfg.max_caption_tokens)
        # Make a lightweight descriptor string to index alongside chunks
        descriptor = f"[CAPTION] {cap}"

        # Index caption/descriptors as their own small chunk
        all_texts.append(descriptor)
        all_metas.append({"source": os.path.basename(img_path), "type": "caption", "chunk_id": f"{os.path.basename(img_path)}#CAP"})

        # Index OCR chunks
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        for i, ch in enumerate(chunks, start=1):
            all_texts.append(ch)
            all_metas.append({"source": os.path.basename(img_path), "type": "ocr", "chunk_id": f"{os.path.basename(img_path)}#C{i}"})

    # Process PDFs
    for pdf_path in pdfs:
        pages = pdf_to_images(pdf_path, cfg.poppler_path)
        base = os.path.basename(pdf_path)
        for pi, page_img in enumerate(pages, start=1):
            text = ocr_text(page_img)
            cap = caption_image(page_img, device_hint=cfg.device_vision, max_new_tokens=cfg.max_caption_tokens)
            descriptor = f"[CAPTION] {cap}"
            all_texts.append(descriptor)
            all_metas.append({"source": f"{base}:page{pi}", "type": "caption", "chunk_id": f"{base}:page{pi}#CAP"})
            chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            for ci, ch in enumerate(chunks, start=1):
                all_texts.append(ch)
                all_metas.append({"source": f"{base}:page{pi}", "type": "ocr", "chunk_id": f"{base}:page{pi}#C{ci}"})

    print(f"Indexing {len(all_texts)} items...")
    vs = Chroma.from_texts(texts=all_texts, embedding=embeddings, metadatas=all_metas, collection_name=cfg.collection_name)
    return vs

def retrieve(vs: Chroma, query: str, k: int = 6) -> List[Dict]:
    docs = vs.similarity_search(query, k=k)
    out = []
    for d in docs:
        out.append({
            "id": d.metadata.get("chunk_id", "?"),
            "source": d.metadata.get("source", "?"),
            "type": d.metadata.get("type", "?"),
            "text": d.page_content
        })
    return out

def format_evidence(items: List[Dict], max_len=350) -> str:
    lines = []
    for x in items:
        t = x["text"]
        if len(t) > max_len:
            t = t[:max_len] + "..."
        lines.append(f"{x['id']} ({x['source']} | {x['type']}): {t}")
    return "\n".join(lines)

def answer_with_citations(question: str, retrieved: List[Dict], llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model)
    block = format_evidence(retrieved)
    prompt = f"""
You are a careful multimodal RAG assistant. Use ONLY the retrieved evidence below to answer. 
Rules:
- Cite chunk IDs in parentheses, e.g., (FILE.png#C2) or (report.pdf:page3#CAP).
- Prefer OCR chunks for factual claims; captions are context only.
- If evidence is insufficient, say what is missing and ask for a better image or more files.

Retrieved Evidence:
{block}

Question: {question}

Answer (with citations):
"""
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out)

def main():
    cfg = CFG()
    # Use Poppler without requiring PATH changes (set to your local install)
    cfg.poppler_path = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    # Build or load index
    vs = build_index(cfg)

    # Example questions
    questions = [
        "List vendor names and dates mentioned across the collection. Provide sources.",
        "Find the total amounts and any tax values. If multiple, show per document.",
        "Summarize the key points visible in diagrams or screenshots."
    ]
    for q in questions:
        print("\n=== QUESTION ===")
        print(q)
        top = retrieve(vs, q, k=6)
        print("\nTop evidence:")
        print(format_evidence(top))
        ans = answer_with_citations(q, top, cfg.llm_model)
        print("\n=== ANSWER ===")
        print(ans)

if __name__ == "__main__":
    main()
