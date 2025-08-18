import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"G:\RAM FILES\PROGRAM FILES\tesseract.exe"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

@dataclass
class Config:
    image_path: str = "sample_invoice.png"   # replace with your image
    use_cuda_if_available: bool = True
    llm_model: str = "llama3"

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def ocr_text(img: Image.Image) -> str:
    # Simple OCR; consider configuring languages with lang="eng"
    text = pytesseract.image_to_string(img, lang="eng")
    return text.strip()

def caption_image(img: Image.Image, device: str = "cpu") -> str:
    # BLIP image captioning (small, CPU-friendly; use GPU if available)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

def consolidate_with_llm(ocr: str, caption: str, question: Optional[str], model_name: str = "llama3") -> str:
    llm = OllamaLLM(model=model_name)
    template = PromptTemplate.from_template(
        "You are a multimodal assistant. Use both the OCR text and the image caption to answer.\n\n"
        "OCR Text:\n{ocr}\n\n"
        "Image Caption:\n{caption}\n\n"
        "Task: First, provide a structured summary of the document/image.\n"
        "Then, answer the user question if provided; otherwise, provide the top 3 insights.\n"
        "Be concise and avoid speculation beyond the provided content.\n\n"
        "User Question: {question}\n"
    )
    chain = template | llm
    q = question if question else "(no question)"
    response = chain.invoke({"ocr": ocr, "caption": caption, "question": q})
    return response.strip() if isinstance(response, str) else str(response)

def main():
    cfg = Config()
    device = "cuda" if (cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu"

    img = load_image(cfg.image_path)
    ocr = ocr_text(img)
    cap = caption_image(img, device=device)
    question = "What are the key details present (e.g., vendor, date, amounts) and any anomalies?"

    result = consolidate_with_llm(ocr, cap, question, model_name=cfg.llm_model)
    print("=== OCR EXTRACT ===")
    print(ocr[:800] + ("..." if len(ocr) > 800 else ""))
    print("\n=== IMAGE CAPTION ===")
    print(cap)
    print("\n=== LLM CONSOLIDATED ANSWER ===")
    print(result)

if __name__ == "__main__":
    main()
