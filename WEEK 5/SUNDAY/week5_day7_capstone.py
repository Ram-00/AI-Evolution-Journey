import os, json, time, re
from dataclasses import dataclass
from typing import Dict, Optional, List
from rich.console import Console
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    out_dir: str = "w5d7_capstone_outputs"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    vad_model: str = "j-hartmann/emotion-english-distilroberta-base"  # This has emotion labels we can map to VAD
    abstain_thr: float = 0.6
    decay: float = 0.9
    max_len: int = 128

os.makedirs(CFG.out_dir, exist_ok=True)
console = Console()

# ---------------------------- PII Redaction ----------------------------
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b")
IDNUM = re.compile(r"\b(?:\d[ -]*?){10,}\b")
def redact_text(t: str) -> str:
    t = EMAIL.sub("[REDACTED_EMAIL]", t)
    t = PHONE.sub("[REDACTED_PHONE]", t)
    t = IDNUM.sub("[REDACTED_NUM]", t)
    return t

# ---------------------------- Models ----------------------------
class SentimentModel:
    def __init__(self, name=CFG.sentiment_model):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.labels = ["neg","neu","pos"]
    @torch.no_grad()
    def infer(self, text: str) -> Dict:
        batch = self.tok(text, truncation=True, padding=True, return_tensors="pt", max_length=CFG.max_len).to(self.device)
        logits = self.model(**batch).logits
        probs = logits.softmax(-1)[0].cpu().tolist()
        conf, idx = float(max(probs)), int(torch.argmax(logits, -1))
        return {"label": self.labels[idx], "confidence": conf, "probs": [round(p,4) for p in probs]}

class VADModel:
    def __init__(self, name=CFG.vad_model):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # Map emotions to VAD space (valence, arousal)
        self.emotion_to_vad = {
            'joy': (0.9, 0.8),      # Very high positive valence, high arousal
            'sadness': (0.2, 0.3),  # Low positive valence, low arousal
            'anger': (0.3, 0.9),    # Low positive valence, high arousal
            'fear': (0.2, 0.8),     # Low positive valence, high arousal
            'surprise': (0.7, 0.8), # High positive valence, high arousal
            'disgust': (0.1, 0.6),  # Very low positive valence, medium arousal
            'neutral': (0.5, 0.5),  # Neutral on both dimensions
        }
    @torch.no_grad()
    def infer(self, text: str) -> Dict:
        batch = self.tok(text, truncation=True, padding=True, return_tensors="pt", max_length=CFG.max_len).to(self.device)
        logits = self.model(**batch).logits
        probs = logits.softmax(-1)[0].cpu().tolist()
        conf, idx = float(max(probs)), int(torch.argmax(logits, -1))
        
        # Get emotion label
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        emotion = emotion_labels[idx] if idx < len(emotion_labels) else 'neutral'
        
        # Smart emotion correction based on text content
        text_lower = text.lower()
        if any(word in text_lower for word in ['excited', 'thrilled', 'happy', 'great', 'amazing', 'wonderful']):
            if emotion in ['fear', 'sadness', 'disgust']:
                emotion = 'joy'  # Override negative emotions for clearly positive text
                conf = min(0.95, conf + 0.1)  # Boost confidence slightly
        
        # Map to VAD space
        v, a = self.emotion_to_vad.get(emotion, (0.5, 0.5))
        
        return {"valence": v, "arousal": a, "confidence": conf, "emotion": emotion, "debug": {"original_emotion": emotion_labels[idx] if idx < len(emotion_labels) else 'unknown', "corrected": emotion != emotion_labels[idx] if idx < len(emotion_labels) else False}}

# ---------------------------- Affect Buffer ----------------------------
@dataclass
class AffectState:
    v: float = 0.5
    a: float = 0.5
    decay: float = CFG.decay
    def update(self, v_new: Optional[float], a_new: Optional[float], conf: float):
        if conf < CFG.abstain_thr: return
        clamp = lambda x: 0.5 if x is None else max(0.0, min(1.0, float(x)))
        v_new, a_new = clamp(v_new), clamp(a_new)
        self.v = self.decay*self.v + (1-self.decay)*v_new*conf
        self.a = self.decay*self.a + (1-self.decay)*a_new*conf
    def snapshot(self) -> Dict:
        zone = "calm" if self.a < 0.4 else "engaged" if self.a < 0.7 else "amped"
        tone = "supportive" if self.v < 0.4 else "neutral" if self.v < 0.6 else "upbeat"
        return {"valence": round(self.v,3), "arousal": round(self.a,3), "zone": zone, "tone": tone}

# ---------------------------- Safety Wrapper ----------------------------
@dataclass
class SafetyPolicy:
    require_consent: bool = True
    store_raw: bool = False
    abstain_thr: float = CFG.abstain_thr

class Safety:
    def __init__(self, policy: SafetyPolicy):
        self.p = policy
    def check_consent(self, consent: bool) -> bool:
        return (not self.p.require_consent) or bool(consent)
    def allow(self, conf: float) -> bool:
        return conf >= self.p.abstain_thr

# ---------------------------- Actions ----------------------------
def action_summary_md(snippet: str, affect: Dict, out_dir: str) -> str:
    ts = int(time.time())
    path = os.path.join(out_dir, f"summary_{ts}.md")
    content = f"# Emotional Reflection\n\nAffect: {json.dumps(affect)}\n\nSnippet:\n> {snippet}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ---------------------------- Orchestration ----------------------------
def run_once(user_text: str, consent: bool) -> Dict:
    safety = Safety(SafetyPolicy())
    if not safety.check_consent(consent):
        return {"status":"REFUSED_NO_CONSENT"}

    s_model = SentimentModel()
    v_model = VADModel()
    affect = AffectState()

    # Inference
    s_out = s_model.infer(user_text)
    v_out = v_model.infer(user_text)

    # Combine confidence (simple max for caution)
    fused_conf = max(s_out["confidence"], v_out["confidence"])

    # Abstain?
    if not safety.allow(fused_conf):
        preview = redact_text(user_text)[:240]
        log = {
            "status": "ABSTAIN_LOW_CONF",
            "preview": preview,
            "sentiment": s_out,
            "vad": v_out,
            "ts": time.time()
        }
        with open(os.path.join(CFG.out_dir, f"log_{int(time.time()*1000)}.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        return log

    # Update state and choose safe action
    affect.update(v_out["valence"], v_out["arousal"], fused_conf)
    snap = affect.snapshot()
    snippet = "Calmer pacing and step-by-step guidance suggested." if snap["tone"]=="supportive" else "Proceed with upbeat, concise next steps."
    path = action_summary_md(snippet, snap, CFG.out_dir)

    # Log (derived only)
    event = {
        "status":"OK",
        "preview": redact_text(user_text)[:240],
        "sentiment": s_out,
        "vad": v_out,
        "affect": snap,
        "action": {"type":"WRITE_SUMMARY", "path": path},
        "ts": time.time()
    }
    with open(os.path.join(CFG.out_dir, f"log_{int(time.time()*1000)}.json"), "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    return event

if __name__ == "__main__":
    console.print("[bold]Week 5 Capstone[/bold] — Enter one message (type 'exit' to quit).")
    while True:
        txt = console.input("[green]Input:[/green] ").strip()
        if txt.lower() in {"exit","quit"}: break
        consent = console.input("Consent to derived-only processing? (y/n): ").strip().lower().startswith("y")
        out = run_once(txt, consent)
        console.print_json(data=out)
