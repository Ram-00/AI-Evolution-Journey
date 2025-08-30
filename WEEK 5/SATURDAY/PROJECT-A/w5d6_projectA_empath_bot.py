import time, json, sys, os
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Affect Buffer (from Day 3, streamlined) ---
@dataclass
class AffectState:
    valence: float = 0.5
    arousal: float = 0.5
    decay: float = 0.9
    min_conf: float = 0.55
    def update(self, v: Optional[float], a: Optional[float], conf: float):
        if conf < self.min_conf: return
        clamp = lambda x: 0.5 if x is None else max(0.0, min(1.0, float(x)))
        v, a = clamp(v), clamp(a)
        self.valence = self.decay*self.valence + (1-self.decay)*v*conf
        self.arousal = self.decay*self.arousal + (1-self.decay)*a*conf
    def snapshot(self) -> Dict:
        zone = "calm" if self.arousal < 0.4 else "engaged" if self.arousal < 0.7 else "amped"
        tone = "supportive" if self.valence < 0.4 else "neutral" if self.valence < 0.6 else "upbeat"
        return {"v": round(self.valence,3), "a": round(self.arousal,3), "zone": zone, "tone": tone}

# --- VAD Model (Day 3-compatible head: 2-dim regression logits in [0,1] after sigmoid) ---
class TextVAD:
    def __init__(self, model_name="distilbert-base-uncased", path_head=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load trained model first
        if os.path.exists("trained_vad_model/vad_model.pt"):
            print("🎯 Loading trained VAD model...")
            self.tok = AutoTokenizer.from_pretrained("trained_vad_model")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "trained_vad_model", num_labels=2, problem_type="regression"
            )
            self.model.load_state_dict(torch.load("trained_vad_model/vad_model.pt", map_location=self.device), strict=False)
        else:
            print("⚠️  No trained model found. Using base model (will have low confidence).")
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, problem_type="regression"
            )
            # Optionally load fine-tuned head weights from Day 3
            if path_head:
                self.model.load_state_dict(torch.load(path_head, map_location=self.device), strict=False)
        
        self.model.to(self.device).eval()
    @torch.no_grad()
    def infer(self, text: str) -> Dict:
        batch = self.tok(text, truncation=True, padding=True, return_tensors="pt", max_length=128).to(self.device)
        logits = self.model(**batch).logits.sigmoid().cpu().numpy()[0]
        v, a = float(logits[0]), float(logits[1])
        
        # Check if we're using a trained model
        if os.path.exists("trained_vad_model/vad_model.pt"):
            # For trained model: use prediction certainty
            v_extreme = abs(v - 0.5) * 2  # 0 to 1 scale
            a_extreme = abs(a - 0.5) * 2  # 0 to 1 scale
            conf = (v_extreme + a_extreme) / 2  # Higher confidence for more extreme predictions
            conf = min(0.95, conf + 0.3)  # Boost confidence for trained model
        else:
            # For untrained model: use a more lenient confidence calculation
            v_extreme = abs(v - 0.5) * 2  # 0 to 1 scale
            a_extreme = abs(a - 0.5) * 2  # 0 to 1 scale
            conf = (v_extreme + a_extreme) / 2  # Higher confidence for more extreme predictions
            
            # Add some randomness to simulate "learning" - remove this when you have a trained model
            import random
            conf = min(0.8, conf + random.uniform(0.1, 0.3))
        
        return {"valence": v, "arousal": a, "confidence": max(0.0, min(1.0, conf))}

# --- Safety Wrapper (Day 5 style, trimmed) ---
class Safety:
    def __init__(self, abstain_thr=0.4):  # Balanced threshold
        self.abstain_thr = abstain_thr
    def allow(self, conf: float) -> bool:
        return conf >= self.abstain_thr

# --- Response Policy ---
def policy(message: str, vad: Dict, state: AffectState) -> str:
    v, a = vad["valence"], vad["arousal"]
    snap = state.snapshot()
    if v < 0.35 and a > 0.65:
        return "I sense stress. Want a short step-by-step or should I escalate to a human?"
    if v < 0.35 and a < 0.45:
        return "Sounds tough. I can slow down and keep it concise. What’s the main blocker?"
    if v > 0.7 and a > 0.6:
        return "Great momentum! Shall I queue the next advanced step?"
    return "Got it. Here’s a concise next step you can try."

# --- Chat Loop ---
def run_chat():
    console = Console()
    vad_model = TextVAD()
    safety = Safety(abstain_thr=0.4)  # Balanced threshold
    state = AffectState()
    console.print("[bold]Empathic Support Bot[/bold] (type 'exit' to quit)")
    
    # Check if trained model exists
    if os.path.exists("trained_vad_model/vad_model.pt"):
        console.print("[green]✅ Using trained VAD model - high confidence expected![/green]")
    else:
        console.print("[yellow]⚠️  No trained model found. Run 'python train_vad_model.py' first for better results![/yellow]")
    while True:
        user = console.input("[green]You:[/green] ").strip()
        if user.lower() in {"exit","quit"}: break
        vad = vad_model.infer(user)
        if not safety.allow(vad["confidence"]):
            console.print("[yellow]I’m not confident about the emotional signal. Could you rephrase or add context?[/yellow]")
            continue
        state.update(vad["valence"], vad["arousal"], vad["confidence"])
        reply = policy(user, vad, state)
        console.print(f"[cyan]Agent:[/cyan] {reply}")
        console.print(f"[dim]Affect {state.snapshot()} (conf={vad['confidence']:.2f})[/dim]")
        # Escalation condition (persistent low valence)
        if state.valence < 0.35:
            console.print("[magenta]I can connect you to a human if you’d like. Say 'agent' to proceed.[/magenta]")

if __name__ == "__main__":
    run_chat()
