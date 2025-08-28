import os, json, math, numpy as np, librosa, cv2, torch, torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
    audio_sr: int = 16000
    vad_out: bool = True       # also predict Valence/Arousal
    n_classes: int = 6         # joy, sadness, anger, fear, surprise, neutral (customize)
    hidden: int = 256
    dropout: float = 0.2
    min_conf: float = 0.55     # abstain threshold on fused confidence
    out_dir: str = "w5d4_outputs"

os.makedirs(CFG.out_dir, exist_ok=True)

# ---------------------------- Text Encoder ----------------------------
class TextEncoder:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(CFG.device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(CFG.device)
        out = self.model(**batch).last_hidden_state
        emb = out.mean(dim=1)  # simple mean pooling
        return nn.functional.normalize(emb, p=2, dim=-1)

# ---------------------------- Audio Encoder (Prosody Features) ----------------------------# Lightweight handcrafted features for CPU; swap for wav2vec/x-vector later.
def audio_features(wav: np.ndarray, sr: int) -> np.ndarray:
    f0 = librosa.yin(wav, fmin=50, fmax=400, sr=sr)
    e = librosa.feature.rms(y=wav).flatten()
    zc = librosa.feature.zero_crossing_rate(y=wav).flatten()
    f0 = np.nan_to_num(f0, nan=np.nanmean(f0) if not np.isnan(f0).all() else 0.0)
    stats = lambda x: np.array([np.mean(x), np.std(x), np.percentile(x, 25), np.percentile(x, 75)])
    return np.concatenate([stats(f0), stats(e), stats(zc)], axis=0)  # 12-dim

class AudioEncoder:
    def __init__(self, sr: int):
        self.sr = sr
        self.dim = 12

    def encode(self, wavs: List[np.ndarray]) -> torch.Tensor:
        feats = [audio_features(w, self.sr) for w in wavs]
        x = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=CFG.device)
        return nn.functional.normalize(x, p=2, dim=-1)

# ---------------------------- Vision Encoder (Face Landmarks) ----------------------------
def face_landmark_embed(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Simple OpenCV-based face detection as mediapipe alternative
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    # Use face bounding box features as placeholder
    x, y, w, h = faces[0]
    return np.array([x/img_bgr.shape[1], y/img_bgr.shape[0], w/img_bgr.shape[1], h/img_bgr.shape[0], w*h/(img_bgr.shape[0]*img_bgr.shape[1]), 1.0], dtype=np.float32)

class VisionEncoder:
    def __init__(self):
        # 6 dims: normalized x,y,w,h, area_ratio, confidence
        self.dim = 6

    def encode(self, images: List[np.ndarray]) -> torch.Tensor:
        out = []
        for im in images:
            v = face_landmark_embed(im)
            if v is None:
                v = np.zeros((6,), dtype=np.float32)
            out.append(v)
        x = torch.tensor(np.stack(out, axis=0), dtype=torch.float32, device=CFG.device)
        return nn.functional.normalize(x, p=2, dim=-1)

# ---------------------------- Fusion Head ----------------------------
class FusionHead(nn.Module):
    def __init__(self, d_text: int, d_audio: int, d_vision: int, n_classes: int, hidden: int, dropout: float, vad_out: bool):
        super().__init__()
        self.pt = nn.Linear(d_text, 256)
        self.pa = nn.Linear(d_audio, 256)
        self.pv = nn.Linear(d_vision, 256)
        self.fuse = nn.Sequential(nn.Linear(256*3, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, n_classes)
        self.vad_out = vad_out
        if vad_out:
            self.reg = nn.Linear(hidden, 2)

    def forward(self, zt: torch.Tensor, za: torch.Tensor, zv: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = torch.cat([self.pt(zt), self.pa(za), self.pv(zv)], dim=-1)
        h = self.fuse(z)
        logits = self.cls(h)
        if self.vad_out:
            vad = self.reg(h).sigmoid()  # in [0,1]
            return logits, vad
        return logits, None

# ---------------------------- Wrapper with Confidence & Missing Modalities ----------------------------
class MultimodalEmotion:
    def __init__(self):
        self.text = TextEncoder(CFG.text_model)
        self.audio = AudioEncoder(CFG.audio_sr)
        self.vision = VisionEncoder()
        self.model = FusionHead(self.text.dim, self.audio.dim, self.vision.dim, CFG.n_classes, CFG.hidden, CFG.dropout, CFG.vad_out).to(CFG.device)
        self.labels = ["joy","sadness","anger","fear","surprise","neutral"]

    @torch.no_grad()
    def infer(self, texts: List[str], audios: List[Optional[np.ndarray]], images: List[Optional[np.ndarray]]) -> List[Dict]:
        # Handle missing modalities with zeros + low confidence weights
        zt = self.text.encode(texts)
        za_list, zv_list, w = [], [], []
        for a, v in zip(audios, images):
            if a is None:
                za_list.append(torch.zeros((1, self.audio.dim), device=CFG.device))
                a_w = 0.2
            else:
                za_list.append(self.audio.encode([a]))
                a_w = 1.0
            if v is None:
                zv_list.append(torch.zeros((1, self.vision.dim), device=CFG.device))
                v_w = 0.2
            else:
                zv_list.append(self.vision.encode([v]))
                v_w = 1.0
            w.append((a_w, v_w))
        za = torch.cat(za_list, dim=0)
        zv = torch.cat(zv_list, dim=0)
        # simple rescaling by weights
        za = za * torch.tensor([w[i][0] for i in range(len(w))], device=CFG.device).unsqueeze(-1)
        zv = zv * torch.tensor([w[i][1] for i in range(len(w))], device=CFG.device).unsqueeze(-1)

        logits, vad = self.model(zt, za, zv)
        probs = logits.softmax(-1)
        conf, idx = probs.max(-1)
        out = []
        for i in range(len(texts)):
            entry = {
                "label": self.labels[int(idx[i])],
                "probs": (probs[i].detach().cpu() * 10000).round().int().tolist(),
                "confidence": float(conf[i]),
            }
            if vad is not None:
                entry["valence"] = float(vad[i,0])
                entry["arousal"] = float(vad[i,1])
            entry["abstain"] = entry["confidence"] < CFG.min_conf
            out.append(entry)
        return out

# ---------------------------- Demo (with synthetic audio/vision placeholders) ----------------------------
if __name__ == "__main__":
    try:
        print("Initializing Multimodal Emotion Recognition...")
        emo = MultimodalEmotion()
        print("✓ Model loaded successfully")
        
        texts = [
            "I can't believe it finally worked—this is awesome!",
            "Please stop. This is getting really frustrating.",
            "Hmm, okay, I guess it's fine."
        ]
        print(f"Processing {len(texts)} text samples...")
        
        # For demo, use noise as audio and blank images; replace with real inputs in practice.
        audios = [np.random.randn(5*CFG.audio_sr).astype(np.float32) * 0.01,  # quiet
                  np.random.randn(5*CFG.audio_sr).astype(np.float32) * 0.2,   # energetic
                  None]
        images = [None, None, None]  # camera off
        
        print("Running inference...")
        preds = emo.infer(texts, audios, images)
        
        print("✓ Inference completed!")
        print("\nResults:")
        print(json.dumps(preds, indent=2))
        
        # Save results
        with open(os.path.join(CFG.out_dir, "demo_output.json"), "w") as f:
            json.dump(preds, f, indent=2)
        print(f"\n✓ Results saved to {CFG.out_dir}/demo_output.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
