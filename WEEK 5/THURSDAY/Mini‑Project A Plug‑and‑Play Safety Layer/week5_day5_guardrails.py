import os, re, json, time, hashlib, yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# ---------------------------- Config ----------------------------
@dataclass
class Policy:
    require_consent: bool = True
    store_raw_inputs: bool = False
    store_raw_media: bool = False
    retain_days: int = 0                 # 0 = do not retain
    abstain_threshold: float = 0.55
    allow_sensitive_inference: bool = False
    enable_action_layer: bool = True     # allow "do something" after inference
    redaction_preview_maxlen: int = 240

@dataclass
class Runtime:
    model_name: str = "distilbert-base-uncased-sentiment"
    version: str = "w5d5"
    out_dir: str = "w5d5_safety_logs"

policy = Policy()
rt = Runtime()
os.makedirs(rt.out_dir, exist_ok=True)

# --- Optional: load policy from YAML ---
def load_policy(path="safety_policy.yaml"):
    global policy
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    policy = Policy(**d)
    print("Loaded policy:", policy)

# ---------------------------- PII Redaction ----------------------------
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b")
IDNUM = re.compile(r"\b(?:\d[ -]*?){10,}\b")
def redact_text(text: str) -> str:
    t = EMAIL.sub("[REDACTED_EMAIL]", text)
    t = PHONE.sub("[REDACTED_PHONE]", t)
    t = IDNUM.sub("[REDACTED_NUM]", t)
    return t

# ---------------------------- Consent Ledger ----------------------------
CONSENT_DB = os.path.join(rt.out_dir, "consent.jsonl")
def record_consent(user_id: str, granted: bool, purpose: str) -> None:
    rec = {"ts": time.time(), "user_hash": sha(user_id), "granted": bool(granted), "purpose": purpose}
    with open(CONSENT_DB, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec)+"\n")

def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()

# ---------------------------- Minimal Logger (Derived Only) ----------------------------
def log_event(event: Dict[str, Any]) -> None:
    # No raw inputs unless explicitly allowed
    safe = {k:v for k,v in event.items() if k not in {"raw_text","raw_audio_path","raw_video_path"}}
    path = os.path.join(rt.out_dir, f"log_{int(time.time()*1000)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)

# ---------------------------- Action Gating ----------------------------
def gate_action(pred_label: Optional[str], confidence: float) -> str:
    if not policy.enable_action_layer:
        return "DENY_ACTION"
    if confidence < policy.abstain_threshold:
        return "ABSTAIN_LOW_CONF"
    # Disallow manipulative responses when negative/high-arousal is inferred
    if pred_label in {"anger","fear","sadness"}:
        return "ALLOW_SUPPORTIVE_ONLY"
    return "ALLOW"

# ---------------------------- Wrapper ----------------------------
class EmotionSafetyWrapper:
    def __init__(self, infer_fn, label_space: List[str]):
        """
        infer_fn: callable(text: str) -> dict with keys:
                  label (str), confidence (float in [0,1]),
                  optional valence (0-1), arousal (0-1)
        """
        self.infer_fn = infer_fn
        self.labels = set(label_space)

    def run(self, user_id: str, text: str, consent_granted: bool, purpose: str="UX_improvement") -> Dict[str, Any]:
        # Consent check
        if policy.require_consent and not consent_granted:
            record_consent(user_id, False, purpose)
            out = {"status":"REFUSED_NO_CONSENT"}
            log_event({"event":"refuse_no_consent","model":rt.model_name,"version":rt.version,"purpose":purpose})
            return out

        record_consent(user_id, True, purpose)

        # PII scrub preview for logs/UI; keep raw only if allowed
        preview = redact_text(text)[:policy.redaction_preview_maxlen]
        raw_kept = text if policy.store_raw_inputs else None

        # Inference
        pred = self.infer_fn(text)

        # Validate output schema
        label = pred.get("label")
        conf = float(pred.get("confidence", 0.0))
        if label not in self.labels:
            label = "unknown"
        valence = pred.get("valence")
        arousal = pred.get("arousal")

        # Abstain policy
        abstain = conf < policy.abstain_threshold

        # Action gate
        action_policy = gate_action(label, conf)

        # Minimal, derived-only log
        log_event({
            "event":"inference",
            "model": rt.model_name, "version": rt.version,
            "user_hash": sha(user_id),
            "preview": preview,
            "label": label, "confidence": conf,
            "valence": valence, "arousal": arousal,
            "abstain": abstain,
            "action_policy": action_policy
        })

        return {
            "status": "OK",
            "label": label,
            "confidence": conf,
            "valence": valence,
            "arousal": arousal,
            "abstain": abstain,
            "action_policy": action_policy,
            "raw_echo": raw_kept  # likely None
        }

# ---------------------------- Example Integrations ----------------------------
if __name__ == "__main__":
    # Dummy classifier for demo; replace with your Day 2/3/4 models
    def dummy_infer(text: str):
        # Pretend anything with "!" is excited positive; sadness for "tired"
        if "tired" in text.lower():
            return {"label":"sadness","confidence":0.78,"valence":0.25,"arousal":0.42}
        if "!" in text:
            return {"label":"joy","confidence":0.72,"valence":0.8,"arousal":0.7}
        return {"label":"neutral","confidence":0.53,"valence":0.55,"arousal":0.4}

    wrapper = EmotionSafetyWrapper(dummy_infer, ["joy","sadness","anger","fear","surprise","neutral","unknown"])
    samples = [
        ("user_123","Please stop emailing me at john.doe@example.com", True),
        ("user_456","I’m so tired of these crashes", True),
        ("user_789","Great job team!", False)  # no consent
    ]
    for uid, txt, ok in samples:
        print(wrapper.run(uid, txt, ok))
