from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class AffectState:
    valence: float = 0.5
    arousal: float = 0.5
    dominance: Optional[float] = None
    decay: float = 0.9
    min_conf: float = 0.55  # abstain threshold on model confidence

    def update(self, v: Optional[float], a: Optional[float], conf: float = 1.0, d: Optional[float] = None):
        # Clamp inputs
        def clamp(x): 
            return 0.5 if x is None else max(0.0, min(1.0, float(x)))
        v, a = clamp(v), clamp(a)
        # Abstain under low confidence
        if conf < self.min_conf:
            return  # keep state unchanged# Exponential decay + confidence weighting
        self.valence = self.decay*self.valence + (1-self.decay)*v*conf
        self.arousal = self.decay*self.arousal + (1-self.decay)*a*conf
        if d is not None:
            d = clamp(d)
            self.dominance = (self.dominance if self.dominance is not None else 0.5)*self.decay + (1-self.decay)*d*conf

    def snapshot(self) -> Dict:
        zone = "calm" if self.arousal < 0.4 else "engaged" if self.arousal < 0.7 else "amped"
        tone = "supportive" if self.valence < 0.4 else "neutral" if self.valence < 0.6 else "upbeat"
        return {"valence": round(self.valence,3), "arousal": round(self.arousal,3), "zone": zone, "tone": tone}

# Demo
if __name__ == "__main__":
    buf = AffectState()
    stream = [
        {"v":0.7, "a":0.6, "conf":0.8},
        {"v":0.3, "a":0.8, "conf":0.9},   # stress spike
        {"v":0.55,"a":0.45,"conf":0.6},
        {"v":0.6, "a":0.35,"conf":0.4},   # below threshold -> ignore
    ]
    for step in stream:
        buf.update(step["v"], step["a"], step["conf"])
        print(buf.snapshot())
