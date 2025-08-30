import numpy as np, librosa, sounddevice as sd, matplotlib.pyplot as plt, time, scipy.signal as sg
from dataclasses import dataclass

SR = 16000
DUR = 8  # seconds

@dataclass
class SegmentVAD:
    valence: float
    arousal: float
    conf: float

def record_audio(seconds=DUR, sr=SR):
    print(f"Recording {seconds}s at {sr} Hz...")
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()

def prosody_feats(y, sr=SR):
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    e = librosa.feature.rms(y=y).flatten()
    zc = librosa.feature.zero_crossing_rate(y=y).flatten()
    f0 = np.nan_to_num(f0, nan=np.nanmean(f0) if not np.isnan(f0).all() else 0.0)
    return f0, e, zc

def map_to_vad(f0, e):
    # Simple heuristic: higher energy -> higher arousal; valence loosely tied to pitch relative to speaker baseline
    a_raw = (e - e.min()) / (e.max() - e.min() + 1e-6)
    f0n = (f0 - np.percentile(f0, 10)) / (np.percentile(f0, 90) - np.percentile(f0, 10) + 1e-6)
    v_raw = np.clip(0.4 + 0.2*(f0n - 0.5), 0.0, 1.0)
    # Smooth
    a = sg.savgol_filter(a_raw, max(5, len(a_raw)//15*2+1), 3, mode="nearest")
    v = sg.savgol_filter(v_raw, max(5, len(v_raw)//15*2+1), 3, mode="nearest")
    conf = np.clip(1 - np.std(a_raw - a), 0.3, 0.95)
    return v, a, conf

def main():
    y = record_audio()
    f0, e, zc = prosody_feats(y)
    v, a, conf = map_to_vad(f0, e)
    t_v = np.linspace(0, len(v)/len(y)*DUR, num=len(v))
    t_a = np.linspace(0, len(a)/len(y)*DUR, num=len(a))

    plt.figure(figsize=(8,4))
    plt.plot(t_v, v, label="Valence (est.)")
    plt.plot(t_a, a, label="Arousal (est.)")
    plt.ylim(0,1)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized")
    plt.title(f"Voice Coaching V–A Trajectory (conf≈{conf:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("w5d6_voice_vad.png")
    print("Saved plot to w5d6_voice_vad.png")

if __name__ == "__main__":
    main()
