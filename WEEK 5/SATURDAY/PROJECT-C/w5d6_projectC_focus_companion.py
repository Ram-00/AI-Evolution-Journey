import time, os, platform, subprocess, yaml, numpy as np
from dataclasses import dataclass

# --- Config ---
@dataclass
class Config:
    window: int = 20              # seconds between checks
    arousal_trigger: float = 0.7  # trigger threshold for arousal
    valence_floor: float = 0.4    # if below, treat as stress
    cool_down: int = 60           # seconds to keep notifications paused
    notifications_cmd_off: str = "gsettings set org.gnome.desktop.notifications show-banners false"
    notifications_cmd_on:  str = "gsettings set org.gnome.desktop.notifications show-banners true"

def run_cmd(cmd):
    try:
        subprocess.run(cmd.split(), check=True)
        return True
    except Exception:
        return False

def estimate_vad_proxy() -> tuple:
    # Placeholder: integrate Day 3/4 models here.# Simulate a sensor reading; in practice, call your text/audio VAD functions.
    v = np.random.uniform(0.3, 0.8)
    a = np.random.uniform(0.3, 0.9)
    conf = np.random.uniform(0.5, 0.95)
    return v, a, conf

def main():
    cfg = Config()
    paused_until = 0
    print("Focus Companion running. Ctrl+C to exit.")
    while True:
        v, a, conf = estimate_vad_proxy()
        now = time.time()
        stress = (a >= cfg.arousal_trigger and v <= cfg.valence_floor and conf >= 0.6)
        if stress and now >= paused_until:
            print(f"[Action] High stress detected (v={v:.2f}, a={a:.2f}, conf={conf:.2f}). Pausing notifications for {cfg.cool_down}s.")
            run_cmd(cfg.notifications_cmd_off)
            paused_until = now + cfg.cool_down
        elif now >= paused_until:
            # ensure notifications are back on
            run_cmd(cfg.notifications_cmd_on)
        time.sleep(cfg.window)

if __name__ == "__main__":
    main()
