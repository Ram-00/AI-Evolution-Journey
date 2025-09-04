import numpy as np
from dataclasses import dataclass

@dataclass
class CFG:
    seed: int = 1
    layers: int = 3         # data re-uploading depth
    dim: int = 8            # parameter count
    steps: int = 200
    lr: float = 0.1

rng = np.random.default_rng(CFG.seed)

def encode(x: np.ndarray) -> np.ndarray:
    # Angle encoding stand-in
    return np.concatenate([np.cos(x), np.sin(x)])

def ansatz(vec: np.ndarray, theta: np.ndarray) -> float:
    # Simple interference-like readout
    k = min(len(vec), len(theta))
    w = theta[:k]
    v = vec[:k]
    b = theta[k:].mean() if len(theta) > k else 0.0
    return float(np.tanh(np.dot(w, v) + b))

def forward(x: np.ndarray, theta: np.ndarray) -> float:
    # Data re-uploading: interleave encoding with trainable layers
    vec = encode(x)
    for _ in range(CFG.layers):
        y = ansatz(vec, theta)
        vec = encode(vec * y)  # nonlinearity + re-encoding
    return (y + 1) / 2.0  # map to [0,1] as class-1 prob

def train(X, y):
    theta = rng.normal(0, 0.3, size=(CFG.dim,))
    for t in range(1, CFG.steps+1):
        i = rng.integers(0, len(X))
        p = forward(X[i], theta)
        g = (p - y[i])  # pseudo-gradient w.r.t. scalar readout
        theta -= CFG.lr * g * rng.normal(0.8, 0.2, size=theta.shape)
        if t % 40 == 0:
            preds = (np.array([forward(x, theta) for x in X]) > 0.5).astype(int)
            acc = (preds == y).mean()
            print(f"[step {t:03d}] acc={acc:.3f}")
    return theta

if __name__ == "__main__":
    X = rng.normal(0, 1, size=(400, 6))
    y = (X.mean(axis=1) > 0.1).astype(int)
    theta = train(X, y)