import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    seed: int = 0
    n_samples: int = 600
    n_features: int = 6
    test_size: float = 0.25
    noise: float = 0.15  # kernel estimate noise (simulating shot noise)
    scale: float = 1.2   # feature-map nonlinearity scale

cfg = CFG()
rng = np.random.default_rng(cfg.seed)

# ---------------------------- Data ----------------------------
def make_toy():
    X1 = rng.normal(0, 1, size=(cfg.n_samples//2, cfg.n_features))
    X2 = rng.normal(0.8, 1, size=(cfg.n_samples//2, cfg.n_features))
    X = np.vstack([X1, X2]); y = np.array([0]*(cfg.n_samples//2) + [1]*(cfg.n_samples//2), dtype=int)
    return X, y

# ---------------------------- "Quantum" Kernel Stub ----------------------------
def feature_map(x: np.ndarray) -> np.ndarray:
    # Nonlinear embedding mimicking a quantum feature map UΦ(x): use cos/sin interactions
    z = np.concatenate([np.cos(cfg.scale*x), np.sin(cfg.scale*x), np.cos(cfg.scale*np.outer(x, x).flatten())[:len(x)]])
    z = z / np.linalg.norm(z)  # normalize -> like a state vector
    return z

def kernel_val(x: np.ndarray, y: np.ndarray, noise: float) -> float:
    # Fidelity-like inner product with measurement noise
    fx, fy = feature_map(x), feature_map(y)
    val = float(np.dot(fx, fy))
    return val + rng.normal(0, noise / np.sqrt(len(x) + 1))

def build_kernel_matrix(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    K = np.zeros((len(Xa), len(Xb)))
    for i in range(len(Xa)):
        for j in range(len(Xb)):
            K[i, j] = kernel_val(Xa[i], Xb[j], cfg.noise)
    return K

# ---------------------------- Train/Eval with Precomputed Kernel ----------------------------
def run():
    X, y = make_toy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    Ktr = build_kernel_matrix(Xtr, Xtr)
    svm = SVC(kernel="precomputed", C=1.0)
    svm.fit(Ktr, ytr)
    Kte = build_kernel_matrix(Xte, Xtr)
    pred = svm.predict(Kte)
    acc = accuracy_score(yte, pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(yte, pred, digits=4))
    return acc

if __name__ == "__main__":
    run()
