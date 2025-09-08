import numpy as np, csv
from dataclasses import dataclass

@dataclass
class CFG:
    r_values = np.linspace(0.5, 2.5, 20)  # "bond lengths"
    steps: int = 120
    lr: float = 0.1
    seed: int = 3

rng = np.random.default_rng(CFG.seed)

def true_energy(r):
    # Morse-like curve surrogate
    D_e, a, r_e = 1.0, 1.5, 1.1
    return D_e * (1 - np.exp(-a*(r - r_e)))**2 - 1.0  # shifted min at ~-1

def vqe_step(theta, r):
    # pretend expectation near true energy with noise
    base = true_energy(r) + 0.05*np.tanh(np.sin(theta).mean())
    noise = rng.normal(0, 0.01)
    return base + noise

def run_scan():
    theta = rng.normal(0, 0.3, size=(16,))
    rows = [("r","energy","steps")]
    for r in CFG.r_values:
        best = 10.0
        for k in range(CFG.steps):
            e = vqe_step(theta, r)
            if e < best: best = e
            # SGD-like update toward lower e
            theta -= CFG.lr * np.tanh(e) * rng.normal(0.9, 0.2, size=theta.shape)
        rows.append((float(r), float(best), CFG.steps))
    with open("w6d6_vqe_bond_scan.csv","w",newline="") as f:
        writer = csv.writer(f); writer.writerows(rows)
    print("Saved w6d6_vqe_bond_scan.csv")

if __name__ == "__main__":
    run_scan()
