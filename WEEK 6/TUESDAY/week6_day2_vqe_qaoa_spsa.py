import numpy as np
from dataclasses import dataclass

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    seed: int = 7
    problem: str = "vqe"      # "vqe" -> minimize; "qaoa" -> maximize
    dim: int = 16             # number of parameters (≈ ansatz size)
    steps: int = 250
    shots: int = 2048
    noise: float = 0.02       # measurement noise scale
    a_lr: float = 0.2         # SPSA learning-rate base
    c_pert: float = 0.2       # SPSA perturbation base
    decay: float = 0.101      # learning-rate decay
    decay_c: float = 0.101    # perturbation decay

np.random.seed(CFG.seed)

# ---------------------------- "Quantum" Measurement Stub ----------------------------
def measure_cost(theta: np.ndarray, shots: int, noise: float, problem: str) -> float:
    """
    Simulate an expectation value with shot noise.
    For VQE-like tasks, lower is better; for QAOA-like tasks, higher is better.
    Replace this with a real QC backend call that returns an estimated cost.
    """
    # A non-convex landscape: sum cos + coupled terms
    base = np.sum(np.cos(theta)) / theta.size
    couple = 0.15 * np.mean(np.cos(theta[:-1] - theta[1:]))
    raw = base + couple
    # Problem-specific shift/scale to mimic different objectives
    val = raw - 0.6 if problem == "vqe" else raw + 0.3
    # Add shot noise ~ N(0, noise / sqrt(shots))
    return float(val + np.random.randn() * (noise / np.sqrt(max(1, shots))))

# ---------------------------- SPSA Optimizer ----------------------------
def spsa(theta, k, grad_fn):
    """
    One SPSA step using simultaneous perturbations.
    a_k and c_k follow standard decays a/(k+A)^alpha, c/(k)^gamma (simplified here).
    """
    a_k = CFG.a_lr / (k + 1) ** CFG.decay
    c_k = CFG.c_pert / (k + 1) ** CFG.decay_c
    # Rademacher perturbation
    delta = np.random.choice([-1.0, 1.0], size=theta.shape)
    ghat = grad_fn(theta, c_k, delta)
    return theta - a_k * ghat

def grad_estimator(theta, c_k, delta):
    th_plus  = theta + c_k * delta
    th_minus = theta - c_k * delta
    y_plus  = measure_cost(th_plus,  CFG.shots, CFG.noise, CFG.problem)
    y_minus = measure_cost(th_minus, CFG.shots, CFG.noise, CFG.problem)
    # Estimate gradient along each coordinate (2-sided SPSA)
    ghat = (y_plus - y_minus) / (2.0 * c_k) * delta**(-1)
    return ghat

# ---------------------------- Training Loop ----------------------------
def run():
    theta = np.random.uniform(-np.pi, np.pi, size=(CFG.dim,))
    history = []
    best_val = np.inf if CFG.problem == "vqe" else -np.inf
    best_th  = theta.copy()

    for k in range(CFG.steps):
        cost = measure_cost(theta, CFG.shots, CFG.noise, CFG.problem)
        history.append(cost)

        # Track best
        better = cost < best_val if CFG.problem == "vqe" else cost > best_val
        if better:
            best_val, best_th = cost, theta.copy()

        # SPSA update
        theta = spsa(theta, k, grad_estimator)

        if (k+1) % 20 == 0:
            print(f"[{CFG.problem.upper()}] step {k+1:03d} | cost={cost:.4f} | best={best_val:.4f}")

    return {"best": best_val, "theta": best_th.tolist(), "history": history}

if __name__ == "__main__":
    # VQE-style (minimization)
    CFG.problem = "vqe"
    res_vqe = run()
    print("VQE best:", res_vqe["best"])

    # QAOA-style (maximization)
    CFG.problem = "qaoa"
    res_qaoa = run()
    print("QAOA best:", res_qaoa["best"])
