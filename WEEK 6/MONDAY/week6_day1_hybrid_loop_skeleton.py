import numpy as np
from dataclasses import dataclass

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    seed: int = 42
    p_layers: int = 2          # circuit depth hyperparameter
    steps: int = 200           # classical optimization steps
    lr: float = 0.1            # learning rate for simple SGD
    shots: int = 2048          # measurement shots (simulated)
    problem: str = "maxcut"    # or "ground_state"
    noise: float = 0.01        # shot noise std (simulated)

np.random.seed(CFG.seed)

# ---------------------------- Quantum Stubs (to be replaced) ----------------------------
def quantum_expectation(params: np.ndarray, problem: str, shots: int, noise: float) -> float:
    """
    Simulate an expectation value <H> with noise and finite-sample effects.
    Replace with a real backend call later.
    """
    # toy nonconvex landscape: sum of cosines + problem-dependent shift
    base = np.sum(np.cos(params))
    shift = -0.5 if problem == "ground_state" else 0.0
    # shot noise ~ N(0, noise/sqrt(shots))
    measured = base / params.size + shift + np.random.randn() * (noise / np.sqrt(max(1, shots)))
    return float(measured)

def parameter_shift_grad(params: np.ndarray, problem: str, shots: int, noise: float, s: float=0.5) -> np.ndarray:
    """
    Mimic parameter-shift gradient via finite difference with ±s.
    """
    grads = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy();  p_plus[i]  += s
        p_minus = params.copy(); p_minus[i] -= s
        grads[i] = (quantum_expectation(p_plus, problem, shots, noise) -
                    quantum_expectation(p_minus, problem, shots, noise)) / (2*s)
    return grads

# ---------------------------- Hybrid Optimization Loop ----------------------------
def run_hybrid(cfg: CFG):
    dim = 2 * cfg.p_layers
    theta = np.random.uniform(-np.pi, np.pi, size=(dim,))
    history = []
    best = (float("inf"), theta.copy()) if cfg.problem == "ground_state" else (-float("inf"), theta.copy())

    for step in range(1, cfg.steps+1):
        exp_cost = quantum_expectation(theta, cfg.problem, cfg.shots, cfg.noise)
        grad = parameter_shift_grad(theta, cfg.problem, cfg.shots, cfg.noise)
        theta -= cfg.lr * grad

        # track best depending on objective
        if cfg.problem == "ground_state":
            if exp_cost < best[0]: best = (exp_cost, theta.copy())
        else:
            if exp_cost > best[0]: best = (exp_cost, theta.copy())

        history.append(exp_cost)
        if step % 20 == 0:
            print(f"[step {step:03d}] cost={exp_cost:.4f} | ||grad||={np.linalg.norm(grad):.3f}")

    return {"history": history, "best_value": best[0], "best_params": best[1].tolist()}

if __name__ == "__main__":
    res_vqe  = run_hybrid(CFG)                         # pretend ground state (minimize)
    CFG.problem = "ground_state"
    print("Ground-state style objective:", res_vqe["best_value"])
    CFG.problem = "maxcut"
    res_qaoa = run_hybrid(CFG)                         # pretend maxcut (maximize)
    print("Max-cut style objective:", res_qaoa["best_value"])
