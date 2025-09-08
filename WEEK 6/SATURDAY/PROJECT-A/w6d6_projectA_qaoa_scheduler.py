import numpy as np, json, time
from dataclasses import dataclass, asdict

@dataclass
class SLO:
    max_latency_ms: int = 120
    min_quality: float = 0.90
    shots: int = 1024
    max_retries: int = 2

rng = np.random.default_rng(0)

# Tiny 2-machine, 3-job toy: durations matrix
D = np.array([[3,5,2],[4,4,3]])  # shape [machines, jobs]

def greedy_schedule(durations):
    # Assign shortest job next to machine with earliest finish
    m, n = durations.shape
    t = np.zeros(m); order = []
    jobs = list(range(n))
    while jobs:
        # choose job that minimizes makespan increment
        best = None
        for j in jobs:
            k = int(np.argmin(t))  # earliest machine
            inc = durations[k, j]
            score = max(t[k] + inc, t[1-k])
            if best is None or score < best:
                best = (score, j, k)
        score, j, k = best
        t[k] += durations[k, j]
        order.append((j, k))
        jobs.remove(j)
    makespan = t.max()
    quality = 1.0 - (makespan / (durations.sum()/2 + 1e-6))
    return {"order": order, "makespan": float(makespan), "quality": float(quality)}

def qaoa_objective(theta):
    # Nonconvex landscape proxy: better params -> lower makespan -> higher quality
    base = 0.92 - 0.1*np.tanh(np.linalg.norm(theta)/10.0)
    noise = rng.normal(0, 0.02)
    return base + noise

def run_qaoa_like(slo: SLO):
    theta = rng.uniform(-np.pi, np.pi, size=(8,))
    best = -1.0; best_theta = theta.copy(); attempts=[]
    shots = slo.shots; backend="simulator"
    for k in range(slo.max_retries+1):
        t0 = time.time()
        val = qaoa_objective(theta)
        latency = 20 + rng.integers(0, 30)  # ms
        attempts.append({"k":k,"backend":backend,"shots":shots,"value":float(val),"latency_ms":latency})
        if val >= slo.min_quality and latency <= slo.max_latency_ms:
            best = val; best_theta = theta.copy(); break
        # adapt: switch to "qpu", halve shots, perturb theta
        backend = "qpu" if backend=="simulator" else "simulator"
        shots = max(256, shots//2)
        theta += rng.normal(0, 0.2, size=theta.shape)
    return {"value": best, "theta": best_theta.tolist(), "attempts": attempts}

if __name__ == "__main__":
    slo = SLO()
    qc = run_qaoa_like(slo)
    if qc["value"] < 0:  # did not meet SLOs
        fb = greedy_schedule(D)
        decision = {"source":"classical_fallback","detail":fb}
    else:
        fb = greedy_schedule(D)  # baseline for comparison
        decision = {"source":"quantum_like","quality":qc["value"],"baseline_quality":fb["quality"],"attempts":qc["attempts"]}
    print(json.dumps(decision, indent=2))
