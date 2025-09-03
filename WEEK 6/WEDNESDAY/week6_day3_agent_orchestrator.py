import time, json, yaml, numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# ---------------------------- Policy & SLO ----------------------------
@dataclass
class SLO:
    max_latency_ms: int = 200
    min_quality: float = 0.90
    max_retries: int = 2
    prefer_simulator: bool = True
    shots: int = 1024

@dataclass
class RunMeta:
    request_id: str
    policy_version: str = "w6d3_v1"
    backend: str = "simulator"  # "qpu" or "simulator"
    shots: int = 1024
    seed: int = 42

# ---------------------------- Stubs to replace later ----------------------------
def quantum_candidate(params: np.ndarray, backend: str, shots: int, seed: int) -> Dict[str, Any]:
    """
    Simulate a quantum subroutine returning an objective estimate with variance ~ 1/sqrt(shots).
    """
    rng = np.random.default_rng(seed)
    # toy objective: higher is better
    true_val = 0.95 - 0.1*np.tanh(np.linalg.norm(params)/10.0)
    noise = (0.04 if backend == "qpu" else 0.02)  # pretend hardware noisier
    est = true_val + rng.normal(0, noise/np.sqrt(max(1, shots)))
    latency = (8 if backend == "simulator" else 35) + rng.integers(0, 20)  # ms
    return {"value": float(est), "latency_ms": int(latency)}

def classical_fallback(params: np.ndarray, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed+1)
    # lower variance, slightly lower mean value to emulate heuristic baseline
    value = 0.90 - 0.02*np.tanh(np.linalg.norm(params)/10.0) + rng.normal(0, 0.005)
    latency = 12 + rng.integers(0, 6)  # ms
    return {"value": float(value), "latency_ms": int(latency)}

# ---------------------------- Agent Orchestrator ----------------------------
class QCOrchestrator:
    def __init__(self, slo: SLO):
        self.slo = slo

    def decide_backend(self) -> str:
        return "simulator" if self.slo.prefer_simulator else "qpu"

    def run(self, params: np.ndarray, req_id: str, allow_fallback: bool = True) -> Dict[str, Any]:
        meta = RunMeta(request_id=req_id, backend=self.decide_backend(), shots=self.slo.shots)
        logs = {"meta": asdict(meta), "slo": asdict(self.slo), "attempts": []}
        best = None

        for attempt in range(self.slo.max_retries + 1):
            start = time.time()
            out = quantum_candidate(params, meta.backend, meta.shots, meta.seed + attempt)
            wall = int((time.time() - start) * 1000)
            record = {"attempt": attempt, "backend": meta.backend, "shots": meta.shots,
                      "value": out["value"], "latency_ms": out["latency_ms"], "wall_ms": wall}
            logs["attempts"].append(record)

            meets_quality = out["value"] >= self.slo.min_quality
            meets_latency = out["latency_ms"] <= self.slo.max_latency_ms
            if meets_quality and meets_latency:
                best = {"source": "quantum", **out}
                break

            # adapt policy: if latency too high, reduce shots; if quality low, switch backend or retry
            if out["latency_ms"] > self.slo.max_latency_ms and meta.shots > 256:
                meta.shots = max(256, meta.shots // 2)  # speed up next try
            elif not meets_quality and meta.backend == "simulator":
                meta.backend = "qpu"  # try hardware for diversity
            else:
                # tweak seed only to shake variance
                meta.seed += 11

        # fallback if still not OK
        if best is None and allow_fallback:
            fb = classical_fallback(params, meta.seed)
            best = {"source": "classical_fallback", **fb}

        logs["decision"] = best
        return logs

# ---------------------------- Demo ----------------------------
if __name__ == "__main__":
    slo = SLO(max_latency_ms=60, min_quality=0.92, max_retries=2, prefer_simulator=True, shots=1024)
    agent = QCOrchestrator(slo)
    params = np.random.uniform(-np.pi, np.pi, size=(24,))
    log = agent.run(params, req_id="req-2025-09-03-001")
    print(json.dumps(log, indent=2))
    # Save JSON for audits
    with open("w6d3_orchestration_log.json", "w") as f:
        json.dump(log, f, indent=2)
