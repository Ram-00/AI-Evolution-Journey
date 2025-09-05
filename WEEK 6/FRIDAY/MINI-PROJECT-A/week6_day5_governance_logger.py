import time, json, numpy as np, yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# ---------------------------- Policies ----------------------------
@dataclass
class SLO:
    max_latency_ms: int = 100
    min_quality: float = 0.92
    max_retries: int = 1
    shots: int = 1024
    allow_fallback: bool = True

@dataclass
class Metadata:
    backend: str = "simulator"     # "simulator" | "qpu"
    provider: str = "local"
    qubits_used: int = 8
    transpilation_level: int = 1
    seed: int = 42
    error_mitigation: str = "none" # e.g., "ZNE", "M3", "CliffordDataRegression"

@dataclass
class EnergyModel:
    # Simple energy/cost proxies (illustrative)
    joules_per_shot: float = 2e-6      # proxy for control/readout cost
    fixed_overhead_j: float = 0.2      # cryo/control startup
    price_per_kj_usd: float = 0.0003   # proxy price# ---------------------------- Candidate + Fallback Stubs ----------------------------
def quantum_step(params, meta: Metadata, shots: int) -> Dict[str, Any]:
    rng = np.random.default_rng(meta.seed)
    value = 0.93 - 0.05*np.tanh(np.linalg.norm(params)/10.0) + rng.normal(0, 0.02/np.sqrt(max(1, shots)))
    latency = (10 if meta.backend=="simulator" else 45) + rng.integers(0, 20)
    return {"value": float(value), "latency_ms": int(latency)}

def classical_baseline(params, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed+7)
    value = 0.90 - 0.02*np.tanh(np.linalg.norm(params)/10.0) + rng.normal(0, 0.004)
    latency = 12 + rng.integers(0, 8)
    return {"value": float(value), "latency_ms": int(latency)}

# ---------------------------- Governance Wrapper ----------------------------
class GovernanceRunner:
    def __init__(self, slo: SLO, meta: Metadata, energy: EnergyModel):
        self.slo = slo; self.meta = meta; self.energy = energy

    def energy_proxy(self, shots: int) -> Dict[str, float]:
        joules = self.energy.fixed_overhead_j + shots * self.energy.joules_per_shot
        return {"joules": joules, "usd_est": joules/1000.0 * self.energy.price_per_kj_usd}

    def run(self, params, run_id: str) -> Dict[str, Any]:
        log = {
            "run_id": run_id,
            "slo": asdict(self.slo),
            "meta": asdict(self.meta),
            "attempts": [],
            "decision": None
        }

        best = None
        meta = self.meta
        shots = self.slo.shots

        for attempt in range(self.slo.max_retries + 1):
            t0 = time.time()
            out = quantum_step(params, meta, shots)
            wall = int((time.time()-t0)*1000)
            energy = self.energy_proxy(shots)

            rec = {
                "attempt": attempt,
                "backend": meta.backend,
                "shots": shots,
                "seed": meta.seed + attempt,
                "value": out["value"],
                "latency_ms": out["latency_ms"],
                "wall_ms": wall,
                "energy_proxy": energy
            }
            log["attempts"].append(rec)

            ok_quality = out["value"] >= self.slo.min_quality
            ok_latency = out["latency_ms"] <= self.slo.max_latency_ms

            if ok_quality and ok_latency:
                best = {"source":"quantum", **out, "energy_proxy": energy}
                break

            # Adaptive tweaks: change shots or switch backend
            if out["latency_ms"] > self.slo.max_latency_ms and shots > 256:
                shots = max(256, shots // 2)
            elif not ok_quality and meta.backend=="simulator":
                meta = Metadata(**{**asdict(meta), "backend":"qpu"})  # switch to "qpu"
            else:
                meta = Metadata(**{**asdict(meta), "seed": meta.seed + 13})

        if best is None and self.slo.allow_fallback:
            fb = classical_baseline(params, meta.seed)
            best = {"source":"classical_fallback", **fb, "energy_proxy":{"joules":0.0,"usd_est":0.0}}

        log["decision"] = best
        return log

if __name__ == "__main__":
    slo = SLO(max_latency_ms=60, min_quality=0.92, max_retries=2, shots=1024)
    meta = Metadata(backend="simulator", provider="demo", qubits_used=12, transpilation_level=2, error_mitigation="ZNE")
    energy = EnergyModel()
    gov = GovernanceRunner(slo, meta, energy)

    params = np.random.uniform(-np.pi, np.pi, size=(20,))
    out = gov.run(params, "w6d5-0001")
    print(json.dumps(out, indent=2))
    with open("w6d5_governance_log.json", "w") as f:
        json.dump(out, f, indent=2)
