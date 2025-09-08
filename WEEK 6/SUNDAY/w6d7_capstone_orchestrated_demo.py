import numpy as np, json, time
from dataclasses import dataclass, asdict

@dataclass
class SLO:
    max_latency_ms: int = 80
    min_quality: float = 0.93
    max_retries: int = 2
    shots: int = 1024
    allow_fallback: bool = True

@dataclass
class Meta:
    backend: str = "simulator"
    qubits: int = 8
    seed: int = 1
    mitigation: str = "none"

def quantum_estimate(params, meta: Meta, shots: int):
    rng = np.random.default_rng(meta.seed)
    val = 0.92 - 0.05 * np.tanh(np.linalg.norm(params)/10.0) + rng.normal(0, 0.01)
    latency = 8 if meta.backend=="simulator" else 35 + rng.integers(0, 20)
    return {"value": float(val), "latency_ms": int(latency)}

def classical_estimate(params, seed: int):
    rng = np.random.default_rng(seed+9)
    val = 0.90 - 0.02 * np.tanh(np.linalg.norm(params)/10.0) + rng.normal(0, 0.003)
    latency = 12 + rng.integers(0, 14)
    return {"value": float(val), "latency_ms": int(latency)}

class Orchestrator:
    def __init__(self, slo:SLO, meta:Meta):
        self.slo = slo; self.meta = meta
    def run(self, params, req_id:str):
        log = {"run_id":req_id,"slo":asdict(self.slo),"meta":asdict(self.meta),"attempts":[]}
        best = None; meta = self.meta; shots = self.slo.shots
        for attempt in range(self.slo.max_retries+1):
            out = quantum_estimate(params, meta, shots)
            log["attempts"].append({**out,"backend":meta.backend,"shots":shots,"seed":meta.seed+attempt})
            if out["value"] >= self.slo.min_quality and out["latency_ms"] <= self.slo.max_latency_ms:
                best = {"source":"quantum",**out}; break
            # tweak: halve shots or switch backend for next try
            shots = max(shots//2,256)
            meta = Meta(**{**asdict(meta),"backend":"qpu" if meta.backend=="simulator" else "simulator"})
        if best is None and self.slo.allow_fallback:
            fb = classical_estimate(params,meta.seed)
            best = {"source":"classical_fallback",**fb}
        log["decision"]=best
        return log

if __name__=="__main__":
    slo = SLO(max_latency_ms=80,min_quality=0.93,max_retries=2,shots=1024)
    meta = Meta(backend="simulator",qubits=12,mitigation="ZNE")
    orch = Orchestrator(slo,meta)
    params = np.random.uniform(-np.pi,np.pi,size=(16,))
    out = orch.run(params,"w6d7-001")
    print(json.dumps(out,indent=2))
