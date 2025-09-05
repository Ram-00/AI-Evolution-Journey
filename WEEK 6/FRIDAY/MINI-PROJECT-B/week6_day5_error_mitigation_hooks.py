from dataclasses import dataclass, asdict
import json, time, numpy as np

@dataclass
class MitigationPolicy:
    method: str = "ZNE"             # "ZNE" | "M3" | "MREM"
    strength: float = 0.5           # tuning knob
    calibrate_every: int = 1000     # shots between recalibration

def apply_mitigation(raw_value: float, policy: MitigationPolicy) -> float:
    # Placeholder: emulate bias reduction proportional to strength
    bias = 0.02 * policy.strength
    return max(min(raw_value + bias, 1.0), 0.0)

def run_measurement_with_mitigation(theta, shots=1024, policy=MitigationPolicy()):
    rng = np.random.default_rng(123)
    raw = 0.90 + rng.normal(0, 0.02/np.sqrt(shots))
    mitigated = apply_mitigation(raw, policy)
    return {"raw": raw, "mitigated": mitigated, "policy": asdict(policy)}

if __name__ == "__main__":
    res = run_measurement_with_mitigation(theta=np.zeros(4))
    print(json.dumps(res, indent=2))
