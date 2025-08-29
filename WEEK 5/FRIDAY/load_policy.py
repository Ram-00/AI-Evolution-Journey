import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).parent
mod_path = ROOT / "MiniProjectA_PlugAndPlay_SafetyLayer" / "week5_day5_guardrails.py"
yaml_path = ROOT / "MiniProjectB_YAMLPolicy_and_RuntimeThresholds" / "safety_policy.yaml"

spec = importlib.util.spec_from_file_location("week5_day5_guardrails", str(mod_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {mod_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.load_policy(str(yaml_path))

