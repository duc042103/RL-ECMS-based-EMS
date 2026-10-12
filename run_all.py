"""Full pipeline: tune ECMS / A-ECMS at 600 kg -> train DDPG -> test (+DP) -> figures."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ("tune_baselines.py", "train_rl.py", "evaluate.py", "make_figures.py")

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== {step}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], check=True)
