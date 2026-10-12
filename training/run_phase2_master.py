"""
Master Orchestrator for Phase 2 Training Sprint.
Runs student training loop followed by evaluation metrics calculation.
"""

import subprocess
import sys
from pathlib import Path


def run_phase_2():
    print("====================================================")
    print("   CV MAJOR: PHASE 2 TRAINING SPRINT PIPELINE       ")
    print("====================================================")

    base_dir = Path(__file__).resolve().parent

    steps = [
        (str(base_dir / "train_zooming.py"), "Running Training Loop (Wavelets + Multi-Loss)"),
        (str(base_dir.parent / "models" / "metrics.py"), "Calculating Final Evaluation Metrics"),
    ]

    for script, description in steps:
        print(f"\n▶️  STEP: {description} ({Path(script).name})")
        print("-" * 50)
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"\n❌ ERROR: {script} failed with exit code {result.returncode}")
            sys.exit(1)
        print(f"✅ COMPLETED: {description}")

    print("\n" + "=" * 50)
    print("🎉 ALL PHASE 2 STEPS COMPLETED SUCCESSFULLY")
    print("====================================================")


if __name__ == "__main__":
    run_phase_2()
