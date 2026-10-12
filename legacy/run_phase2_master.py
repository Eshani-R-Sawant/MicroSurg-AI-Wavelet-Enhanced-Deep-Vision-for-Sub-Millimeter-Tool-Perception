import sys
import subprocess

def run_phase_2():
    print("====================================================")
    print("   CV MAJOR: PHASE 2 TRAINING SPRINT PIPELINE       ")
    print("====================================================")
    
    steps = [
        
        
        ("train_zooming.py", "Running Training Loop (Wavelets + Multi-Loss)"),
        ("metrics.py", "Calculating Final mIoU and mAP")
    ]
    
    for script, description in steps:
        print(f"\n▶️  STEP: {description} ({script})")
        print("-" * 50)
        
        # Use python3 to execute the script
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
