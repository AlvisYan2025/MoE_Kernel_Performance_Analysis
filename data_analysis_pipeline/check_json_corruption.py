import json
from pathlib import Path
import sys

def check_json_corruption(log_dir):
    """Check which JSON files are corrupted"""
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("gate_logs_*.json"))
    
    print(f"Checking {len(log_files)} files in {log_dir}...\n")
    
    corrupted = []
    valid = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                json.load(f)
            valid.append(log_file.name)
            print(f"✓ {log_file.name}")
        except json.JSONDecodeError as e:
            corrupted.append(log_file.name)
            print(f"✗ {log_file.name} - ERROR: {e}")
        except Exception as e:
            corrupted.append(log_file.name)
            print(f"✗ {log_file.name} - ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"Valid files:     {len(valid)}")
    print(f"Corrupted files: {len(corrupted)}")
    print(f"{'='*60}")
    
    if corrupted:
        print("\nCorrupted files:")
        for f in corrupted:
            print(f"  - {f}")

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/h/hy676/final_project/vllm_gate_logs"
    check_json_corruption(log_dir)