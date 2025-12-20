#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path

METRIC_TO_SCRIPT = {
    "TTFT": "TTFT-group_12/extract.py",
    "TPOT": "TPOT-group_12/extract.py",
    "request_throughput": "requestThroughput-group_12/extract.py",
    "output_throughput": "outputThroughput-group_12/extract.py",
    "token_to_expert_assignment": "tokenToExpertAssignment-group_12/extract.py",
}

def main():
    parser = argparse.ArgumentParser(description="Extract metrics from trace directories")
    parser.add_argument("--trace", required=True, help="Path to trace directory")
    parser.add_argument("--metric", required=True, help="Name of metric to extract")
    
    args = parser.parse_args()
    
    trace_dir = Path(args.trace)
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.metric not in METRIC_TO_SCRIPT:
        print(f"Error: Unknown metric: {args.metric}", file=sys.stderr)
        print(f"Available metrics: {', '.join(METRIC_TO_SCRIPT.keys())}", file=sys.stderr)
        sys.exit(1)
    
    script_path = Path(__file__).parent / METRIC_TO_SCRIPT[args.metric]
    if not script_path.exists():
        print(f"Error: Metric script not found: {script_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(trace_dir)],
            capture_output=True,
            text=True,
            check=False
        )
        
        sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running metric script: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

