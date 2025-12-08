import json
import csv
from pathlib import Path
import re
import sys

def parse_filename(filename):
    """
    Parse filename to extract kernel type and batch size.
    Expected format: {kernel}_{batch_size}.json
    Examples: mixed_1.json, baseline_4.json
    """
    match = re.match(r'([a-zA-Z_]+)_(\d+)\.json', filename)
    if match:
        kernel_type = match.group(1)
        batch_size = int(match.group(2))
        return kernel_type, batch_size
    return None, None

def extract_metrics(json_file):
    """Extract TTFT and TPOT metrics from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return {
        'mean_ttft_ms': data.get('mean_ttft_ms'),
        'median_ttft_ms': data.get('median_ttft_ms'),
        'p99_ttft_ms': data.get('p99_ttft_ms'),
        'mean_tpot_ms': data.get('mean_tpot_ms'),
        'median_tpot_ms': data.get('median_tpot_ms'),
        'p99_tpot_ms': data.get('p99_tpot_ms'),
        'request_throughput': data.get('request_throughput'),
        'input_throughput': data.get('input_throughput'),
        'output_throughput': data.get('output_throughput'),
        'imbalance_score': data.get('imbalance_score')
    }

def process_directory(input_dir, output_csv):
    """Process all JSON files in directory and create CSV"""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    results = []
    
    for json_file in json_files:
        kernel_type, batch_size = parse_filename(json_file.name)
        
        if kernel_type is None:
            print(f"Skipping {json_file.name} - doesn't match expected pattern")
            continue
        
        try:
            metrics = extract_metrics(json_file)
            
            row = {
                'kernel_type': kernel_type,
                'batch_size': batch_size,
                **metrics
            }
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Sort by kernel_type and batch_size
    results.sort(key=lambda x: (x['kernel_type'], x['batch_size']))
    
    # Write to CSV
    if results:
        fieldnames = ['kernel_type', 'batch_size', 'mean_ttft_ms', 'median_ttft_ms', 
                     'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
                     'request_throughput', 'input_throughput', 'output_throughput',
                     'imbalance_score']
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Successfully created {output_csv} with {len(results)} rows")
    else:
        print("No valid results to write")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_batch_metrics.py <input_directory> [output.csv]")
        print("Example: python extract_batch_metrics.py ./results batch_size_results.csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "batch_size_metrics.csv"
    
    process_directory(input_dir, output_csv)