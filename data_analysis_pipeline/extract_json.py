#extract json output and write to csv 
import json
import csv
import sys
from pathlib import Path


def extract_metrics(json_file):
    """Extract relevant metrics from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return {
        'file': json_file.name,
        'date': data.get('date', ''),
        'model_id': data.get('model_id', ''),
        'imbalance_score': data.get('imbalance_score', ''),
        'mean_ttft_ms': data.get('mean_ttft_ms', ''),
        'p99_ttft_ms': data.get('p99_ttft_ms', ''),
        'mean_tpot_ms': data.get('mean_tpot_ms', ''),
        'p99_tpot_ms': data.get('p99_tpot_ms', ''),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_metrics.py <input_path> <output_csv>")
        print("\n  input_path: Directory containing JSON files or a single JSON file")
        print("  output_csv: Path to output CSV file")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_csv = sys.argv[2]
    
    # Collect all JSON files
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.glob('*.json'))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    # Extract metrics from all files
    all_metrics = []
    for json_file in json_files:
        try:
            metrics = extract_metrics(json_file)
            all_metrics.append(metrics)
            print(f"Processed: {json_file.name}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not all_metrics:
        print("No metrics extracted")
        sys.exit(1)
    
    # Write to CSV
    fieldnames = ['file', 'date', 'model_id', 'imbalance_score', 
                  'mean_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'p99_tpot_ms']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\nSuccessfully wrote {len(all_metrics)} records to {output_csv}")


if __name__ == '__main__':
    main()