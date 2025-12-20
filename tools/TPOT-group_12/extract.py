# Extracts mean_tpot_ms metric from benchmark JSON files
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
results_dir = exp_dir / "results_json"
values = []
for json_file in results_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
        if "mean_tpot_ms" in data:
            values.append(data["mean_tpot_ms"])
print(sum(values) / len(values) if values else 0)

