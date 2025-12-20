# Plots TPOT metrics vs max-num-seqs and max-num-batched-tokens from trace_collection experiments
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from collections import defaultdict

trace_collection_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trace_collection")
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("graphs")

FIXED_MAX_SEQS = 32
FIXED_MAX_TOKENS = 8192

data_by_seqs = defaultdict(lambda: {"Baseline": [], "DefaultAll2All": []})
data_by_tokens = defaultdict(lambda: {"Baseline": [], "DefaultAll2All": []})

for exp_dir in trace_collection_dir.iterdir():
    if not exp_dir.is_dir():
        continue
    
    if "EPLBOFF" in exp_dir.name or "EPLBON" in exp_dir.name:
        continue
    
    parts = exp_dir.name.split("-")
    if len(parts) != 8 or parts[0] != "Mixtral" or parts[4] != "NERSC" or parts[-1] != "12":
        continue
    
    model_type_str = parts[3]
    if "Baseline" in model_type_str:
        model_type = "Baseline"
    elif "DefaultAll2All" in model_type_str:
        model_type = "DefaultAll2All"
    else:
        continue
    
    max_seqs = int(parts[5])
    max_tokens = int(parts[6])
    
    results_dir = exp_dir / "results_json"
    if not results_dir.exists():
        continue
    
    values = []
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            if "mean_tpot_ms" in data:
                values.append(data["mean_tpot_ms"])
    
    if values:
        avg_value = sum(values) / len(values)
        if max_seqs == FIXED_MAX_SEQS:
            data_by_tokens[max_tokens][model_type].append(avg_value)
        if max_tokens == FIXED_MAX_TOKENS:
            data_by_seqs[max_seqs][model_type].append(avg_value)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for model_type in ["Baseline", "DefaultAll2All"]:
    seqs_data = {}
    for max_seqs in data_by_seqs.keys():
        if model_type in data_by_seqs[max_seqs] and data_by_seqs[max_seqs][model_type]:
            seqs_data[max_seqs] = sum(data_by_seqs[max_seqs][model_type]) / len(data_by_seqs[max_seqs][model_type])
    
    if seqs_data:
        sorted_seqs = sorted(seqs_data.keys())
        sorted_values = [seqs_data[s] for s in sorted_seqs]
        ax1.plot(sorted_seqs, sorted_values, marker='o', label=model_type, linewidth=2)
    
    tokens_data = {}
    for max_tokens in data_by_tokens.keys():
        if model_type in data_by_tokens[max_tokens] and data_by_tokens[max_tokens][model_type]:
            tokens_data[max_tokens] = sum(data_by_tokens[max_tokens][model_type]) / len(data_by_tokens[max_tokens][model_type])
    
    if tokens_data:
        sorted_tokens = sorted(tokens_data.keys())
        sorted_values = [tokens_data[t] for t in sorted_tokens]
        ax2.plot(sorted_tokens, sorted_values, marker='o', label=model_type, linewidth=2)

ax1.set_xlabel("Max Num Seqs", fontsize=12)
ax1.set_ylabel("Mean TPOT (ms)", fontsize=12)
ax1.set_title(f"TPOT vs Max Num Seqs (Max Tokens = {FIXED_MAX_TOKENS})", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f"Fixed: Max Tokens = {FIXED_MAX_TOKENS}", transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel("Max Num Batched Tokens", fontsize=12)
ax2.set_ylabel("Mean TPOT (ms)", fontsize=12)
ax2.set_title(f"TPOT vs Max Num Batched Tokens (Max Seqs = {FIXED_MAX_SEQS})", fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f"Fixed: Max Seqs = {FIXED_MAX_SEQS}", transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "tpot_vs_config.png", dpi=300, bbox_inches='tight')
plt.close()

