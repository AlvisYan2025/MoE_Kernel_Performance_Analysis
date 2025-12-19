import json
import matplotlib.pyplot as plt
import argparse
import os

def load_json(path):
    """Load JSON benchmark result file."""
    with open(path, "r") as f:
        return json.load(f)

def plot_metric(batch_sizes, values, title, ylabel, filename, color=None):
    """Generic plotting helper."""
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, values, marker="o", color=color)
    plt.title(title)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved: {filename}")
    plt.close()


def main(json_path):
    data = load_json(json_path)

    results = data["results"]
    batch_sizes = sorted(int(k) for k in results.keys())

    avg_latency = [results[str(b)]["avg_latency_ms"] for b in batch_sizes]
    p50_latency = [results[str(b)]["p50_latency_ms"] for b in batch_sizes]
    p95_latency = [results[str(b)]["p95_latency_ms"] for b in batch_sizes]
    ttft = [results[str(b)]["avg_ttfb_ms"] for b in batch_sizes]
    tpot = [results[str(b)]["avg_tpot_ms"] for b in batch_sizes]

    mode = data.get("mode", "unknown")

    # ----- Plot 1: Latency -----
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, avg_latency, marker="o", label="Avg Latency")
    plt.plot(batch_sizes, p50_latency, marker="o", label="P50 Latency")
    plt.plot(batch_sizes, p95_latency, marker="o", label="P95 Latency")
    plt.title(f"Latency vs Batch Size ({mode})")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_vs_batch.png", dpi=200)
    print("Saved: latency_vs_batch.png")
    plt.close()

    # ----- Plot 2: TTFT -----
    plot_metric(
        batch_sizes, ttft,
        f"TTFT vs Batch Size ({mode})",
        "TTFT (ms)",
        "ttft_vs_batch.png",
        color="purple"
    )

    # ----- Plot 3: TPOT -----
    plot_metric(
        batch_sizes, tpot,
        f"TPOT vs Batch Size ({mode})",
        "TPOT (ms/token)",
        "tpot_vs_batch.png",
        color="green"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to benchmark result JSON file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    main(args.json)