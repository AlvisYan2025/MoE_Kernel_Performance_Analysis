import time
import argparse
import statistics
import requests

API_URL = "http://localhost:8000/v1/completions"

def run_benchmark(num_prompts, batch_size, max_tokens):
    prompts = ["Hello world"] * num_prompts  # same prompt for stability
    latencies = []

    print(f"Sending {num_prompts} requests (batch size {batch_size})...")

    for i in range(0, num_prompts, batch_size):
        batch = prompts[i:i+batch_size]

        payload = {
            "model": "mistralai/Mixtral-8x7B-v0.1",
            "prompt": batch,
            "max_tokens": max_tokens
        }

        start = time.time()
        response = requests.post(API_URL, json=payload)
        end = time.time()

        if response.status_code != 200:
            print("❌ Error:", response.text)
            continue

        latency = (end - start) * 1000  # ms
        latencies.append(latency)

        print(f"[{i+1}/{num_prompts}] batch latency = {latency:.2f} ms")

    if len(latencies) == 0:
        print("No successful requests.")
        return

    # Stats
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # approximate 95th percentile
    throughput = num_prompts / (sum(latencies) / 1000)

    print("\n========== Benchmark Results ==========")
    print(f"Total requests: {num_prompts}")
    print(f"Batch size: {batch_size}")
    print(f"Average latency: {avg:.2f} ms")
    print(f"P50 latency: {p50:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print(f"Throughput: {throughput:.2f} req/s")
    print("=======================================\n")

    # Save results
    with open("benchmark_results.csv", "a") as f:
        f.write(f"{num_prompts},{batch_size},{avg:.2f},{p50:.2f},{p95:.2f},{throughput:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8)

    args = parser.parse_args()
    run_benchmark(args.num_prompts, args.batch_size, args.max_tokens)