import time
import requests
import json
import os
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"

def benchmark_one_request(prompt="Hello", max_tokens=16):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    r = requests.post(API_URL, json=payload)
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000
    data = r.json()

    out = data["choices"][0]["text"]
    num_tokens = max(1, len(out.split()))

    ttfb_ms = latency_ms / (num_tokens + 1)
    tpot_ms = latency_ms / num_tokens

    return latency_ms, ttfb_ms, tpot_ms, num_tokens


def run_batch(batch_size):
    results = []

    def worker():
        return benchmark_one_request()

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        futures = [ex.submit(worker) for _ in range(batch_size)]
        for f in as_completed(futures):
            results.append(f.result())

    return results


def aggregate(all_results):
    lat = [x[0] for x in all_results]
    ttfb = [x[1] for x in all_results]
    tpot = [x[2] for x in all_results]

    return {
        "avg_latency_ms": statistics.mean(lat),
        "p50_latency_ms": statistics.median(lat),
        "p95_latency_ms": statistics.quantiles(lat, n=20)[-1],
        "avg_ttfb_ms": statistics.mean(ttfb),
        "avg_tpot_ms": statistics.mean(tpot),
        "num_requests": len(all_results),
    }


def run_experiment(mode, batch_sizes, total_requests, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    final_result = {
        "mode": mode,
        "total_requests": total_requests,
        "results": {}
    }

    for bs in batch_sizes:
        print(f"\n========== Running batch_size = {bs} ==========")

        all_results = []
        num_batches = total_requests // bs

        for i in range(num_batches):
            print(f"  batch {i+1}/{num_batches} ...")
            batch_res = run_batch(bs)
            all_results.extend(batch_res)

        summary = aggregate(all_results)

        final_result["results"][str(bs)] = summary
        print(f"  Done batch_size = {bs}")

    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"\nSaved JSON → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--total_requests", type=int, default=100)
    parser.add_argument("--out", type=str, default="results_json/baseline.json")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    run_experiment(
        mode=args.mode,
        batch_sizes=batch_sizes,
        total_requests=args.total_requests,
        output_file=args.out,
    )