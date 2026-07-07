#!/usr/bin/env python3
"""
Latency/throughput benchmark for v1/embeddings-compatible endpoints.

Example:
  Text output to STDOUT:
  python benchmark_embeddings.py --endpoint-url https://<host>/v1/embeddings --batch-size 8 --samples 1000 --output-format text

  JSON output to STDOUT:
  python benchmark_embeddings.py --endpoint-url https://<host>/v1/embeddings --batch-size 8 --samples 1000 --output-format json

Notes:
  This script benchmarks endpoint behavior only (network + server + response),
  and does not run local model inference with torch. The server is expected
  to already have the desired model loaded.
"""

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from urllib import request


@dataclass
class BenchmarkResult:
    batch_size: int
    samples: int
    total_seconds: float
    throughput_samples_per_sec: float
    avg_batch_latency_ms: float
    p50_batch_latency_ms: float
    p95_batch_latency_ms: float
    avg_sample_latency_ms: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model selection and benchmark settings.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark embeddings endpoint latency and throughput")
    parser.add_argument("-b", "--batch-size", "--batch-sizes", dest="batch_size", type=int, default=8, help="Batch size to benchmark")
    parser.add_argument("-s", "--samples", type=int, default=1000, help="Number of input texts per batch size")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Number of warmup iterations per batch size")
    parser.add_argument("-r", "--runs", type=int, default=5, help="Number of repeated benchmark runs")
    parser.add_argument("-R", "--rng-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-o",
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format written to STDOUT",
    )
    parser.add_argument("-e", "--endpoint-url", type=str, required=True, help="Full URL to the v1/embeddings endpoint")
    parser.add_argument("-k", "--api-key", type=str, default=None, help="Optional bearer token for endpoint auth")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="Number of parallel endpoint requests")
    parser.add_argument("-u", "--request-timeout-sec", type=float, default=60.0, help="Per-request endpoint timeout in seconds")
    return parser.parse_args()


def _mean_and_stddev(values: List[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation (0.0 for single value)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize_runs(batch_size: int, run_results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Aggregate repeated run metrics into mean/stddev and include per-run stats."""
    total_seconds_mean, total_seconds_stddev = _mean_and_stddev(
        [r.total_seconds for r in run_results]
    )
    throughput_mean, throughput_stddev = _mean_and_stddev(
        [r.throughput_samples_per_sec for r in run_results]
    )
    avg_batch_latency_mean, avg_batch_latency_stddev = _mean_and_stddev(
        [r.avg_batch_latency_ms for r in run_results]
    )
    p50_latency_mean, p50_latency_stddev = _mean_and_stddev(
        [r.p50_batch_latency_ms for r in run_results]
    )
    p95_latency_mean, p95_latency_stddev = _mean_and_stddev(
        [r.p95_batch_latency_ms for r in run_results]
    )
    avg_sample_latency_mean, avg_sample_latency_stddev = _mean_and_stddev(
        [r.avg_sample_latency_ms for r in run_results]
    )

    run_stats = []
    for index, result in enumerate(run_results, start=1):
        run_payload = asdict(result)
        run_payload["run_number"] = index
        run_stats.append(run_payload)

    samples_value = run_results[0].samples if run_results else 0
    return {
        "batch_size": batch_size,
        "samples": samples_value,
        "total_seconds": total_seconds_mean,
        "total_seconds_stddev": total_seconds_stddev,
        "throughput_samples_per_sec": throughput_mean,
        "throughput_samples_per_sec_stddev": throughput_stddev,
        "avg_batch_latency_ms": avg_batch_latency_mean,
        "avg_batch_latency_ms_stddev": avg_batch_latency_stddev,
        "p50_batch_latency_ms": p50_latency_mean,
        "p50_batch_latency_ms_stddev": p50_latency_stddev,
        "p95_batch_latency_ms": p95_latency_mean,
        "p95_batch_latency_ms_stddev": p95_latency_stddev,
        "avg_sample_latency_ms": avg_sample_latency_mean,
        "avg_sample_latency_ms_stddev": avg_sample_latency_stddev,
        "run_stats": run_stats,
    }


def generate_texts(n: int, min_words: int = 8, max_words: int = 32) -> List[str]:
    """Generate synthetic text inputs for repeatable throughput/latency testing.

    Args:
        n: Number of texts to generate.
        min_words: Minimum number of words per text.
        max_words: Maximum number of words per text.

    Returns:
        List[str]: Generated synthetic texts.
    """
    words = [
        "performance",
        "latency",
        "throughput",
        "vector",
        "search",
        "embedding",
        "benchmark",
        "query",
        "document",
        "token",
        "index",
        "model",
        "semantic",
        "ranking",
        "retrieval",
    ]

    texts = []
    for _ in range(n):
        k = random.randint(min_words, max_words)
        toks = random.choices(words, k=k)
        suffix = "".join(random.choices(string.ascii_lowercase, k=6))
        texts.append(" ".join(toks) + f" {suffix}")
    return texts


def percentile(values: List[float], p: float) -> float:
    """Return a linear-interpolated percentile from a pre-sorted list.

    Args:
        values: Sorted numeric values.
        p: Percentile in [0.0, 1.0].

    Returns:
        float: Interpolated percentile value.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return values[low]
    frac = idx - low
    return values[low] * (1 - frac) + values[high] * frac


def encode_batch_via_endpoint(
    endpoint_url: str,
    api_key: str | None,
    texts: List[str],
    timeout_sec: float,
) -> None:
    """Send a single embeddings request to a v1/embeddings-compatible endpoint.

    Args:
        endpoint_url: Full endpoint URL.
        api_key: Optional bearer token.
        texts: Input texts for this request.
        timeout_sec: Request timeout in seconds.

    Returns:
        None
    """
    payload = json.dumps({"input": texts}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(endpoint_url, data=payload, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_sec) as response:
        _ = response.read()


def benchmark_one_batch_size_endpoint(
    endpoint_url: str,
    api_key: str | None,
    batch_size: int,
    samples: int,
    warmup: int,
    concurrency: int,
    timeout_sec: float,
) -> BenchmarkResult:
    """Benchmark one batch size using remote v1/embeddings endpoint requests.

    Args:
        endpoint_url: Full endpoint URL.
        api_key: Optional bearer token.
        batch_size: Request batch size to benchmark.
        samples: Number of total input texts to process.
        warmup: Number of warmup requests before timing.
        concurrency: Number of requests sent in parallel.
        timeout_sec: Per-request timeout in seconds.

    Returns:
        BenchmarkResult: Aggregated metrics for this batch size.
    """
    texts = generate_texts(samples)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    if not batches:
        raise ValueError("No batches generated. Increase --samples.")

    warmup_batch = batches[0]
    for _ in range(warmup):
        encode_batch_via_endpoint(endpoint_url, api_key, warmup_batch, timeout_sec)

    latencies = []
    total_texts = 0
    total_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i in range(0, len(batches), concurrency):
            chunk = batches[i : i + concurrency]
            chunk_start = time.perf_counter()

            futures = [
                executor.submit(
                    encode_batch_via_endpoint,
                    endpoint_url,
                    api_key,
                    b,
                    timeout_sec,
                )
                for b in chunk
            ]

            for future in futures:
                future.result()

            chunk_end = time.perf_counter()
            chunk_latency = chunk_end - chunk_start

            for b in chunk:
                latencies.append(chunk_latency)
                total_texts += len(b)

    total_end = time.perf_counter()
    total_seconds = total_end - total_start

    lat_sorted = sorted(latencies)
    avg_batch_latency_s = statistics.mean(latencies)

    return BenchmarkResult(
        batch_size=batch_size,
        samples=total_texts,
        total_seconds=total_seconds,
        throughput_samples_per_sec=(total_texts / total_seconds) if total_seconds > 0 else 0.0,
        avg_batch_latency_ms=avg_batch_latency_s * 1000,
        p50_batch_latency_ms=percentile(lat_sorted, 0.50) * 1000,
        p95_batch_latency_ms=percentile(lat_sorted, 0.95) * 1000,
        avg_sample_latency_ms=((total_seconds / total_texts) * 1000) if total_texts > 0 else 0.0,
    )


def print_results(results: List[BenchmarkResult], target: str) -> None:
    """Print benchmark metrics in a compact tabular format.

    Args:
        results: Benchmark results for each batch size.
        target: Benchmark target label (for example, endpoint URL).

    Returns:
        None
    """
    print("\n=== Embeddings Benchmark ===")
    print(f"Target: {target}")
    print()
    header = (
        f"{'Batch':>6}  {'Samples':>8}  {'Total(s)':>9}  {'Thrpt(smp/s)':>13}  "
        f"{'AvgBatch(ms)':>12}  {'P50(ms)':>8}  {'P95(ms)':>8}  {'AvgSample(ms)':>13}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.batch_size:>6}  {r.samples:>8}  {r.total_seconds:>9.3f}  "
            f"{r.throughput_samples_per_sec:>13.2f}  {r.avg_batch_latency_ms:>12.3f}  "
            f"{r.p50_batch_latency_ms:>8.3f}  {r.p95_batch_latency_ms:>8.3f}  "
            f"{r.avg_sample_latency_ms:>13.4f}"
        )


def print_results_json(results: List[BenchmarkResult], target: str) -> None:
    """Print benchmark metrics as a JSON object to STDOUT.

    Args:
        results: Benchmark results for each batch size.
        target: Benchmark target label (for example, endpoint URL).

    Returns:
        None
    """
    payload = {
        "target": target,
        "results": results,
    }
    json.dump(payload, sys.stdout, indent=2)
    print()


def main() -> None:
    """Run endpoint benchmark for the configured batch size and print results.

    Returns:
        None
    """
    args = parse_args()
    random.seed(args.rng_seed)

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.request_timeout_sec <= 0:
        raise ValueError("--request-timeout-sec must be > 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    print(f"Using endpoint '{args.endpoint_url}'...", file=sys.stderr)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    run_results = []
    for run_index in range(1, args.runs + 1):
        print(
            f"Benchmarking batch_size={args.batch_size}, run {run_index}/{args.runs}...",
            file=sys.stderr,
        )
        result = benchmark_one_batch_size_endpoint(
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            batch_size=args.batch_size,
            samples=args.samples,
            warmup=args.warmup,
            concurrency=args.concurrency,
            timeout_sec=args.request_timeout_sec,
        )
        run_results.append(result)

    summary = summarize_runs(args.batch_size, run_results)
    avg_result = BenchmarkResult(
        batch_size=summary["batch_size"],
        samples=summary["samples"],
        total_seconds=summary["total_seconds"],
        throughput_samples_per_sec=summary["throughput_samples_per_sec"],
        avg_batch_latency_ms=summary["avg_batch_latency_ms"],
        p50_batch_latency_ms=summary["p50_batch_latency_ms"],
        p95_batch_latency_ms=summary["p95_batch_latency_ms"],
        avg_sample_latency_ms=summary["avg_sample_latency_ms"],
    )

    if args.output_format == "json":
        print_results_json([summary], args.endpoint_url)
    else:
        print_results([avg_result], args.endpoint_url)


if __name__ == "__main__":
    main()
