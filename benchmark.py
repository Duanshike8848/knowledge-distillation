from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    bytes_to_mb,
    count_parameters,
    get_device,
    get_path_size_bytes,
    load_config,
    resolve_path,
    save_json,
    synchronize_device,
)


DEFAULT_TEXTS = [
    "Wall Street stocks rose after technology shares recovered in afternoon trading.",
    "The local team won the championship after a dramatic final match.",
    "Scientists reported a new discovery in renewable energy storage.",
    "The government announced a new policy for international trade.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure PyTorch model size, parameters, and latency.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/student_distilled")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")
    return requested


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model)
    output_path = resolve_path(args.output) if args.output else model_path / "benchmark.json"

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    bench_cfg = config.get("benchmark", {})
    warmup_runs = int(bench_cfg.get("warmup_runs", 5))
    timed_runs = int(bench_cfg.get("timed_runs", 30))
    batch_size = int(bench_cfg.get("batch_size", 1))
    texts = (DEFAULT_TEXTS * ((batch_size + len(DEFAULT_TEXTS) - 1) // len(DEFAULT_TEXTS)))[:batch_size]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(config["dataset"]["max_length"]),
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(**inputs)
        synchronize_device(device)

        latencies_ms = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            synchronize_device(device)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    size_bytes = get_path_size_bytes(model_path)
    metrics = {
        "model": str(model_path),
        "device": device.type,
        "parameters": count_parameters(model),
        "model_size_bytes": size_bytes,
        "model_size_mb": bytes_to_mb(size_bytes),
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "latency_ms_avg": round(sum(latencies_ms) / len(latencies_ms), 4),
        "latency_ms_min": round(min(latencies_ms), 4),
        "latency_ms_max": round(max(latencies_ms), 4),
        "throughput_samples_per_second": round(batch_size * 1000 / (sum(latencies_ms) / len(latencies_ms)), 4),
    }
    save_json(metrics, output_path)
    print(metrics)


if __name__ == "__main__":
    main()
