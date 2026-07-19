from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import bytes_to_mb, get_path_size_bytes, load_config, resolve_path, save_json


SAMPLE_TEXT = "Technology companies reported strong earnings after the market closed."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a saved PyTorch classifier to ONNX.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/student_distilled")
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def validate_with_onnxruntime(onnx_path: Path, torch_logits: np.ndarray, inputs: dict[str, torch.Tensor]) -> dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
    }
    ort_logits = session.run(["logits"], ort_inputs)[0]
    diff = np.abs(torch_logits - ort_logits)
    return {
        "onnx_max_abs_diff": float(diff.max()),
        "onnx_mean_abs_diff": float(diff.mean()),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output) if args.output else resolve_path(config["paths"]["onnx_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to("cpu")
    model.eval()

    inputs = tokenizer(
        SAMPLE_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=int(config["dataset"]["max_length"]),
    )

    with torch.no_grad():
        torch_outputs = model(**inputs)
        torch_logits = torch_outputs.logits.cpu().numpy()

    # Portability note:
    # Exporting on CPU avoids macOS MPS/CUDA-specific graph quirks. The ONNX file
    # can then be copied to Windows, Linux, or edge conversion toolchains.
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=int(args.opset),
        do_constant_folding=True,
    )

    validation = validate_with_onnxruntime(onnx_path, torch_logits, inputs)
    metrics = {
        "source_model": str(model_path),
        "onnx_path": str(onnx_path),
        "opset": int(args.opset),
        "onnx_size_bytes": get_path_size_bytes(onnx_path),
        "onnx_size_mb": bytes_to_mb(get_path_size_bytes(onnx_path)),
        **validation,
    }
    save_json(metrics, output_dir / "onnx_export_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
