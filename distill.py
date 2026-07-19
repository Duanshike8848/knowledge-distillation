from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)

from data.ag_news import load_ag_news_splits
from distillation.trainer import DistillationSchedule, DistillationTrainer
from evaluation.metrics import compute_classification_metrics
from utils import get_device, load_config, resolve_path, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a teacher model into a small student model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--teacher",
        default=None,
        help="Optional teacher path or Hugging Face model name. Defaults to local teacher_dir if present.",
    )
    return parser.parse_args()


def choose_teacher_source(config: dict, override: str | None) -> str:
    if override:
        return override
    local_teacher = resolve_path(config["paths"]["teacher_dir"])
    if (local_teacher / "config.json").exists():
        return str(local_teacher)
    return config["models"]["teacher_name"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"]["seed"]))

    device = get_device()
    student_name = config["models"]["student_name"]
    teacher_source = choose_teacher_source(config, args.teacher)

    # Tokenizer compatibility note:
    # This first version uses English BERT-family models that share the uncased
    # WordPiece vocabulary. If you switch to a Chinese teacher/student pair on
    # Windows, choose models with compatible tokenizers or add separate teacher
    # tokenization in the distillation dataset.
    datasets, tokenizer, num_labels = load_ag_news_splits(config, student_name)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_source,
        num_labels=num_labels,
    )
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_name,
        num_labels=num_labels,
    )

    teacher_hidden = int(teacher_model.config.hidden_size)
    student_hidden = int(student_model.config.hidden_size)
    projection = None
    if teacher_hidden != student_hidden:
        projection = torch.nn.Linear(student_hidden, teacher_hidden)

    distill_cfg = config["distillation"]
    schedule = DistillationSchedule(
        alpha_start=float(distill_cfg["alpha_start"]),
        alpha_end=float(distill_cfg["alpha_end"]),
        temperature_start=float(distill_cfg["temperature_start"]),
        temperature_end=float(distill_cfg["temperature_end"]),
        feature_loss_weight=float(distill_cfg["feature_loss_weight"]),
    )

    student_dir = resolve_path(config["paths"]["student_dir"])
    output_dir = resolve_path(config["training"]["output_dir"]) / "distill_runs"
    student_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"]["eval_batch_size"]),
        num_train_epochs=float(config["training"]["epochs"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=bool(config["training"]["fp16"] and device.type == "cuda"),
        dataloader_num_workers=int(config["training"]["num_workers"]),
        report_to=[],
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        schedule=schedule,
        feature_projection=projection,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_classification_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(datasets["test"])

    trainer.save_model(str(student_dir))
    tokenizer.save_pretrained(str(student_dir))
    if projection is not None:
        torch.save(projection.state_dict(), student_dir / "feature_projection.pt")
        AutoConfig.from_pretrained(student_name).save_pretrained(student_dir / "student_base_config")
    save_json(
        {
            "teacher_source": teacher_source,
            "student_source": student_name,
            "device": device.type,
            "metrics": metrics,
        },
        Path(student_dir) / "distill_metrics.json",
    )

    print(f"Distilled student saved to: {student_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
