from __future__ import annotations

import argparse

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from data.ag_news import load_ag_news_splits
from evaluation.metrics import compute_classification_metrics
from utils import get_device, load_config, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier on the AG News test split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/student_distilled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model)

    datasets, tokenizer, num_labels = load_ag_news_splits(config, str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=str(resolve_path(config["training"]["output_dir"]) / "eval_runs"),
        per_device_eval_batch_size=int(config["training"]["eval_batch_size"]),
        dataloader_num_workers=int(config["training"]["num_workers"]),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_classification_metrics,
    )
    metrics = trainer.evaluate()
    save_json(
        {
            "model": str(model_path),
            "device": get_device().type,
            "metrics": metrics,
        },
        model_path / "eval_metrics.json",
    )
    print(metrics)


if __name__ == "__main__":
    main()
