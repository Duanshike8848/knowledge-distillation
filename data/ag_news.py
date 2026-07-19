from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_ag_news_splits(config: dict[str, Any], tokenizer_name: str) -> tuple[DatasetDict, PreTrainedTokenizerBase, int]:
    """Download, split, and tokenize AG News.

    The Hugging Face datasets cache is platform independent. On Windows, the
    same function works unchanged as long as `datasets` can access the network
    or the dataset has already been cached.
    """
    dataset_cfg = config["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    raw = load_dataset(dataset_cfg["name"])

    split = raw["train"].train_test_split(
        test_size=float(dataset_cfg["validation_split_ratio"]),
        seed=int(config["training"]["seed"]),
        stratify_by_column=dataset_cfg["label_column"],
    )
    datasets = DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )

    max_train = dataset_cfg.get("max_train_samples")
    max_eval = dataset_cfg.get("max_eval_samples")
    if max_train:
        datasets["train"] = datasets["train"].select(range(min(int(max_train), len(datasets["train"]))))
    if max_eval:
        for name in ("validation", "test"):
            datasets[name] = datasets[name].select(range(min(int(max_eval), len(datasets[name]))))

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch[dataset_cfg["text_column"]],
            truncation=True,
            max_length=int(dataset_cfg["max_length"]),
        )

    tokenized = datasets.map(tokenize, batched=True)
    tokenized = tokenized.rename_column(dataset_cfg["label_column"], "labels")
    keep_columns = ["input_ids", "attention_mask", "labels"]
    tokenized = tokenized.remove_columns(
        [col for col in tokenized["train"].column_names if col not in keep_columns]
    )
    return tokenized, tokenizer, raw["train"].features[dataset_cfg["label_column"]].num_classes
