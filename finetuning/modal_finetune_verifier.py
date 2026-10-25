"""
Setup
Setup modal
Add hf token and wandb api key as modal secrets. Also configure wandb.init() entity to your account
Create finetune-verifier-vol volume 
Upload dataset to data_dir specified in finetune_config.yaml into volume
Then run with --detach from directory containing yaml config"
"""

from pathlib import Path
import json

import yaml
from munch import munchify
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from loguru import logger
import modal
import wandb

app = modal.App("finetune-verifier")

app_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "datasets",
        "accelerate",
        "torch",
        "munch",
        "pyyaml",
        "loguru",
        "tqdm",
        "peft",
        "scikit-learn",
        "tensorboard",
        "evaluate",
        "wandb",
        "hf_transfer"  # faster downloads
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("finetune-verifier-vol", create_if_missing=True)

BASE_DIR = Path("/data")
CONFIG_PATH = "finetune_config.yaml"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
SEED = 42
GPU = "A100-80GB"
EXAMPLE_TEMPLATE = "Problem: {problem}\nExplanation: {explanation}"
INCLUDE_PROBLEM = False

def prepare_datasets(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    def load_split(split):  # split is one of "train" or "test"
        path = data_dir / f"explanations_dataset_{split}.jsonl"
        dataset = load_dataset("json", data_files=str(path), split="train") # split="train" to just load the whole file
        # cast student_correct bools to int label
        dataset = dataset.map(lambda x: {"labels": int(x["student_correct"])})
        return dataset
    
    train_dataset, test_dataset = load_split("train"), load_split("test")
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=SEED)
    
    return {
        "train": train_val_split["train"],
        "val": train_val_split["test"],
        "test": test_dataset,
    }

@app.function(
    image=app_image,
    gpu=GPU,
    timeout=86400,
    volumes={BASE_DIR: vol},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb-api-key")],
)
def train_and_evaluate(config):
    wandb.init(
        project="cs224n-final-project",
        entity="jsnchon-cs224n",
        name="finetune-verifier",
        config={
            "model": "explanation-verifier",
            "task": "binary-classification",
        }
    )
    
    model_id = config.verifier.id
    logger.info(f"Using {model_id} model")
    logger.info(f"Training settings: {config.training}")

    data_path = Path(BASE_DIR) / Path(config.training.data_dir).name
    raw_datasets = prepare_datasets(str(data_path))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)

    # filter out malformed entries, then tokenize dataset
    for split in raw_datasets:
        logger.info(f"Raw {split} dataset size: {len(raw_datasets[split])}")
    filtered_datasets = {
        split: raw_datasets[split].filter(lambda x: not x["is_malformed"])
        for split in raw_datasets
    }
    for split in filtered_datasets:
        logger.info(f"Filtered {split} dataset size: {len(filtered_datasets[split])}")
        logger.debug(f"First example for {split}: {filtered_datasets[split][0]}")

    def tokenize_function(examples):
        problems = examples["problem"]
        explanations = examples["explanation"]
        if INCLUDE_PROBLEM:
            explanations = [EXAMPLE_TEMPLATE.format(problem=p, explanation=e) for p, e in zip(problems, explanations)]
        return tokenizer(explanations, truncation=True, padding=False)

    logger.info(f"Whether to include the problem in the input: {INCLUDE_PROBLEM}")
    tokenized_datasets = {
        split: filtered_datasets[split].map(tokenize_function, batched=True)
        for split in filtered_datasets
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )

    # llama by default doesn't have pad token, so add one
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = config.training.lora
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        modules_to_save=["classifier", "embed_tokens"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    train_config = config.training
    output_dir = Path(BASE_DIR) / train_config.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=train_config.eval_freq,
        save_strategy="steps",
        save_steps=train_config.eval_freq,
        learning_rate=train_config.lr,
        warmup_steps=train_config.warmup,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        num_train_epochs=train_config.epochs,
        weight_decay=train_config.weight_decay,
        logging_steps=config.logging.logging_freq,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="explanation-classifier",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    logger.info("Evaluating on test set")
    # different metric prefix to separate test set metrics vs validation set metrics
    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    
    wandb.log({"test_accuracy": test_results.get("eval_accuracy"), 
               "test_loss": test_results.get("eval_loss")})
    
    # Save results to volume
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    wandb.finish()
    
    return test_results

@app.local_entrypoint()
def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        config = munchify(config)

    logger.info("Starting explanation verifier finetuning on Modal")
    results = train_and_evaluate.remote(config)
    logger.info(f"Training completed! Test results: {results}")
    return results
