# set HF_TOKEN env var
# TODO: qualtiative analysis script for problems only solvable by student with explanation (iterate through labeled
# dataset, for problems where student_correct = True run student on them without explanation and note any where it becomes
# incorrect)
# TODO: some common utilities in here too, eg setting HF_TOKEN in dotenv like Medhya did
# TODO: add eval at the end, move to modal(?)

import os
from pathlib import Path

import yaml
from munch import munchify
import numpy as np
from dotenv import load_dotenv
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

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    logger.warning(
        "HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not set; gated models may fail to download."
    )

CONFIG_PATH = BASE_DIR / "finetune_config.yaml"
SEED = 42

def prepare_datasets(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir

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

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    config = munchify(config)

model_id = config.verifier.id
logger.info(f"Using {model_id} model")
logger.info(f"Training settings: {config.training}")

raw_datasets = prepare_datasets(config.training.data_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

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
    return tokenizer(examples["explanation"], truncation=True, padding=False)

tokenized_datasets = {
    split: filtered_datasets[split].map(tokenize_function, batched=True)
    for split in filtered_datasets
}

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    token=HF_TOKEN,
)

# for finetuning, we need to add a distinct pad token (using eos might make the model learn undesirable things
# about when to end sequences) and right pad with it
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
    # mark the classifier head to be trained/saved
    modules_to_save=["classifier"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

train_config = config.training
training_args = TrainingArguments(
    output_dir=train_config.out_dir,
    eval_strategy="steps",        # Run validation periodically
    eval_steps=train_config.eval_freq,
    save_strategy="steps",
    save_steps=train_config.eval_freq,
    learning_rate=train_config.lr,
    per_device_train_batch_size=train_config.batch_size,
    per_device_eval_batch_size=train_config.batch_size,
    num_train_epochs=train_config.epochs,
    weight_decay=train_config.weight_decay,
    logging_steps=config.logging.logging_freq,
    load_best_model_at_end=True,
    report_to=config.logging.logger,
    run_name="explanation-classifier",
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

print("\n--- Final Evaluation on Test Set ---")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)
