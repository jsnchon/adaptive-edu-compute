"""
Get these qualtiative metrics:
- Confusion matrix (TP, TN, FP, FN)
- Precision, Recall, F1 score
- Proportion of predicted positives vs negatives
- Human-readable files with predictions organized by class
- Additional analysis (confidence distribution, error patterns)
"""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuration
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
WEIGHTS_DIR = "finetuning/with_problem"
DATASET_DIR = "explanations_dataset"
EXAMPLE_TEMPLATE = "Problem: {problem}\nExplanation: {explanation}"
BATCH_SIZE = 8
OUTPUT_DIR = "finetuning/qualitative_analysis"

def load_test_dataset(data_dir):
    data_dir = Path(data_dir)
    
    path = data_dir / "explanations_dataset_test.jsonl"
    logger.info(f"Loading test dataset from {path}")
    
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.map(lambda x: {"labels": int(x["student_correct"])})
    dataset = dataset.filter(lambda x: not x["is_malformed"])
    
    logger.info(f"Loaded {len(dataset)} filtered test examples")
    return dataset

def load_model_and_tokenizer(model_id, weights_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loading LoRA weights from {weights_dir}")
    model = PeftModel.from_pretrained(model, weights_dir)
    model.eval()
    
    return model, tokenizer

def run_inference(model, tokenizer, dataset, batch_size=BATCH_SIZE):
    predictions = []
    confidences = []
    true_labels = []
    
    logger.info("Running inference on test dataest")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        problems = batch["problem"]
        explanations = batch["explanation"]
        
        formatted = [EXAMPLE_TEMPLATE.format(problem=p, explanation=e) 
                    for p, e in zip(problems, explanations)]
        inputs = tokenizer(formatted, truncation=True, padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        conf = probs.max(dim=-1).values.to(torch.float32).cpu().numpy()
        
        predictions.extend(preds.tolist())
        confidences.extend(conf.tolist())
        true_labels.extend(batch["labels"])
    
    return np.array(predictions), np.array(confidences), np.array(true_labels)

def compute_metrics(predictions, true_labels):
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    prop_positive = np.sum(predictions == 1) / len(predictions)
    prop_negative = np.sum(predictions == 0) / len(predictions)
    
    true_prop_positive = np.sum(true_labels == 1) / len(true_labels)
    true_prop_negative = np.sum(true_labels == 0) / len(true_labels)
    
    return {
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "proportion_predicted_positive": float(prop_positive),
        "proportion_predicted_negative": float(prop_negative),
        "true_proportion_positive": float(true_prop_positive),
        "true_proportion_negative": float(true_prop_negative),
    }

def save_predictions_by_class(dataset, predictions, confidences, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_1_file = output_dir / "predicted_positive_explanations.txt"
    pred_0_file = output_dir / "predicted_negative_explanations.txt"
    positive_examples = []
    negative_examples = []
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        example = {
            "id": dataset["id"][i],
            "problem": dataset["problem"][i],
            "explanation": dataset["explanation"][i],
            "predicted_class": int(pred),
            "confidence": float(conf),
            "ground_truth": int(dataset["labels"][i]),
            "correct": int(pred) == int(dataset["labels"][i]),
        }
        
        if pred == 1:
            positive_examples.append(example)
        else:
            negative_examples.append(example)
    
    positive_examples.sort(key=lambda x: x["confidence"], reverse=True)
    negative_examples.sort(key=lambda x: x["confidence"], reverse=True)
    
    with open(pred_1_file, "w") as f:
        for i, example in enumerate(positive_examples):
            f.write(f"Problem: {example['problem']}\nExplanation: {example['explanation']}\n\n")

    with open(pred_0_file, "w") as f:
        for i, example in enumerate(negative_examples):
            f.write(f"Problem: {example['problem']}\nExplanation: {example['explanation']}\n\n")

def plot_confusion_matrix(predictions, true_labels, output_dir):
    output_dir = Path(output_dir)
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    plot_file = output_dir / "confusion_matrix.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {plot_file}")

def plot_metrics(metrics, output_dir):
    output_dir = Path(output_dir)
    
    metric_names = ["Precision", "Recall", "F1 Score", "Accuracy"]
    metric_values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["accuracy"],
    ]
    
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight="bold")
    
    plt.ylim([0, 1.05])
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title("Performance Metrics", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    plot_file = output_dir / "performance_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved metrics plot to {plot_file}")

def plot_prediction_distribution(predictions, true_labels, output_dir):
    output_dir = Path(output_dir)
    
    pred_positive = np.sum(predictions == 1)
    pred_negative = np.sum(predictions == 0)
    true_positive = np.sum(true_labels == 1)
    true_negative = np.sum(true_labels == 0)
    
    x = np.arange(2)
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, [true_negative, true_positive], width, label="True Labels",
                   color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = plt.bar(x + width/2, [pred_negative, pred_positive], width, label="Predicted Labels",
                   color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    plt.xlabel("Class", fontsize=12, fontweight="bold")
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.title("Label Distribution: True vs Predicted", fontsize=14, fontweight="bold")
    plt.xticks(x, ["Negative (0)", "Positive (1)"])
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    plot_file = output_dir / "prediction_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved prediction distribution plot to {plot_file}")

def plot_confidence_histogram(confidences, predictions, true_labels, output_dir):
    output_dir = Path(output_dir)
    
    # separate confidences by prediction and correctness
    correct_mask = predictions == true_labels
    incorrect_mask = predictions != true_labels
    
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[incorrect_mask]
    
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0.5, 1.0, 30)
    plt.hist(correct_confidences, bins=bins, alpha=0.7, label=f"Correct ({len(correct_confidences)})",
            color="#2ca02c", edgecolor="black", linewidth=1.2)
    plt.hist(incorrect_confidences, bins=bins, alpha=0.7, label=f"Incorrect ({len(incorrect_confidences)})",
            color="#d62728", edgecolor="black", linewidth=1.2)
    
    plt.xlabel("Confidence Score", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title("Distribution of Model Confidence Scores", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    plot_file = output_dir / "confidence_histogram.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved confidence histogram to {plot_file}")
    return plot_file

def main():
    dataset = load_test_dataset(DATASET_DIR)
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, WEIGHTS_DIR)
    
    predictions, confidences, true_labels = run_inference(model, tokenizer, dataset)
    
    metrics = compute_metrics(predictions, true_labels)
    logger.info(f"Metrics: {metrics}")
    
    save_predictions_by_class(dataset, predictions, confidences, OUTPUT_DIR)
    
    plot_confusion_matrix(predictions, true_labels, OUTPUT_DIR)
    plot_metrics(metrics, OUTPUT_DIR)
    plot_prediction_distribution(predictions, true_labels, OUTPUT_DIR)
    plot_confidence_histogram(confidences, predictions, true_labels, OUTPUT_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"{'=' * 50}SUMMARY{'=' * 50}") 
    cm = metrics["confusion_matrix"]
    logger.info(f"TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    logger.info(f"Predicted Positive: {metrics['proportion_predicted_positive']:.4f} | Predicted Negative: {metrics['proportion_predicted_negative']:.4f}")

if __name__ == "__main__":
    main()
