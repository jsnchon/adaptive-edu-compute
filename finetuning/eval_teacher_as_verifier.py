#!/usr/bin/env python3
"""
Evaluate the teacher model as an explanation verifier.

We use the same test split that was used for verifier finetuning
(`explanations_dataset_test.jsonl`), and ask the teacher to classify
each (problem, explanation) pair as whether it would lead the student
to the CORRECT final answer. We then compare the teacher's predictions
against the ground-truth `student_correct` label.

Intermediate results (teacher output per example, predicted/gold label, correct flag)
are written to a checkpoint JSONL. The script is resumable: re-run to continue
from the last checkpoint.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
from munch import munchify
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CONFIG_PATH = ROOT_DIR / "config.yaml"
FINETUNE_CONFIG_PATH = BASE_DIR / "finetune_config.yaml"


def _torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _primary_device(model) -> torch.device:
    return next(model.parameters()).device


def load_teacher_model_and_tokenizer(model_id: str, token: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(),
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    logger.info(f"Loaded teacher model {model_id} on {_primary_device(model)}")
    return model, tok


def load_test_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found at {path}")
    ds = load_dataset("json", data_files=str(path), split="train")
    if "student_correct" not in ds.column_names:
        raise KeyError(f"'student_correct' field missing. Columns: {ds.column_names}")
    if "is_malformed" in ds.column_names:
        ds = ds.filter(lambda x: not x["is_malformed"])
    logger.info(f"Loaded test dataset with {len(ds)} examples from {path}")
    return ds


def append_checkpoint_row(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(path: Path) -> Tuple[Dict[int, Dict], int]:
    """
    Load existing checkpoint JSONL. Returns (results_by_idx, num_correct).
    Each row has keys: idx, teacher_output, predicted_label, gold_label, correct.
    """
    results_by_idx: Dict[int, Dict] = {}
    num_correct = 0
    if not path.exists():
        return results_by_idx, num_correct
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("idx")
            if idx is None:
                continue
            results_by_idx[idx] = row
            if row.get("correct", False):
                num_correct += 1
    return results_by_idx, num_correct


VERIFIER_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert judge of mathematical explanations.

You will be given:
1) A math problem.
2) A short explanation that a student used while solving the problem.

Your task is to decide whether this explanation is good enough that a student
following it carefully would arrive at the true final numerical answer.

Respond with exactly one word on the first line:
either "CORRECT" or "INCORRECT".

Do NOT show your reasoning, and do NOT restate the problem or explanation.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Student explanation:
{explanation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


_LABEL_TOKEN_IDS: Tuple[int, int] | None = None  # (incorrect_id, correct_id)


def _get_label_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Get token ids for the first token of ' INCORRECT' and ' CORRECT'.
    Returns (incorrect_id, correct_id).
    """
    global _LABEL_TOKEN_IDS
    if _LABEL_TOKEN_IDS is not None:
        return _LABEL_TOKEN_IDS
    incorrect_ids = tokenizer.encode(" INCORRECT", add_special_tokens=False)
    correct_ids = tokenizer.encode(" CORRECT", add_special_tokens=False)
    if not incorrect_ids or not correct_ids:
        raise ValueError("Failed to tokenize label words 'INCORRECT'/'CORRECT'.")
    _LABEL_TOKEN_IDS = (incorrect_ids[0], correct_ids[0])
    return _LABEL_TOKEN_IDS


@torch.inference_mode()
def teacher_predict_batch(
    model, tokenizer, batch: List[Dict], max_new_tokens: int = 8
) -> Tuple[List[int], List[str]]:
    """
    Given a batch (list of dicts with 'problem' and 'explanation'), return
    (preds, raw_outputs): preds are 1 for CORRECT, 0 for INCORRECT;
    raw_outputs are the decoded model response strings.
    """
    prompts = [
        VERIFIER_PROMPT_TEMPLATE.format(
            problem=ex["problem"],
            explanation=ex["explanation"],
        )
        for ex in batch
    ]
    device = _primary_device(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    preds: List[int] = []
    raw_outputs: List[str] = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw_outputs.append(text)
        text_lower = text.lower()
        first_line = text_lower.splitlines()[0] if text_lower else ""
        if "incorrect" in first_line:
            preds.append(0)
        elif "correct" in first_line:
            preds.append(1)
        else:
            preds.append(0)
    return preds, raw_outputs


@torch.inference_mode()
def compute_batch_loss(
    model, tokenizer, batch: List[Dict], gold_labels: List[int]
) -> float:
    """
    Compute total cross-entropy loss over this batch, treating the next token
    after the prompt as a 2-way classification over {INCORRECT, CORRECT}.
    """
    prompts = [
        VERIFIER_PROMPT_TEMPLATE.format(
            problem=ex["problem"],
            explanation=ex["explanation"],
        )
        for ex in batch
    ]
    device = _primary_device(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)
    logits = outputs.logits  # (B, L, V)
    attn = inputs["attention_mask"]
    last_idx = attn.sum(dim=1) - 1  # (B,) index of last prompt token
    next_logits = logits[torch.arange(len(batch), device=device), last_idx]  # (B, V)

    incorrect_id, correct_id = _get_label_token_ids(tokenizer)
    # two-class logits: [:, 0] = INCORRECT, [:, 1] = CORRECT
    two_logits = torch.stack(
        [next_logits[:, incorrect_id], next_logits[:, correct_id]], dim=-1
    )  # (B, 2)
    gold = torch.tensor(gold_labels, device=device, dtype=torch.long)
    # Sum loss over batch; caller will normalize by total count
    loss = F.cross_entropy(two_logits, gold, reduction="sum")
    return float(loss.item())


def evaluate_teacher_as_verifier(
    checkpoint_path: Path | None = None,
    batch_size: int = 8,
) -> None:
    load_dotenv(ROOT_DIR / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not set; gated models may fail to download."
        )

    with CONFIG_PATH.open("r") as f:
        main_cfg = munchify(yaml.safe_load(f))
    teacher_model_id = main_cfg.teacher_model.id
    # Use the labeled explanations dataset produced by the pipeline
    test_path = ROOT_DIR / "verifier_dataset_generation" / "labeled_datasets" / "explanations_dataset.jsonl"

    if checkpoint_path is None:
        checkpoint_path = BASE_DIR / "teacher_as_verifier_checkpoint.jsonl"
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset for teacher-as-verifier baseline...")
    ds = load_test_dataset(test_path)

    results_by_idx, _ = load_checkpoint(checkpoint_path)
    completed_ids = set(results_by_idx.keys())
    # Pending: list of (dataset_index, example) for examples not yet in checkpoint
    pending: List[Tuple[int, Dict]] = []
    for i in range(len(ds)):
        ex = ds[i]
        idx = ex.get("idx", i)
        if idx not in completed_ids:
            pending.append((i, ex))

    total_loss = 0.0

    if pending:
        logger.info(
            f"Resuming: {len(completed_ids)} already in checkpoint, {len(pending)} remaining."
        )
        logger.info(f"Loading teacher model {teacher_model_id} for verifier baseline...")
        model, tokenizer = load_teacher_model_and_tokenizer(teacher_model_id, token)

        for start in tqdm(
            range(0, len(pending), batch_size), desc="Teacher-as-verifier eval"
        ):
            batch_data = pending[start : start + batch_size]
            batch = [ex for _, ex in batch_data]
            preds, raw_outputs = teacher_predict_batch(model, tokenizer, batch)
            gold_labels = [1 if ex["student_correct"] else 0 for ex in batch]
            batch_loss = compute_batch_loss(model, tokenizer, batch, gold_labels)
            total_loss += batch_loss

            for (ds_idx, ex), pred, raw, gold in zip(
                batch_data, preds, raw_outputs, gold_labels
            ):
                idx = ex.get("idx", ds_idx)
                correct = pred == gold
                row = {
                    "idx": idx,
                    "teacher_output": raw,
                    "predicted_label": pred,
                    "gold_label": gold,
                    "correct": correct,
                }
                results_by_idx[idx] = row
                append_checkpoint_row(checkpoint_path, row)
    else:
        logger.info("Checkpoint complete; no remaining examples to process.")

    total = len(results_by_idx)
    num_correct = sum(1 for r in results_by_idx.values() if r.get("correct", False))
    accuracy = num_correct / total if total > 0 else 0.0
    test_loss = total_loss / total if total > 0 else float("nan")

    logger.info(
        f"Teacher-as-verifier accuracy: {accuracy:.4f} ({num_correct}/{total}), "
        f"loss: {test_loss:.4f}"
    )
    logger.info(f"Checkpoint (per-example results): {checkpoint_path}")
    print(
        f"Teacher-as-verifier accuracy: {accuracy:.4f} ({num_correct}/{total}), "
        f"loss: {test_loss:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")

    summary_path = checkpoint_path.with_suffix(".summary.json")
    summary = {
        "accuracy": accuracy,
        "loss": test_loss,
        "num_correct": num_correct,
        "total": total,
        "checkpoint_path": str(checkpoint_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate teacher model as explanation verifier (resumable)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint JSONL (default: finetuning/teacher_as_verifier_checkpoint.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    evaluate_teacher_as_verifier(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )

