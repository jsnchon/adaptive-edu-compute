#!/usr/bin/env python3
"""
Measure performance of frozen Phi-3-mini-4k-instruct on Hendrycks MATH (EleutherAI/hendrycks_math).
Uses only the problem text for inference; solutions are used only for scoring.
Designed to run on AWS/GCP (GPU recommended).
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# --- MATH answer extraction (no evaluation info used during generation) ---

def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} content from model or reference solution."""
    if not text or not isinstance(text, str):
        return None
    # Match \boxed{...} with nested braces allowed (take last occurrence)
    matches = list(re.finditer(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_answer(s: str) -> str:
    """Normalize for comparison: strip, collapse whitespace, optional case/frac handling."""
    if not s:
        return ""
    s = " ".join(s.split()).strip()
    # Common MATH normalization: remove trailing punctuation from numbers
    s = re.sub(r"\.$", "", s)
    return s


def answers_match(pred: str | None, ref: str | None) -> bool:
    """Compare predicted and reference answers after normalization."""
    if pred is None or ref is None:
        return False
    return normalize_answer(pred) == normalize_answer(ref)


# --- Model and dataset (Hugging Face) ---

def get_model_and_tokenizer(model_id: str, device: str | None = None):
    """Load Phi-3 (or other HF causal LM) and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    return model, tokenizer, device


def build_prompt(problem: str, tokenizer) -> str:
    """Phi-3 chat format: only problem as user message (no solution)."""
    # Phi-3 chat format per model card
    return (
        "<|system|>\nYou are a helpful assistant. Solve the following math problem and put your final answer in \\boxed{}.<|end|>\n"
        "<|user|>\n" + problem.strip() + "<|end|>\n"
        "<|assistant|>\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Run frozen Phi-3 on Hendrycks MATH.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Hugging Face model id")
    parser.add_argument("--dataset", default="EleutherAI/hendrycks_math", help="Hugging Face dataset id")
    parser.add_argument("--config", type=str, default=None, help="Dataset config (e.g. algebra, prealgebra); omit for default")
    parser.add_argument("--split", default="test", help="Dataset split (e.g. test, validation)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of problems (default: all)")
    parser.add_argument("--output", type=str, default="math_eval_results.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per reply")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (1 is safest for long sequences)")
    args = parser.parse_args()

    # Load dataset (question set only; we use solution only for scoring)
    print("Loading dataset...")
    try:
        if args.config:
            ds = load_dataset(args.dataset, args.config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except Exception:
        ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    if "problem" not in ds.column_names or "solution" not in ds.column_names:
        raise KeyError(f"Dataset must have 'problem' and 'solution' columns. Got: {ds.column_names}")

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    n_total = len(ds)
    print(f"Evaluating on {n_total} problems.")

    # Load model
    print("Loading model...")
    model, tokenizer, device = get_model_and_tokenizer(args.model, args.device)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    results = []
    correct = 0
    ref_answers = []

    for i in tqdm(range(n_total), desc="Inference"):
        row = ds[i]
        problem = row["problem"]
        reference_solution = row["solution"]
        ref_boxed = extract_boxed_answer(reference_solution)
        ref_answers.append(ref_boxed)

        prompt = build_prompt(problem, tokenizer)
        out = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        generated = out[0]["generated_text"] if out else ""

        pred_boxed = extract_boxed_answer(generated)
        is_correct = answers_match(pred_boxed, ref_boxed)
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "predicted_answer": pred_boxed,
            "reference_answer": ref_boxed,
            "correct": is_correct,
            "generated_text": generated[:2000],  # truncate for file size
        })

    accuracy = correct / n_total if n_total else 0.0
    metrics = {"accuracy": accuracy, "correct": correct, "total": n_total}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    print(f"Accuracy: {correct}/{n_total} = {accuracy:.4f}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
