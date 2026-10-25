import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_REPO = "EleutherAI/hendrycks_math"
SHUFFLE_SEED = 42


def _torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _primary_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(model_id: str, token: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(),
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info(f"Loaded {model_id} on {_primary_device(model)}")
    return model, tokenizer


@torch.inference_mode()
def token_entropies_from_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    model_device = _primary_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    kept_token_count = len(tokenizer.encode(text, add_special_tokens=False))

    if len(out.scores) == 0:
        return text, []

    entropies = []
    for step_logits in out.scores:
        logits = step_logits[0].float()
        probs = F.softmax(logits, dim=-1)
        entropies.append(torch.special.entr(probs).sum().item())
    return text, entropies[:kept_token_count]


def extract_boxed_expr(text: str):
    start_str = "\\boxed{"
    start_idx = text.rfind(start_str)
    if start_idx == -1:
        return None
    content_start = start_idx + len(start_str)
    stack = 1
    content_end = content_start
    for i in range(content_start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
        if stack == 0:
            content_end = i
            break
    else:
        return None
    return text[content_start:content_end].replace(" ", "")


def model_is_correct(response: str, solution: str) -> bool:
    response_ans = extract_boxed_expr(response)
    solution_ans = extract_boxed_expr(solution)
    if response_ans is None or solution_ans is None:
        return False
    return response_ans == solution_ans


def extract_teacher_explanation(teacher_output: str) -> str:
    explain_match = re.search(r"<explain>(.*?)</explain>", teacher_output, re.DOTALL)
    if explain_match:
        explanation = explain_match.group(1).strip()
        if explanation:
            return explanation
    boxed_start = teacher_output.rfind("\\boxed{")
    fallback = teacher_output[:boxed_start].strip() if boxed_start != -1 else teacher_output.strip()
    return fallback if fallback else "None"


def entropy_stats(entropies: list[float]):
    if not entropies:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}
    hs = sorted(entropies)
    n = len(entropies)
    return {
        "num_tokens": n,
        "mean_entropy_nats": sum(entropies) / n,
        "median_entropy_nats": hs[n // 2],
        "p90_entropy_nats": hs[int(0.9 * (n - 1))],
    }


def explanation_entropy_slice_with_fallback(
    teacher_output: str,
    token_entropies_nats: list[float],
    tokenizer,
):
    if not token_entropies_nats:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    explain_match = re.search(r"<explain>(.*?)</explain>", teacher_output, re.DOTALL)
    if explain_match:
        before_text = teacher_output[:explain_match.start(1)]
        explanation_text = explain_match.group(1)
    else:
        boxed_start = teacher_output.rfind("\\boxed{")
        before_text = ""
        explanation_text = teacher_output[:boxed_start] if boxed_start != -1 else teacher_output

    explanation_text = explanation_text.strip()
    if not explanation_text:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    before_tokens = len(tokenizer.encode(before_text, add_special_tokens=False))
    explanation_tokens = len(tokenizer.encode(explanation_text, add_special_tokens=False))
    if explanation_tokens == 0:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    explanation_entropies = token_entropies_nats[before_tokens : before_tokens + explanation_tokens]
    return entropy_stats(explanation_entropies)


def auto_choose_threshold(corr_json_path: Path, entropy_key: str, threshold_source: str) -> float:
    if not corr_json_path.exists():
        logger.warning(f"{corr_json_path} not found; using fallback threshold 0.22")
        return 0.22
    payload = json.loads(corr_json_path.read_text())
    rows = payload.get("rows", [])
    correct_vals = []
    wrong_vals = []
    all_vals = []
    for row in rows:
        entropy_obj = row.get("explanation_entropy", {})
        value = entropy_obj.get(entropy_key)
        if value is None:
            continue
        all_vals.append(float(value))
        if row.get("student_correct"):
            correct_vals.append(float(value))
        else:
            wrong_vals.append(float(value))

    if threshold_source == "wrong_median" and wrong_vals:
        wrong_vals.sort()
        chosen = wrong_vals[len(wrong_vals) // 2]
        logger.info(f"Auto threshold from wrong median: {chosen:.4f}")
        return chosen
    if threshold_source == "wrong_max" and wrong_vals:
        chosen = max(wrong_vals)
        logger.info(f"Auto threshold from wrong max: {chosen:.4f}")
        return chosen
    if threshold_source == "correct_median" and correct_vals:
        correct_vals.sort()
        chosen = correct_vals[len(correct_vals) // 2]
        logger.info(f"Auto threshold from correct median: {chosen:.4f}")
        return chosen

    if correct_vals and wrong_vals:
        correct_vals.sort()
        wrong_vals.sort()
        c_med = correct_vals[len(correct_vals) // 2]
        w_med = wrong_vals[len(wrong_vals) // 2]
        chosen = 0.5 * (c_med + w_med)
        logger.info(f"Auto threshold from medians: correct={c_med:.4f}, wrong={w_med:.4f}, chosen={chosen:.4f}")
        return chosen

    if all_vals:
        all_vals.sort()
        idx = int(0.4 * (len(all_vals) - 1))
        chosen = all_vals[idx]
        logger.info(f"Auto threshold from p40 of historical entropy: {chosen:.4f}")
        return chosen

    logger.warning("No usable entropy rows in correlation file; using fallback threshold 0.22")
    return 0.22


def threshold_passes(metric_value: float | None, threshold: float, threshold_mode: str) -> bool:
    if metric_value is None:
        return False
    return metric_value <= threshold if threshold_mode == "low" else metric_value >= threshold


def cfg_get(config: dict, dotted_path: str, default=None):
    cur = config
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return cur


def cfg_first(config: dict, dotted_paths: list[str], default=None):
    for dotted_path in dotted_paths:
        val = cfg_get(config, dotted_path, None)
        if val is not None:
            return val
    return default


def build_transition_analysis(rows: list[dict]) -> dict:
    buckets = {
        "baseline_right_gated_right": [],
        "baseline_right_gated_wrong": [],
        "baseline_wrong_gated_wrong": [],
        "baseline_wrong_gated_right": [],
    }
    teacher_transition_stats = {
        "baseline_right_gated_wrong": {"teacher_correct": 0, "teacher_wrong_or_missing": 0, "total": 0},
        "baseline_wrong_gated_right": {"teacher_correct": 0, "teacher_wrong_or_missing": 0, "total": 0},
    }

    for row in rows:
        idx = row["idx"]
        baseline_correct = bool(row["baseline_no_explanation"]["student_correct"])
        gated_correct = bool(row["student_with_gated_explanation"]["student_correct"])
        teacher_correct = bool(row.get("selected_teacher_correct", False))

        if baseline_correct and gated_correct:
            key = "baseline_right_gated_right"
        elif baseline_correct and not gated_correct:
            key = "baseline_right_gated_wrong"
        elif not baseline_correct and not gated_correct:
            key = "baseline_wrong_gated_wrong"
        else:
            key = "baseline_wrong_gated_right"

        buckets[key].append(idx)
        if key in teacher_transition_stats:
            teacher_transition_stats[key]["total"] += 1
            if teacher_correct:
                teacher_transition_stats[key]["teacher_correct"] += 1
            else:
                teacher_transition_stats[key]["teacher_wrong_or_missing"] += 1

    n = len(rows) if rows else 1
    return {
        "summary_counts": {**{k: len(v) for k, v in buckets.items()}, "total_examples": len(rows)},
        "rates": {k: len(v) / n for k, v in buckets.items()},
        "teacher_correctness_in_key_transitions": teacher_transition_stats,
        "bucket_indices": buckets,
    }


def build_teacher_correct_only_analysis(rows: list[dict]) -> dict:
    filtered_rows = [row for row in rows if row.get("selected_teacher_correct", False)]
    n = len(filtered_rows)
    if n == 0:
        return {
            "num_filtered_examples": 0,
            "summary": {
                "baseline_student_accuracy_no_explanation": 0.0,
                "entropy_gated_student_accuracy": 0.0,
                "accuracy_improvement_absolute": 0.0,
                "accuracy_improvement_relative": None,
                "baseline_right_gated_wrong_count": 0,
                "baseline_wrong_gated_right_count": 0,
            },
            "indices": [],
        }

    baseline_acc = sum(1 for r in filtered_rows if r["baseline_no_explanation"]["student_correct"]) / n
    gated_acc = sum(1 for r in filtered_rows if r["student_with_gated_explanation"]["student_correct"]) / n
    baseline_right_gated_wrong = sum(
        1
        for r in filtered_rows
        if r["baseline_no_explanation"]["student_correct"] and not r["student_with_gated_explanation"]["student_correct"]
    )
    baseline_wrong_gated_right = sum(
        1
        for r in filtered_rows
        if not r["baseline_no_explanation"]["student_correct"] and r["student_with_gated_explanation"]["student_correct"]
    )

    return {
        "num_filtered_examples": n,
        "summary": {
            "baseline_student_accuracy_no_explanation": baseline_acc,
            "entropy_gated_student_accuracy": gated_acc,
            "accuracy_improvement_absolute": gated_acc - baseline_acc,
            "accuracy_improvement_relative": None if baseline_acc == 0 else (gated_acc - baseline_acc) / baseline_acc,
            "baseline_right_gated_wrong_count": baseline_right_gated_wrong,
            "baseline_wrong_gated_right_count": baseline_wrong_gated_right,
        },
        "indices": [r["idx"] for r in filtered_rows],
    }


def main():
    parser = argparse.ArgumentParser(description="Entropy-thresholded explanation retry experiment.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--max_passes", type=int, default=5)
    parser.add_argument("--threshold", type=str, default="auto", help="Float entropy threshold or 'auto'")
    parser.add_argument(
        "--entropy_key",
        type=str,
        default="mean_entropy_nats",
        choices=["mean_entropy_nats", "median_entropy_nats", "p90_entropy_nats"],
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="low",
        choices=["low", "high"],
        help="'low' accepts entropy <= threshold, 'high' accepts entropy >= threshold",
    )
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="midpoint",
        choices=["midpoint", "correct_median", "wrong_median", "wrong_max"],
        help="Auto-threshold source used only when --threshold auto",
    )
    parser.add_argument("--output", type=Path, default=Path("pipeline_results/entropy_retry_experiment.json"))
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output snapshot if present and incomplete.",
    )
    args = parser.parse_args()

    if args.max_passes < 1:
        raise ValueError("--max_passes must be >= 1")
    if args.num_examples < 1:
        raise ValueError("--num_examples must be >= 1")

    load_dotenv(Path(".env"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    cfg = yaml.safe_load(args.config.read_text())
    teacher_model_id = cfg["teacher_model"]["id"]
    student_model_id = cfg["student_model"]["id"]
    teacher_prompt_template = cfg["teacher_model"]["prompt_template"]
    student_prompt_template = cfg["student_model"]["prompt_template"]
    generation_kwargs = {
        "max_new_tokens": int(
            cfg_first(cfg, ["inference.max_new_tokens", "pipeline.max_new_tokens"], 1024)
            if args.max_new_tokens is None
            else args.max_new_tokens
        ),
        "temperature": float(
            cfg_first(cfg, ["inference.temperature", "pipeline.temperature"], 0.8)
            if args.temperature is None
            else args.temperature
        ),
        "top_p": float(
            cfg_first(cfg, ["inference.top_p", "pipeline.top_p"], 0.9) if args.top_p is None else args.top_p
        ),
    }

    metrics_out_dir = Path(cfg_first(cfg, ["general.metrics_out_dir"], "pipeline_results"))
    corr_path = metrics_out_dir / "explanation_entropy_student_accuracy.json"
    if args.threshold.lower() == "auto":
        threshold = auto_choose_threshold(corr_path, args.entropy_key, args.threshold_source)
    else:
        threshold = float(args.threshold)
    comparator = "<=" if args.threshold_mode == "low" else ">="
    logger.info(f"Using entropy threshold ({args.entropy_key}) {comparator} {threshold:.4f}")

    dataset_configs = get_dataset_config_names(DATASET_REPO)
    all_train = [load_dataset(DATASET_REPO, c, split="train") for c in dataset_configs]
    dataset = concatenate_datasets(all_train).shuffle(seed=SHUFFLE_SEED)
    if args.num_examples > len(dataset):
        raise ValueError(f"--num_examples {args.num_examples} exceeds dataset size {len(dataset)}")
    dataset = dataset.select(range(args.num_examples))
    logger.info(f"Running experiment on {len(dataset)} examples")

    teacher_model, teacher_tokenizer = load_model_and_tokenizer(teacher_model_id, token)
    student_model, student_tokenizer = load_model_and_tokenizer(student_model_id, token)

    rows = []
    processed_start = 0
    baseline_correct = 0
    gated_correct = 0
    total_passes = 0
    met_threshold_count = 0

    def write_snapshot(processed_count: int, completed: bool) -> None:
        n = len(rows)
        baseline_acc = (baseline_correct / n) if n > 0 else None
        gated_acc = (gated_correct / n) if n > 0 else None
        transition_analysis = build_transition_analysis(rows)
        teacher_correct_only_analysis = build_teacher_correct_only_analysis(rows)
        payload = {
            "status": {
                "completed": completed,
                "processed_examples": processed_count,
                "num_examples_requested": args.num_examples,
            },
            "config": {
                "num_examples": args.num_examples,
                "num_examples_requested": args.num_examples,
                "num_examples_evaluated": n,
                "max_passes": args.max_passes,
                "threshold": threshold,
                "entropy_key": args.entropy_key,
                "threshold_mode": args.threshold_mode,
                "threshold_source": args.threshold_source,
                "generation_kwargs": generation_kwargs,
                "teacher_model_id": teacher_model_id,
                "student_model_id": student_model_id,
            },
            "summary": {
                "avg_passes_used": (total_passes / n) if n > 0 else None,
                "threshold_met_rate": (met_threshold_count / n) if n > 0 else None,
                "baseline_student_accuracy_no_explanation": baseline_acc,
                "entropy_gated_student_accuracy": gated_acc,
                "accuracy_improvement_absolute": (gated_acc - baseline_acc) if n > 0 else None,
                "accuracy_improvement_relative": None
                if (n == 0 or baseline_acc == 0)
                else (gated_acc - baseline_acc) / baseline_acc,
            },
            "transition_analysis": transition_analysis,
            "teacher_correct_only_analysis": teacher_correct_only_analysis,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))

    if args.resume and args.output.exists():
        prior_payload = json.loads(args.output.read_text())
        prior_rows = prior_payload.get("rows", [])
        if len(prior_rows) >= args.num_examples:
            logger.info(f"Existing output already has {len(prior_rows)} rows at {args.output}; nothing to resume.")
            return

        rows = prior_rows
        processed_start = int(prior_payload.get("status", {}).get("processed_examples", len(rows)))
        processed_start = min(max(processed_start, 0), len(dataset))
        baseline_correct = sum(1 for r in rows if r["baseline_no_explanation"]["student_correct"])
        gated_correct = sum(1 for r in rows if r["student_with_gated_explanation"]["student_correct"])
        total_passes = sum(int(r["entropy_gated_explanation"]["passes_used"]) for r in rows)
        met_threshold_count = sum(int(r["entropy_gated_explanation"]["threshold_met"]) for r in rows)
        logger.info(
            f"Resuming from snapshot: processed={processed_start}/{len(dataset)}, "
            f"evaluated_rows={len(rows)}/{args.num_examples}"
        )
    else:
        write_snapshot(processed_count=0, completed=False)

    processed_count = processed_start
    for idx in range(processed_start, len(dataset)):
        if len(rows) >= args.num_examples:
            break
        ex = dataset[idx]
        problem = ex["problem"]
        solution = ex["solution"]
        processed_count = idx + 1

        # Baseline student run without explanation.
        baseline_prompt = student_prompt_template.format(problem=problem, explanation="None")
        baseline_output, _ = token_entropies_from_generate(
            student_model, student_tokenizer, baseline_prompt, **generation_kwargs
        )
        baseline_is_correct = model_is_correct(baseline_output, solution)
        baseline_correct += int(baseline_is_correct)

        chosen = None
        best_candidate = None
        pass_records = []
        for pass_idx in range(1, args.max_passes + 1):
            teacher_prompt = teacher_prompt_template.format(problem=problem)
            teacher_output, teacher_entropies = token_entropies_from_generate(
                teacher_model, teacher_tokenizer, teacher_prompt, **generation_kwargs
            )
            explanation = extract_teacher_explanation(teacher_output)
            explanation_entropy = explanation_entropy_slice_with_fallback(
                teacher_output, teacher_entropies, teacher_tokenizer
            )
            metric_value = explanation_entropy.get(args.entropy_key)
            pass_record = {
                "pass_idx": pass_idx,
                "teacher_output": teacher_output,
                "explanation": explanation,
                "explanation_entropy": explanation_entropy,
            }
            pass_records.append(pass_record)

            comparable_value = float("inf") if metric_value is None else float(metric_value)
            if best_candidate is None or comparable_value < best_candidate["score"]:
                best_candidate = {"score": comparable_value, "record": pass_record}
            if threshold_passes(metric_value, threshold, args.threshold_mode):
                chosen = pass_record
                met_threshold_count += 1
                break

        if chosen is None:
            chosen = best_candidate["record"]

        passes_used = chosen["pass_idx"]
        total_passes += passes_used

        gated_prompt = student_prompt_template.format(problem=problem, explanation=chosen["explanation"])
        gated_output, _ = token_entropies_from_generate(
            student_model, student_tokenizer, gated_prompt, **generation_kwargs
        )
        gated_is_correct = model_is_correct(gated_output, solution)
        gated_correct += int(gated_is_correct)
        selected_teacher_correct = model_is_correct(chosen["teacher_output"], solution)

        rows.append(
            {
                "idx": idx,
                "problem": problem,
                "selected_teacher_correct": selected_teacher_correct,
                "baseline_no_explanation": {
                    "student_output": baseline_output,
                    "student_correct": baseline_is_correct,
                },
                "entropy_gated_explanation": {
                    "threshold_met": threshold_passes(
                        chosen["explanation_entropy"].get(args.entropy_key), threshold, args.threshold_mode
                    ),
                    "passes_used": passes_used,
                    "selected_explanation_entropy": chosen["explanation_entropy"],
                    "selected_explanation": chosen["explanation"],
                    "all_passes": pass_records,
                },
                "student_with_gated_explanation": {
                    "student_output": gated_output,
                    "student_correct": gated_is_correct,
                },
            }
        )
        logger.info(
            f"Example {idx}: passes={passes_used}, baseline_correct={baseline_is_correct}, "
            f"gated_correct={gated_is_correct}"
        )
        write_snapshot(processed_count=processed_count, completed=False)

    n = len(rows)
    if n < args.num_examples:
        write_snapshot(processed_count=processed_count, completed=False)
        raise RuntimeError(
            f"Dataset exhausted after processing {processed_count} examples; "
            f"only {n} rows were collected (requested {args.num_examples})."
        )

    write_snapshot(processed_count=processed_count, completed=True)
    logger.info(f"Wrote results to {args.output}")
    logger.info(
        json.dumps(
            {
                "avg_passes_used": total_passes / n,
                "threshold_met_rate": met_threshold_count / n,
                "baseline_student_accuracy_no_explanation": baseline_correct / n,
                "entropy_gated_student_accuracy": gated_correct / n,
                "accuracy_improvement_absolute": (gated_correct - baseline_correct) / n,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
