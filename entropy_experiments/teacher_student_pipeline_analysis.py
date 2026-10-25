# NOTE: before running this script, ensure HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is set in the environment

import json
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml
from dotenv import load_dotenv
from munch import munchify
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from loguru import logger
from tqdm import tqdm

ROOT_DIR =  Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DATASET_REPO = "EleutherAI/hendrycks_math"
SHUFFLE_SEED = 42
DEFAULT_VANILLA_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in mathematics and problem solving. Below, the user has asked you a math problem.
Reason through it carefully and step by step.

The final part of your response should be the numerical answer to the problem included in \\boxed{{}}.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: {problem}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
teacher_prompt_template = ""

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
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    logger.info(f"Loaded {model_id} on {_primary_device(model)}")
    return model, tok

# latex allows infinite nesting so we can't use regex for it (holy CS theory)
def extract_boxed_expr(text):
    start_str = "\\boxed{"
    start_idx = text.rfind(start_str)
    if start_idx == -1:
        return None

    content_start = start_idx + len(start_str)

    stack = 1
    content_end = content_start

    for i in range(content_start, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1

        if stack == 0:
            content_end = i
            break
    else:
        return None

    return text[content_start:content_end].replace(" ", "")

def model_is_correct(response, solution):
    response_ans = extract_boxed_expr(response)
    solution_ans = extract_boxed_expr(solution)
    if response_ans is None or solution_ans is None:
        return False
    else:
        return response_ans == solution_ans

def token_entropies_from_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    model_device = _primary_device(model)
    if model_device.type != "cuda" and torch.cuda.is_available():
        logger.warning(f"Model primary device is {model_device} (expected cuda).")

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

    # If nothing generated, return empty entropy list
    if len(out.scores) == 0:
        return text, gen_ids.tolist(), []

    entropies = []
    for step_logits in out.scores:
        logits = step_logits[0].float()    
        probs = F.softmax(logits, dim=-1)   
        H = torch.special.entr(probs).sum().item()  # nats, safe for zeros
        entropies.append(H)

    entropies = entropies[:kept_token_count]
    gen_ids = gen_ids[:kept_token_count]

    return text, gen_ids.tolist(), entropies

@torch.inference_mode()
def token_entropies_from_generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[tuple[str, list[int], list[float]]]:
    if not prompts:
        return []

    model_device = _primary_device(model)
    if model_device.type != "cuda" and torch.cuda.is_available():
        logger.warning(f"Model primary device is {model_device} (expected cuda).")

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)
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

    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    all_gen_ids = out.sequences
    all_texts: list[str] = []
    all_ids_lists: list[list[int]] = []
    all_kept_counts: list[int] = []

    for sample_idx, prompt_len in enumerate(prompt_lens):
        gen_ids = all_gen_ids[sample_idx][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        kept_token_count = len(tokenizer.encode(text, add_special_tokens=False))
        all_texts.append(text)
        all_ids_lists.append(gen_ids.tolist()[:kept_token_count])
        all_kept_counts.append(kept_token_count)

    # If nothing generated, return empty entropy lists.
    if len(out.scores) == 0:
        return [(all_texts[i], all_ids_lists[i], []) for i in range(len(prompts))]

    entropies_by_sample: list[list[float]] = [[] for _ in prompts]
    for step_logits in out.scores:
        logits = step_logits.float()
        probs = F.softmax(logits, dim=-1)
        entropies = torch.special.entr(probs).sum(dim=-1).tolist()
        for sample_idx, entropy in enumerate(entropies):
            entropies_by_sample[sample_idx].append(entropy)

    results = []
    for i in range(len(prompts)):
        results.append((all_texts[i], all_ids_lists[i], entropies_by_sample[i][:all_kept_counts[i]]))
    return results

def extract_teacher_explanation(teacher_output):
    explanation_pattern = r"<explain>(.*?)<\/explain>"
    matches = re.findall(explanation_pattern, teacher_output, re.DOTALL)
    if matches:
        explanation = matches[-1].strip()  # last match = model's output (prompt is first)
        template_match = re.search(explanation_pattern, teacher_prompt_template, re.DOTALL) if teacher_prompt_template else None
        if template_match is None:
            return explanation
        template_explanation = template_match.group(1).strip()
        if explanation != template_explanation:
            return explanation
    # explain_match = re.search(r'<explain>(.*?)<\/explain>', teacher_output, re.DOTALL)
    # if explain_match:
    #     explanation = explain_match.group(1).strip()
        #if explanation:
            #return explanation

    # Fallback: use full teacher output without final boxed answer.
    boxed_start = teacher_output.rfind("\\boxed{")
    if boxed_start != -1:
        fallback = teacher_output[:boxed_start].strip()
    else:
        fallback = teacher_output.strip()

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

def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)

def explanation_entropy_slice(
    teacher_output: str,
    token_entropies_nats: list[float],
    tokenizer,
) -> dict[str, Any]:
    if not token_entropies_nats:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    explain_match = re.search(r"<explain>(.*?)</explain>", teacher_output, re.DOTALL)
    if not explain_match:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    before_text = teacher_output[:explain_match.start(1)]
    explanation_text = explain_match.group(1)

    before_tokens = len(tokenizer.encode(before_text, add_special_tokens=False))
    explanation_tokens = len(tokenizer.encode(explanation_text, add_special_tokens=False))
    if explanation_tokens == 0:
        return {"num_tokens": 0, "mean_entropy_nats": None, "median_entropy_nats": None, "p90_entropy_nats": None}

    explanation_entropies = token_entropies_nats[before_tokens:before_tokens + explanation_tokens]
    return entropy_stats(explanation_entropies)

def sorted_rows(rows_by_idx: dict[int, dict]) -> list[dict]:
    return [rows_by_idx[i] for i in sorted(rows_by_idx)]

def atomic_write_json(path: Path, data) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_rows_by_idx(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    rows_by_idx: dict[int, dict] = {}
    if path.suffix == ".jsonl":
        with path.open("r") as f:
            for fallback_idx, line in enumerate(f, start=0):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Ignoring malformed checkpoint line in {path}")
                    continue
                idx = row.get("idx", fallback_idx)
                rows_by_idx[idx] = row
        return rows_by_idx

    with path.open("r") as f:
        raw = json.load(f)

    for fallback_idx, row in enumerate(raw, start=0):
        idx = row.get("idx", fallback_idx)
        rows_by_idx[idx] = row
    return rows_by_idx

def load_stage_rows(checkpoint_path: Path, results_path: Path) -> dict[int, dict]:
    rows_by_idx = load_rows_by_idx(checkpoint_path)
    if rows_by_idx:
        return rows_by_idx

    rows_by_idx = load_rows_by_idx(results_path)
    if rows_by_idx:
        for row in sorted_rows(rows_by_idx):
            append_jsonl(checkpoint_path, row)
    return rows_by_idx

def materialize_explanations(path: Path, rows_by_idx: dict[int, dict]) -> None:
    explanation_rows = [
        {
            "idx": idx,
            "problem": row["problem"],
            "explanation": row["explanation"],
            "real_solution": row["real_solution"],
            "student_correct": row["student_correct"],
        }
        for idx, row in sorted(rows_by_idx.items())
    ]
    atomic_write_json(path, explanation_rows)

def cfg_get(config, dotted_path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur

def cfg_first(config, dotted_paths: list[str], default: Any = None) -> Any:
    for dotted_path in dotted_paths:
        val = cfg_get(config, dotted_path, None)
        if val is not None:
            return val
    return default

def run_signature(
    config,
    split_size: int | None,
    vanilla_prompt_template: str,
    generation_kwargs: dict[str, Any],
    batch_size: int,
) -> dict:
    return {
        "dataset_repo": DATASET_REPO,
        "split_size": split_size,
        "shuffle_seed": SHUFFLE_SEED,
        "teacher_model_id": config.teacher_model.id,
        "student_model_id": config.student_model.id,
        "teacher_prompt_template": config.teacher_model.prompt_template,
        "student_prompt_template": config.student_model.prompt_template,
        "vanilla_prompt_template": vanilla_prompt_template,
        "generation_kwargs": {
            "max_new_tokens": generation_kwargs["max_new_tokens"],
            "temperature": generation_kwargs["temperature"],
            "top_p": generation_kwargs["top_p"],
            "batch_size": batch_size,
        },
    }

def pipeline_value(config, key: str, default: Any) -> Any:
    return cfg_first(config, [f"inference.{key}", f"pipeline.{key}"], default)

def batched_indices(dataset_len: int, existing_rows: dict[int, dict], batch_size: int): #splits into chunks
    pending = [idx for idx in range(dataset_len) if idx not in existing_rows]
    for start in range(0, len(pending), batch_size):
        yield pending[start:start + batch_size]
def ensure_run_manifest(path: Path, signature: dict) -> None:
    if not path.exists():
        atomic_write_json(path, signature)
        return

    with path.open("r") as f:
        existing = json.load(f)
    if existing != signature:
        raise RuntimeError(
            "Existing checkpoints were created with a different config or prompt setup. "
            "Use a new output directory or remove the old checkpoint files before resuming."
        )

load_dotenv(ROOT_DIR / ".env")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    logger.warning("HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not set; gated models may fail to download.")

with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)
    config = munchify(config)
teacher_prompt_template = cfg_first(config, ["teacher_model.prompt_template"], "")

metrics_out_dir = Path(cfg_first(config, ["general.metrics_out_dir"], "pipeline_results"))
metrics_out_dir.mkdir(parents=True, exist_ok=True)

dataset_out_file = cfg_first(config, ["general.dataset_out_file"], None)
if dataset_out_file is not None:
    explanations_path = Path(dataset_out_file)
    explanations_path.parent.mkdir(parents=True, exist_ok=True)
else:
    dataset_out_dir = Path(cfg_first(config, ["general.dataset_out_dir"], "explanations_dataset"))
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    explanations_path = dataset_out_dir / "explanations.json"

dataset_configs = get_dataset_config_names(DATASET_REPO)

all_train_list = [load_dataset(DATASET_REPO, cfg, split="train") for cfg in dataset_configs]
dataset = concatenate_datasets(all_train_list)
split_size = cfg_first(config, ["general.split_size", "pipeline.split_size"], None)
if split_size is not None:
    dataset = dataset.shuffle(seed=SHUFFLE_SEED).select(range(split_size))
logger.info(f"Dataset length: {len(dataset)}")
logger.debug(f"First example of dataset: {dataset[0]}")

teacher_model, teacher_tokenizer = load_model_and_tokenizer(config.teacher_model.id, token=token)
student_model, student_tokenizer = load_model_and_tokenizer(config.student_model.id, token=token)
vanilla_prompt_template = cfg_first(
    config,
    ["general.vanilla_prompt_template"],
    DEFAULT_VANILLA_PROMPT_TEMPLATE,
)

generation_kwargs = {
    "max_new_tokens": int(cfg_first(config, ["inference.max_new_tokens", "pipeline.max_new_tokens"], 1024)),
    "temperature": float(cfg_first(config, ["inference.temperature", "pipeline.temperature"], 0.8)),
    "top_p": float(cfg_first(config, ["inference.top_p", "pipeline.top_p"], 0.9)),
}
batch_size = int(pipeline_value(config, "batch_size", 1))
if batch_size < 1:
    raise ValueError("inference.batch_size (or pipeline.batch_size) must be >= 1")
logger.info(f"Generation settings: {generation_kwargs}")
logger.info(f"Batch size: {batch_size}")

logger.info(f"Saving metrics to {metrics_out_dir}")

manifest_path = metrics_out_dir / "run_manifest.json"
ensure_run_manifest(
    manifest_path,
    run_signature(config, split_size, vanilla_prompt_template, generation_kwargs, batch_size),
)

teacher_metrics_path = metrics_out_dir / "teacher_entropy_results.json"
teacher_checkpoint_path = metrics_out_dir / "teacher_entropy_results.jsonl"
student_noexp_path = metrics_out_dir / "student_entropy_results_no_explanation.json"
student_noexp_checkpoint_path = metrics_out_dir / "student_entropy_results_no_explanation.jsonl"
student_wexp_path = metrics_out_dir / "student_entropy_results_with_explanation.json"
student_wexp_checkpoint_path = metrics_out_dir / "student_entropy_results_with_explanation.jsonl"
vanilla_path = metrics_out_dir / "vanilla_reasoning.json"
vanilla_checkpoint_path = metrics_out_dir / "vanilla_reasoning.jsonl"
summary_metrics_path = metrics_out_dir / "summary_metrics.json"
explanation_entropy_corr_path = metrics_out_dir / "explanation_entropy_student_accuracy.json"
teacher_rows = load_stage_rows(teacher_checkpoint_path, teacher_metrics_path)
total_correct = sum(1 for row in teacher_rows.values() if row.get("teacher_correct"))
teacher_batches = list(batched_indices(len(dataset), teacher_rows, batch_size))
for batch_indices in tqdm(teacher_batches, desc="Teacher + Entropy"):
    examples = [dataset[i] for i in batch_indices]
    prompts = [config.teacher_model.prompt_template.format(problem=ex["problem"]) for ex in examples]
    outputs = token_entropies_from_generate_batch(teacher_model, teacher_tokenizer, prompts, **generation_kwargs)

    for idx, ex, (text, _, H) in zip(batch_indices, examples, outputs):
        problem, solution = ex["problem"], ex["solution"]
        teacher_correct = model_is_correct(text, solution)
        if teacher_correct:
            total_correct += 1

        real_solution = extract_boxed_expr(solution)
        assert real_solution is not None, "Failed to extract ground truth answer from solution"

        row = {
            "idx": idx,
            "problem": problem,
            "teacher_output": text,
            "real_solution": real_solution,
            "teacher_correct": teacher_correct,
            **entropy_stats(H),
            "token_entropies_nats": H,
        }
        teacher_rows[idx] = row
        append_jsonl(teacher_checkpoint_path, row)

atomic_write_json(teacher_metrics_path, sorted_rows(teacher_rows))
logger.info(f"Accuracy: {total_correct / len(dataset)}")

student_noexp_rows = load_stage_rows(student_noexp_checkpoint_path, student_noexp_path)
total_correct = sum(1 for row in student_noexp_rows.values() if row.get("student_correct"))
student_noexp_batches = list(batched_indices(len(dataset), student_noexp_rows, batch_size))
for batch_indices in tqdm(student_noexp_batches, desc="Student w/out explanation + Entropy"):
    examples = [dataset[i] for i in batch_indices]
    prompts = [
        config.student_model.prompt_template.format(problem=ex["problem"], explanation="None")
        for ex in examples
    ]
    outputs = token_entropies_from_generate_batch(student_model, student_tokenizer, prompts, **generation_kwargs)

    for idx, ex, (text, _, H) in zip(batch_indices, examples, outputs):
        problem, solution = ex["problem"], ex["solution"]
        explanation = "None"
        student_correct = model_is_correct(text, solution)
        if student_correct:
            total_correct += 1

        real_solution = extract_boxed_expr(solution)
        assert real_solution is not None, "Failed to extract ground truth answer from solution"

        row = {
            "idx": idx,
            "problem": problem,
            "explanation": explanation,
            "real_solution": real_solution,
            "student_output": text,
            "student_correct": student_correct,
            **entropy_stats(H),
            "token_entropies_nats": H,
        }
        student_noexp_rows[idx] = row
        append_jsonl(student_noexp_checkpoint_path, row)

atomic_write_json(student_noexp_path, sorted_rows(student_noexp_rows))
logger.info(f"Accuracy: {total_correct / len(dataset)}")

student_wexp_rows = load_stage_rows(student_wexp_checkpoint_path, student_wexp_path)
materialize_explanations(explanations_path, student_wexp_rows)
total_correct = sum(1 for row in student_wexp_rows.values() if row.get("student_correct"))
student_wexp_batches = list(batched_indices(len(dataset), student_wexp_rows, batch_size))
for batch_indices in tqdm(student_wexp_batches, desc="Student w/ explanation + Entropy"):
    examples = [dataset[i] for i in batch_indices]
    explanations = [extract_teacher_explanation(teacher_rows[idx]["teacher_output"]) for idx in batch_indices]
    prompts = [
        config.student_model.prompt_template.format(problem=ex["problem"], explanation=explanation)
        for ex, explanation in zip(examples, explanations)
    ]
    outputs = token_entropies_from_generate_batch(student_model, student_tokenizer, prompts, **generation_kwargs)

    for idx, ex, explanation, (text, _, H) in zip(batch_indices, examples, explanations, outputs):
        problem, solution = ex["problem"], ex["solution"]
        student_correct = model_is_correct(text, solution)
        if student_correct:
            total_correct += 1

        real_solution = extract_boxed_expr(solution)
        assert real_solution is not None, "Failed to extract ground truth answer from solution"

        row = {
            "idx": idx,
            "problem": problem,
            "explanation": explanation,
            "real_solution": real_solution,
            "student_output": text,
            "student_correct": student_correct,
            **entropy_stats(H),
            "token_entropies_nats": H,
        }
        student_wexp_rows[idx] = row
        append_jsonl(student_wexp_checkpoint_path, row)

atomic_write_json(student_wexp_path, sorted_rows(student_wexp_rows))
materialize_explanations(explanations_path, student_wexp_rows)
logger.info(f"Accuracy: {total_correct / len(dataset)}")

vanilla_rows = load_stage_rows(vanilla_checkpoint_path, vanilla_path)
teacher_total_correct = sum(1 for row in vanilla_rows.values() if row["big"]["correct"])
student_total_correct = sum(1 for row in vanilla_rows.values() if row["small"]["correct"])
vanilla_batches = list(batched_indices(len(dataset), vanilla_rows, batch_size))
for batch_indices in tqdm(vanilla_batches, desc="Vanilla reasoning evaluation"):
    examples = [dataset[i] for i in batch_indices]
    prompts = [vanilla_prompt_template.format(problem=ex["problem"]) for ex in examples]
    teacher_outputs = token_entropies_from_generate_batch(teacher_model, teacher_tokenizer, prompts, **generation_kwargs)
    student_outputs = token_entropies_from_generate_batch(student_model, student_tokenizer, prompts, **generation_kwargs)

    for idx, ex, teacher_output, student_output in zip(batch_indices, examples, teacher_outputs, student_outputs):
        problem, solution = ex["problem"], ex["solution"]
        teacher_response = teacher_output[0]
        student_response = student_output[0]

        teacher_correct = model_is_correct(teacher_response, solution)
        if teacher_correct:
            teacher_total_correct += 1

        student_correct = model_is_correct(student_response, solution)
        if student_correct:
            student_total_correct += 1

        row = {
            "idx": idx,
            "problem": problem,
            "big": {
                "output": teacher_response,
                "correct": teacher_correct,
            },
            "small": {
                "output": student_response,
                "correct": student_correct,
            },
        }
        vanilla_rows[idx] = row
        append_jsonl(vanilla_checkpoint_path, row)

atomic_write_json(vanilla_path, sorted_rows(vanilla_rows))
logger.info(f"Teacher model accuracy: {teacher_total_correct / len(dataset)}")
logger.info(f"Student model accuracy: {student_total_correct / len(dataset)}")

teacher_accuracy = sum(1 for row in teacher_rows.values() if row.get("teacher_correct")) / len(dataset)
student_noexp_accuracy = sum(1 for row in student_noexp_rows.values() if row.get("student_correct")) / len(dataset)
student_wexp_accuracy = sum(1 for row in student_wexp_rows.values() if row.get("student_correct")) / len(dataset)
vanilla_teacher_accuracy = sum(1 for row in vanilla_rows.values() if row["big"]["correct"]) / len(dataset)
vanilla_student_accuracy = sum(1 for row in vanilla_rows.values() if row["small"]["correct"]) / len(dataset)

summary_metrics = {
    "dataset_size": len(dataset),
    "accuracies": {
        "teacher_entropy_stage": teacher_accuracy,
        "student_no_explanation_stage": student_noexp_accuracy,
        "student_with_explanation_stage": student_wexp_accuracy,
        "vanilla_teacher": vanilla_teacher_accuracy,
        "vanilla_student": vanilla_student_accuracy,
    },
    "counts": {
        "teacher_correct": sum(1 for row in teacher_rows.values() if row.get("teacher_correct")),
        "student_no_explanation_correct": sum(1 for row in student_noexp_rows.values() if row.get("student_correct")),
        "student_with_explanation_correct": sum(1 for row in student_wexp_rows.values() if row.get("student_correct")),
        "vanilla_teacher_correct": sum(1 for row in vanilla_rows.values() if row["big"]["correct"]),
        "vanilla_student_correct": sum(1 for row in vanilla_rows.values() if row["small"]["correct"]),
    },
}
atomic_write_json(summary_metrics_path, summary_metrics)
logger.info(f"Wrote summary metrics to {summary_metrics_path}")

corr_rows = []
xs: list[float] = []
ys: list[float] = []
for idx in sorted(student_wexp_rows):
    if idx not in teacher_rows:
        continue
    teacher_row = teacher_rows[idx]
    student_row = student_wexp_rows[idx]
    exp_entropy = explanation_entropy_slice(
        teacher_row["teacher_output"],
        teacher_row.get("token_entropies_nats", []),
        teacher_tokenizer,
    )
    row = {
        "idx": idx,
        "student_correct": bool(student_row.get("student_correct", False)),
        "explanation_entropy": exp_entropy,
    }
    corr_rows.append(row)
    mean_entropy = exp_entropy["mean_entropy_nats"]
    if mean_entropy is not None:
        xs.append(mean_entropy)
        ys.append(1.0 if student_row.get("student_correct", False) else 0.0)

correlation_payload = {
    "num_points_with_explanation_entropy": len(xs),
    "correlations": {
        "mean_entropy_nats": pearson_corr(
            [row["explanation_entropy"]["mean_entropy_nats"] for row in corr_rows if row["explanation_entropy"]["mean_entropy_nats"] is not None],
            [1.0 if row["student_correct"] else 0.0 for row in corr_rows if row["explanation_entropy"]["mean_entropy_nats"] is not None],
        ),
        "median_entropy_nats": pearson_corr(
            [row["explanation_entropy"]["median_entropy_nats"] for row in corr_rows if row["explanation_entropy"]["median_entropy_nats"] is not None],
            [1.0 if row["student_correct"] else 0.0 for row in corr_rows if row["explanation_entropy"]["median_entropy_nats"] is not None],
        ),
        "p90_entropy_nats": pearson_corr(
            [row["explanation_entropy"]["p90_entropy_nats"] for row in corr_rows if row["explanation_entropy"]["p90_entropy_nats"] is not None],
            [1.0 if row["student_correct"] else 0.0 for row in corr_rows if row["explanation_entropy"]["p90_entropy_nats"] is not None],
        ),
        "num_tokens": pearson_corr(
            [float(row["explanation_entropy"]["num_tokens"]) for row in corr_rows if row["explanation_entropy"]["num_tokens"] > 0],
            [1.0 if row["student_correct"] else 0.0 for row in corr_rows if row["explanation_entropy"]["num_tokens"] > 0],
        ),
    },
    "rows": corr_rows,
}
atomic_write_json(explanation_entropy_corr_path, correlation_payload)
logger.info(f"Wrote explanation-entropy correlation data to {explanation_entropy_corr_path}")
