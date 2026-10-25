import json
import os
from pathlib import Path
import re
import gc

import yaml
from dotenv import load_dotenv
from munch import munchify
import torch
from transformers import pipeline
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from loguru import logger
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DATASET_REPO = "EleutherAI/hendrycks_math"

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

def extract_teacher_explanation(teacher_output, teacher_prompt_template):
    explanation_pattern = r"<explain>(.*?)<\/explain>"
    matches = re.findall(explanation_pattern, teacher_output, re.DOTALL)
    if matches:
        explanation = matches[-1].strip()  # last match = model's output (prompt is first)
        template_explanation = re.search(explanation_pattern, teacher_prompt_template, re.DOTALL).group(1).strip()
        if explanation != template_explanation:
            return explanation
    return "None"

def count_valid_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                break
            count += 1
    return count

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

load_dotenv(BASE_DIR / ".env")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    logger.warning("HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not set; gated models may fail to download.")

with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)
    config = munchify(config)

dataset_configs = get_dataset_config_names(DATASET_REPO)

all_train_list = [load_dataset(DATASET_REPO, cfg, split="train") for cfg in dataset_configs]
dataset = concatenate_datasets(all_train_list)
if split_size := config.general.split_size:
    dataset = dataset.shuffle(seed=42).select(range(split_size))
logger.info(f"Dataset length: {len(dataset)}")
logger.debug(f"First example of dataset: {dataset[0]}")

generation_kwargs = {
    "do_sample": True,
    "max_new_tokens": config.inference.max_new_tokens,
    "temperature": config.inference.temperature,
    "top_p": config.inference.top_p,
    "return_full_text": False,
}
logger.info(f"Generation settings: {generation_kwargs}")

out_file = BASE_DIR / config.general.dataset_out_file
teacher_cache_file = out_file.with_name(f"{out_file.stem}_teacher_cache.jsonl")
batch_size = config.inference.batch_size
logger.info(f"Saving explanations and quality labels to {out_file}")
logger.info(f"Caching teacher explanations to {teacher_cache_file}")
logger.info(f"Using a batch size of {batch_size}")

def create_pipeline(model_id):
    pipe = pipeline(
        "text-generation", 
        model=model_id,
        token=token,
        dtype=torch.bfloat16,
        device_map="auto",
        batch_size=batch_size,
    )
    pipe.tokenizer.padding_side = "left"  # bruh
    
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe

def unload_pipeline(pipe) -> None:
    if hasattr(pipe, "model"):
        del pipe.model
    if hasattr(pipe, "tokenizer"):
        del pipe.tokenizer
    del pipe
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

teacher_prompt_template = config.teacher_model.prompt_template
teacher_start_idx = count_valid_jsonl_rows(teacher_cache_file)
if teacher_start_idx:
    logger.info(f"Existing teacher explanation cache found. Resuming from index: {teacher_start_idx}")

teacher_pipeline = create_pipeline(config.teacher_model.id)
with teacher_cache_file.open("a", encoding="utf-8") as f:
    remaining_dataset = dataset.select(range(teacher_start_idx, len(dataset)))
    for i in tqdm(range(0, len(remaining_dataset), batch_size), desc="Teacher explanation generation"):
        batch = remaining_dataset.select(range(i, min(i + batch_size, len(remaining_dataset))))
        problems, solutions = zip(*[(ex["problem"], ex["solution"]) for ex in batch])
        teacher_prompts = [teacher_prompt_template.format(problem=problem) for problem in problems]
        teacher_resps = teacher_pipeline(teacher_prompts, **generation_kwargs)

        for j, (resp, problem, solution) in enumerate(zip(teacher_resps, problems, solutions)):
            explanation = extract_teacher_explanation(resp[0]["generated_text"], teacher_prompt_template)
            teacher_row = {
                "id": teacher_start_idx + i + j,
                "problem": problem,
                "explanation": explanation,
                "is_malformed": explanation == "None",
                "teacher_resp": resp,
                "ground_truth_solution": solution,
            }
            f.write(json.dumps(teacher_row) + "\n")
        f.flush()

unload_pipeline(teacher_pipeline)

start_idx = count_valid_jsonl_rows(out_file)
if start_idx:
    logger.info(f"Existing labelled dataset found. Resuming student labelling from index: {start_idx}")

student_pipeline = create_pipeline(config.student_model.id)
malformed_count = 0
processed_count = 0
with out_file.open("a", encoding="utf-8") as f:
    teacher_rows = list(iter_jsonl(teacher_cache_file))[start_idx:]
    for i in tqdm(range(0, len(teacher_rows), batch_size), desc="Explanation labelling"):
        batch_rows = teacher_rows[i:i + batch_size]
        student_prompts = [
            config.student_model.prompt_template.format(problem=row["problem"], explanation=row["explanation"])
            for row in batch_rows
        ]
        student_resps = student_pipeline(student_prompts, **generation_kwargs)

        for row, resp in zip(batch_rows, student_resps):
            malformed_count += int(row["is_malformed"])
            processed_count += 1
            student_correct = model_is_correct(resp[0]["generated_text"], row["ground_truth_solution"])
            dataset_ex = {
                "id": row["id"],
                "problem": row["problem"],
                "explanation": row["explanation"],
                "is_malformed": row["is_malformed"],
                "student_correct": student_correct,
                "student_resp": resp,
                "ground_truth_solution": row["ground_truth_solution"],
            }
            f.write(json.dumps(dataset_ex) + "\n")
        f.flush()

unload_pipeline(student_pipeline)

if processed_count:
    logger.info(f"Percentage of explanations that were malformed: {malformed_count / processed_count}")
