import json
from pathlib import Path
import re
import gc

import yaml
from munch import munchify
import torch
from transformers import pipeline
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from loguru import logger
from tqdm import tqdm
import modal

"""
SETUP (after intsalling modal):
    1) modal setup
    2) modal secret create hf-token HF_TOKEN=your_token_here
    3) modal volume create label-explanations-vol
    4) modal volume put label-explanations-vol config.yaml /config.yaml
    5) modal run --detach modal_label_explanations.py

While the script is running, you can view the process from the web ui or running `modal container list`
then `modal exec <container_id> /bin/bash` to enter the container

Note that if you change the config, you need to reupload it using modal volume rm then modal volume put.
Similarly, if you want to restart from scratch completely you need to delete the cached jsons made so far
using modal volume rm (volumes and the files in them are viewable on the web ui)
"""

GPU = "A100"
BASE_DIR = Path("/data")
HF_CACHE_DIR = BASE_DIR / "hf_cache"
ROOT_DIR =  Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DATASET_REPO = "EleutherAI/hendrycks_math"



app = modal.App("label-explanations")

app_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "datasets",
        "accelerate",
        "torch",
        "munch",
        "pyyaml",
        "python-dotenv",
        "loguru",
        "tqdm",
        "hf_transfer"  # faster downloads
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("label-explanations-vol", create_if_missing=True)

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

def count_malformed_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("is_malformed", False):
                    count += 1
            except json.JSONDecodeError:
                break
    return count

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# load MATH dataset for a specific split
def load_dataset_for_split(split: str):
    dataset_configs = get_dataset_config_names(DATASET_REPO)
    all_split_list = [
        load_dataset(DATASET_REPO, cfg, split=split, cache_dir=HF_CACHE_DIR) 
        for cfg in dataset_configs
    ]
    dataset = concatenate_datasets(all_split_list)
    return dataset

# generate output file paths for different splits
def get_split_output_paths(base_out_file: Path, split: str):
    out_file = base_out_file.with_stem(f"{base_out_file.stem}_{split}")
    teacher_cache_file = out_file.with_name(f"{out_file.stem}_teacher_cache.jsonl")
    return out_file, teacher_cache_file

@app.function(
    image=app_image,
    gpu="A100",
    volumes={BASE_DIR: vol},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("hf-token")] 
)
def run_pipeline():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Please upload config.yaml to the volume at {CONFIG_PATH}")

    with CONFIG_PATH.open("r") as f:
        config = yaml.safe_load(f)
        config = munchify(config)

    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": config.inference.max_new_tokens,
        "temperature": config.inference.temperature,
        "top_p": config.inference.top_p,
        "return_full_text": False,
    }
    logger.info(f"Generation settings: {generation_kwargs}")

    teacher_batch_size, student_batch_size = config.inference.teacher_batch_size, config.inference.student_batch_size
    base_out_file = BASE_DIR / config.general.dataset_out_file
    logger.info(f"Using a batch size of {teacher_batch_size} for generating explanations and {student_batch_size} for labelling them")

    for split in ["train", "test"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"{"="*80}\nProcessing {split.upper()} split\n{"="*80}")

        dataset = load_dataset_for_split(split)
        if split_size := config.general.split_size:
            dataset = dataset.shuffle(seed=42).select(range(split_size))
        logger.info(f"Dataset length for {split}: {len(dataset)}")
        logger.debug(f"First example of {split} dataset: {dataset[0]}")

        out_file, teacher_cache_file = get_split_output_paths(base_out_file, split)
        logger.info(f"Saving {split} explanations and quality labels to {out_file}")
        logger.info(f"Caching {split} teacher explanations to {teacher_cache_file}")
        
        process_split(dataset, out_file, teacher_cache_file, teacher_batch_size, student_batch_size, generation_kwargs, config, split)

def process_split(dataset, out_file, teacher_cache_file, teacher_batch_size, student_batch_size, generation_kwargs, config, split):

    """
    Process a single split of the dataset through the teacher and student pipelines.
    """
    def create_pipeline(model_id, batch_size):
        pipe = pipeline(
            "text-generation", 
            model=model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            batch_size=batch_size,
            model_kwargs={"cache_dir": HF_CACHE_DIR},
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

    if teacher_start_idx >= len(dataset):
        logger.info(f"Teacher explanations already fully processed for {split}")
    else:
        teacher_pipeline = create_pipeline(config.teacher_model.id, teacher_batch_size)
        with teacher_cache_file.open("a", encoding="utf-8") as f:
            remaining_dataset = dataset.select(range(teacher_start_idx, len(dataset)))
            for i in tqdm(range(0, len(remaining_dataset), teacher_batch_size), desc="Teacher explanation generation"):
                batch = remaining_dataset.select(range(i, min(i + teacher_batch_size, len(remaining_dataset))))
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
                vol.commit()

        unload_pipeline(teacher_pipeline)

    student_start_idx = count_valid_jsonl_rows(out_file)
    if student_start_idx:
        logger.info(f"Existing labelled dataset found. Resuming student labelling from index: {student_start_idx}")

    teacher_rows = list(iter_jsonl(teacher_cache_file))[student_start_idx:]
    if not teacher_rows:
        logger.info(f"Student labeling already fully processed for {split}")
    else:
        student_pipeline = create_pipeline(config.student_model.id, student_batch_size)
        malformed_count = count_malformed_jsonl_rows(out_file)
        processed_count = student_start_idx
        logger.debug(f"Starting malformed count: {malformed_count}\nStarting processed count: {processed_count}")
        with out_file.open("a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(teacher_rows), student_batch_size), desc="Explanation labelling"):
                batch_rows = teacher_rows[i:i + student_batch_size]
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
                vol.commit()

        unload_pipeline(student_pipeline)

        if processed_count:
            logger.info(f"Percentage of {split} explanations that were malformed: {(malformed_count / processed_count) * 100}")

@app.local_entrypoint()
def main():
    run_pipeline.remote()
