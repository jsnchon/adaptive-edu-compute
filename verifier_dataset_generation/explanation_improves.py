import json
import os
from pathlib import Path
import re

import yaml
from dotenv import load_dotenv
from munch import munchify
import torch
from transformers import pipeline
from loguru import logger


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def extract_boxed_expr(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} expression, with spaces removed."""
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
    """Return True iff the boxed answer in response matches that in solution."""
    response_ans = extract_boxed_expr(response)
    solution_ans = extract_boxed_expr(solution)
    if response_ans is None or solution_ans is None:
        return False
    return response_ans == solution_ans


def load_config():
    with CONFIG_PATH.open("r") as f:
        cfg = yaml.safe_load(f)
    return munchify(cfg)


def create_student_pipeline(model_id: str, token: str, batch_size: int):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        token=token,
        dtype=torch.bfloat16,
        device_map="auto",
        batch_size=batch_size,
    )
    # Match other scripts: left pad to keep prompts aligned
    if pipe.tokenizer is not None:
        pipe.tokenizer.padding_side = "left"
        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Identify examples where the student succeeds with an explanation "
            "but fails without it, for qualitative analysis."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to labelled explanations dataset JSONL "
        "(e.g. verifier_dataset_generation/labeled_datasets/explanations_dataset_train.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path for selected examples. "
        "Defaults to <input_stem>_explanation_improves.jsonl in the same directory.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of qualifying examples to collect (default: 10).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_explanation_improves.jsonl")
    )

    load_dotenv(ROOT_DIR / ".env")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) not set; gated models may fail to download."
        )

    config = load_config()

    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": config.inference.max_new_tokens,
        "temperature": config.inference.temperature,
        "top_p": config.inference.top_p,
        "return_full_text": False,
    }

    student_batch_size = getattr(config.inference, "student_batch_size", 1)
    student_model_id = config.student_model.id
    student_prompt_template = config.student_model.prompt_template

    logger.info(f"Loading student model: {student_model_id}")
    student_pipeline = create_student_pipeline(
        student_model_id, hf_token, batch_size=student_batch_size
    )

    logger.info(f"Reading labelled dataset from {input_path}")
    logger.info(f"Saving qualifying examples to {output_path}")

    num_kept = 0
    total_seen = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(input_path):
            total_seen += 1

            # Only consider rows where the student was correct with an explanation
            if not row.get("student_correct", False):
                continue
            if row.get("is_malformed", False):
                continue

            problem = row["problem"]
            explanation = row["explanation"]
            solution = row.get("ground_truth_solution")

            if solution is None:
                continue

            # Original student response with explanation
            original_resp_text = None
            try:
                # modal_label_explanations stores HF pipeline outputs as a list
                original_resp_text = row["student_resp"][0]["generated_text"]
            except Exception:
                # Fall back if structure is different
                original_resp_text = str(row.get("student_resp"))

            # Re-run the student WITHOUT the explanation.
            # We pass an empty explanation string so the prompt template's
            # "If no explanation is provided..." branch is used.
            no_explanation_prompt = student_prompt_template.format(
                problem=problem,
                explanation="",
            )

            new_resp = student_pipeline(
                no_explanation_prompt,
                **generation_kwargs,
            )[0]["generated_text"]

            # Check if the student now fails without the explanation
            if not model_is_correct(new_resp, solution):
                example = {
                    "id": row["id"],
                    "problem": problem,
                    "explanation": explanation,
                    "ground_truth_solution": solution,
                    "student_resp_with_explanation": original_resp_text,
                    "student_resp_without_explanation": new_resp,
                }
                out_f.write(json.dumps(example) + "\n")
                num_kept += 1
                logger.info(
                    f"Found {num_kept} examples where explanation improves performance."
                )

                if num_kept >= args.num_examples:
                    break

    logger.info(
        f"Finished. Kept {num_kept} examples (scanned {total_seen} rows) "
        f"from {input_path}."
    )


if __name__ == "__main__":
    main()

# script that identifies query examples where the student can't solve the problem without explanation but can solve the problem with explanation
 