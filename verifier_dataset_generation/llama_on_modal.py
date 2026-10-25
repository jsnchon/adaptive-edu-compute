import modal
from pathlib import Path

# TODO: create some yaml configs to make all this metadata/hyperparams neater
# TODO: change to a class setup so we load model/data once per file entry instead of once per fn call
MAX_NEW_TOKENS = 1000
GPU_TYPE = "L4"
VOLUME_DIR = Path("/volume")
MODEL_CACHE_DIR = VOLUME_DIR / "models"
DATA_CACHE_DIR = VOLUME_DIR / "data"

# setup instructions:
# claim modal credits in 224n email
# run pip install modal
# run modal setup
# request access to llama 3.1/3.2 repos on hf. After getting access (should be pretty quick), 
model_vol = modal.Volume.from_name("llama-math-vol", create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "accelerate", "datasets")
)

app = modal.App("llama")

@app.function(
    image=image,
    gpu=GPU_TYPE, 
    secrets=[modal.Secret.from_name("llama-hf-token")],
    volumes={VOLUME_DIR: model_vol},
    timeout=600,
)
def generate_text():
    from transformers import pipeline
    from datasets import load_dataset
    import torch

    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    print("Loading dataset")
    math_ds = load_dataset("EleutherAI/hendrycks_math", "number_theory", cache_dir=DATA_CACHE_DIR)
    
    import random
    problem_idx = random.randint(0, len(math_ds["train"]) - 1)
    ex = math_ds["train"][problem_idx]
    problem, solution = ex["problem"], ex["solution"]
    prompt = "Explain why small language models are useful in 3 bullet points."
    print(f"Sample Problem: {problem}")

    print("Loading model")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"cache_dir": MODEL_CACHE_DIR},
    )

    model_vol.commit()

    prompt = fr"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in mathematics with a focus on teaching other users and models how to reason about math.
A user has asked the math problem below. Your task is to generate one CONCISE explanation for how to 
arrive at the problem's solution that accomplishes the following:

1) The explanation would allow another user/model to figure out how to answer the problem correctly
2) The explanation is pedagogical, consisting of high-level approaches and conceptual strategy without writing 
out any of the actual intermediate mathematical steps. Mentioning high-level, general formulas, theorems, 
or identities is fine

Please format your response as follows, making sure to include all special tags:

---

<explain>
# General strategy: include a few sentences here. For more difficult problems, this section can be longer
# Tips: include relevant formulas, theorems, or identities here
# Gotchas: include mistakes to avoid here
</explain>

Include any additional non-pedagogical reasoning steps you require to solve the problem here, outside the tags.

The final part of your response should be the numerical answer included in \boxed{{}}.

---

<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: {problem}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    print(f"Prompt: {prompt}")

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.9,
        "repetition_penalty": 1.1, 
    }

    outputs = pipe(
        prompt,
        **gen_kwargs,
    )
    response = outputs[0]["generated_text"]
    return response, solution

@app.local_entrypoint()
def main():
    for _ in range(1):
        response, solution = generate_text.remote()
        print(f"{"-" * 20}Model response{"-" * 20}\n{response}")
        print(f"{"-" * 20}Ground truth solution{"-" * 20}\n{solution}")
