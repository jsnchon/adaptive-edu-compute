# MATH evaluation: frozen Phi-3-mini-4k-instruct

Measures performance of **microsoft/Phi-3-mini-4k-instruct** on **EleutherAI/hendrycks_math** (Hendrycks MATH). The model is used as-is (frozen); only the problem text is shown at inference—solutions are used only for scoring.

## Setup (AWS / GCP)

1. **Launch a GPU instance** (recommended for Phi-3):
   - **AWS**: e.g. `g5.xlarge` (1x A10G) or `p3.2xlarge` (1x V100). Use a **Ubuntu 22.04** AMI.
   - **GCP**: e.g. `n1-standard-4` + 1x T4 or L4 GPU; use a CUDA image or install drivers.

2. **Python 3.10+** and install dependencies:

```bash
cd /path/to/cs224n-final-project
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

3. **Hugging Face** (model + dataset are public; login only if you hit rate limits):

```bash
pip install huggingface_hub
huggingface-cli login   # optional
```

## Run evaluation

- **Quick run (e.g. 50 problems)**:

```bash
python run_math_eval.py --limit 50 --output results_50.json
```

- **Full test split** (no `--limit`):

```bash
python run_math_eval.py --output math_eval_results.json
```

- **Single subject** (if the dataset exposes configs, e.g. `algebra`):

```bash
python run_math_eval.py --config algebra --limit 100 --output algebra_results.json
```

- **CPU** (slow):

```bash
python run_math_eval.py --device cpu --limit 10
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `microsoft/Phi-3-mini-4k-instruct` | Hugging Face model id |
| `--dataset` | `EleutherAI/hendrycks_math` | Hugging Face dataset id |
| `--config` | (none) | Dataset config / subset name |
| `--split` | `test` | Dataset split |
| `--limit` | (all) | Max number of problems |
| `--output` | `math_eval_results.json` | Output JSON path |
| `--device` | cuda if available | `cuda` or `cpu` |
| `--max_new_tokens` | 1024 | Max tokens per answer |

Output JSON contains `metrics` (e.g. `accuracy`, `correct`, `total`) and `results` (per-problem predicted/reference answers and correctness).

## Data and model

- **Question set**: [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (Hendrycks MATH, 12.5k problems).
- **Model**: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (3.8B, 4K context).

Evaluation extracts the final `\boxed{...}` from the model output and compares it to the last `\boxed{...}` in the reference solution (normalized).
