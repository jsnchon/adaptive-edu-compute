#!/usr/bin/env python3
"""
Generate minimal visualizations to judge whether a sampled subset (default: 500, seed 42)
is representative of the full Hendrycks MATH train set, using the same sampling logic as
teacher_student_pipeline / verifier_dataset_generation.
"""

import argparse
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import numpy as np
import re

VIZ_DIR = Path(__file__).resolve().parent
ROOT_DIR = VIZ_DIR.parent
DATASET_REPO = "EleutherAI/hendrycks_math"
SHUFFLE_SEED = 42


def _load_train_concat():
    """Concatenate all Hendrycks MATH train configs, matching project pipelines."""
    configs = get_dataset_config_names(DATASET_REPO)
    # Add "type" (subject) from config name so we can plot by subject after concatenation
    parts = []
    for cfg in configs:
        ds = load_dataset(DATASET_REPO, cfg, split="train")
        if "type" not in ds.column_names:
            ds = ds.add_column("type", [cfg.replace("_", " ").title()] * len(ds))
        parts.append(ds)
    dataset = concatenate_datasets(parts)
    return dataset


def load_500_sample(seed: int = SHUFFLE_SEED, size: int = 500):
    """Load the same sample subset as the pipeline (seed 42, first N after shuffle)."""
    dataset = _load_train_concat()
    dataset = dataset.shuffle(seed=seed).select(range(size))
    return dataset


def _parse_level_to_int(level) -> int | None:
    if level is None:
        return None
    if isinstance(level, (int, float)):
        return int(level)
    s = str(level).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _type_level_table(dataset) -> tuple[list[str], list[int], np.ndarray]:
    """
    Returns (type_names, level_names, counts[type_idx, level_idx]).
    Filters rows with unparseable levels.
    """
    types = [str(t) for t in dataset["type"]]
    levels = [_parse_level_to_int(L) for L in dataset["level"]]
    pairs = [(t, lv) for t, lv in zip(types, levels) if lv is not None]
    if not pairs:
        return [], [], np.zeros((0, 0), dtype=int)

    type_names = sorted({t for t, _ in pairs})
    level_names = sorted({lv for _, lv in pairs})
    type_to_i = {t: i for i, t in enumerate(type_names)}
    level_to_j = {lv: j for j, lv in enumerate(level_names)}

    counts = np.zeros((len(type_names), len(level_names)), dtype=int)
    for t, lv in pairs:
        counts[type_to_i[t], level_to_j[lv]] += 1
    return type_names, level_names, counts


def _plot_type_level_heatmaps(full_ds, sample_ds, out_path: Path, title_prefix: str):
    type_names_f, level_names_f, full_counts = _type_level_table(full_ds)
    type_names_s, level_names_s, sample_counts = _type_level_table(sample_ds)

    type_names = sorted(set(type_names_f) | set(type_names_s))
    level_names = sorted(set(level_names_f) | set(level_names_s))
    type_to_i = {t: i for i, t in enumerate(type_names)}
    level_to_j = {lv: j for j, lv in enumerate(level_names)}

    full = np.zeros((len(type_names), len(level_names)), dtype=float)
    samp = np.zeros((len(type_names), len(level_names)), dtype=float)

    # align full counts
    for i, t in enumerate(type_names_f):
        for j, lv in enumerate(level_names_f):
            full[type_to_i[t], level_to_j[lv]] = float(full_counts[i, j])

    # align sample counts
    for i, t in enumerate(type_names_s):
        for j, lv in enumerate(level_names_s):
            samp[type_to_i[t], level_to_j[lv]] = float(sample_counts[i, j])

    full_prop = full / max(full.sum(), 1.0)
    samp_prop = samp / max(samp.sum(), 1.0)
    diff = samp_prop - full_prop

    fig, axs = plt.subplots(1, 3, figsize=(16, max(5, 0.35 * len(type_names))), constrained_layout=True)

    vmax = max(full_prop.max(), samp_prop.max(), 1e-12)
    im0 = axs[0].imshow(full_prop, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    axs[0].set_title(f"{title_prefix}: full train (proportion)")
    im1 = axs[1].imshow(samp_prop, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    axs[1].set_title(f"{title_prefix}: sample (proportion)")

    d = np.max(np.abs(diff)) if diff.size else 1.0
    im2 = axs[2].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-d, vmax=d)
    axs[2].set_title(f"{title_prefix}: sample − full (Δ proportion)")

    for ax in axs:
        ax.set_yticks(range(len(type_names)))
        ax.set_yticklabels(type_names)
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels([str(x) for x in level_names])
        ax.set_xlabel("Level")
    axs[0].set_ylabel("Type (subject)")

    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_marginal_overlays(full_ds, sample_ds, out_path: Path, seed: int, size: int):
    # type proportions
    full_types = [str(t) for t in full_ds["type"]]
    samp_types = [str(t) for t in sample_ds["type"]]
    type_names = sorted(set(full_types) | set(samp_types))
    full_type_counts = {t: 0 for t in type_names}
    samp_type_counts = {t: 0 for t in type_names}
    for t in full_types:
        full_type_counts[t] += 1
    for t in samp_types:
        samp_type_counts[t] += 1

    full_type_prop = np.array([full_type_counts[t] for t in type_names], dtype=float) / max(len(full_types), 1)
    samp_type_prop = np.array([samp_type_counts[t] for t in type_names], dtype=float) / max(len(samp_types), 1)

    # level proportions (parsed)
    full_lv = [_parse_level_to_int(L) for L in full_ds["level"]]
    samp_lv = [_parse_level_to_int(L) for L in sample_ds["level"]]
    full_lv = [lv for lv in full_lv if lv is not None]
    samp_lv = [lv for lv in samp_lv if lv is not None]
    level_names = sorted(set(full_lv) | set(samp_lv))
    full_lv_counts = {lv: 0 for lv in level_names}
    samp_lv_counts = {lv: 0 for lv in level_names}
    for lv in full_lv:
        full_lv_counts[lv] += 1
    for lv in samp_lv:
        samp_lv_counts[lv] += 1
    full_lv_prop = np.array([full_lv_counts[lv] for lv in level_names], dtype=float) / max(len(full_lv), 1)
    samp_lv_prop = np.array([samp_lv_counts[lv] for lv in level_names], dtype=float) / max(len(samp_lv), 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    # Type overlay (horizontal bars)
    y = np.arange(len(type_names))
    axs[0].barh(y - 0.2, full_type_prop, height=0.4, label="full", color="lightsteelblue", edgecolor="black", linewidth=0.3)
    axs[0].barh(y + 0.2, samp_type_prop, height=0.4, label="sample", color="coral", edgecolor="black", linewidth=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(type_names)
    axs[0].set_xlabel("Proportion")
    axs[0].set_title("Type distribution (proportions)")
    axs[0].legend()

    # Level overlay (discrete bars)
    x = np.arange(len(level_names), dtype=float)
    axs[1].bar(
        x - 0.2,
        full_lv_prop,
        width=0.4,
        label="full",
        color="lightsteelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    axs[1].bar(
        x + 0.2,
        samp_lv_prop,
        width=0.4,
        label="sample",
        color="coral",
        edgecolor="black",
        linewidth=0.3,
    )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([str(lv) for lv in level_names])
    axs[1].set_xlabel("Level")
    axs[1].set_ylabel("Proportion")
    axs[1].set_title("Level distribution (proportions)")
    axs[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axs[1].legend()

    fig.suptitle(f"Representativeness check for sample size={size}, seed={seed}", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate representativeness visualizations (sample vs full train)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=VIZ_DIR / "figures",
        help="Directory to save figures (default: visualizations/figures)",
    )
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED, help="Shuffle seed (default: 42)")
    parser.add_argument("--size", type=int, default=500, help="Sample size (default: 500)")
    args = parser.parse_args()

    dataset = load_500_sample(seed=args.seed, size=args.size)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = _load_train_concat()
    if "type" not in full_ds.column_names or "level" not in full_ds.column_names:
        raise RuntimeError(f"Full dataset missing 'type' or 'level'. Available: {full_ds.column_names}")
    if "type" not in dataset.column_names or "level" not in dataset.column_names:
        raise RuntimeError(f"Sample dataset missing 'type' or 'level'. Available: {dataset.column_names}")

    _plot_type_level_heatmaps(
        full_ds,
        dataset,
        out_dir / "representativeness_full_vs_sample_type_x_level.png",
        title_prefix="Type × level",
    )
    print(f"Saved full-vs-sample heatmaps to {out_dir / 'representativeness_full_vs_sample_type_x_level.png'}")

    _plot_marginal_overlays(
        full_ds,
        dataset,
        out_dir / "representativeness_full_vs_sample_marginals.png",
        seed=args.seed,
        size=args.size,
    )
    print(f"Saved marginal overlays to {out_dir / 'representativeness_full_vs_sample_marginals.png'}")


if __name__ == "__main__":
    main()
