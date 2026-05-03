#!/usr/bin/env python3
"""Aggregate proof-technique annotations and plot deltas vs a baseline model.

This script expects annotated JSONL files produced by
`notebook/annotate_proof_techniques_async.py`.

It computes, for each `(problem_id, technique_tag)`, the frequency with which the
technique appears across rollouts for each model. It then subtracts the baseline
frequency, also keeps raw count deltas, and writes red/blue delta heatmaps,
per-technique sign-count bar plots, plus CSV summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GENERIC_TAGS = {
    "contradiction",
    "casework",
    "construction",
    "uniqueness_argument",
    "existence_argument",
    "exhaustive_search",
    "induction",
    "infinite_descent",
}

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATION_DIR = SCRIPT_DIR / "model_solution_annotations"

DEFAULT_MODEL_PATHS = {
    "instruct": ANNOTATION_DIR / "qwen3-4b-instruct-proofbench-summary-graded-techniques.jsonl",
    "grpo": ANNOTATION_DIR / "stage1_proof-qwen3-4b-grpo-proofbench-summary-graded-techniques.jsonl",
    "dense_process": ANNOTATION_DIR / "stage1_proof-qwen3-4b-dense-process-proofbench-summary-graded-techniques.jsonl",
    "sft": ANNOTATION_DIR / "stage1_proof-qwen3-4b-sft-proofbench-summary-graded-techniques.jsonl",
    "self_distill": ANNOTATION_DIR / "stage1_proof-qwen3-4b-self-distill-proofbench-summary-graded-techniques.jsonl",
}

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "proof_technique_delta_analysis"


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def parse_input_specs(specs: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input spec {spec!r}. Expected LABEL=PATH.")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --input spec {spec!r}.")
        parsed[label] = path
    return parsed


def get_problem_key(row: dict[str, Any], path: str) -> str:
    for field in ("problem_id", "question_id", "problem"):
        value = row.get(field)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str

    source_index = row.get("source_index")
    if source_index is not None:
        return f"source_index:{source_index}"

    raise ValueError(
        "Could not determine a problem identifier in row from "
        f"{path}. Tried problem_id, question_id, problem, and source_index."
    )


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_tags_from_row(
    row: dict[str, Any],
    *,
    include_attempted: bool,
    exclude_generic: bool,
) -> list[str]:
    annotation = row.get("proof_technique_annotation") or {}
    techniques = annotation.get("proof_techniques") or []
    tags = []
    for entry in techniques:
        tag = entry.get("tag")
        status = entry.get("status")
        if not tag:
            continue
        if status == "attempted" and not include_attempted:
            continue
        if exclude_generic and tag in GENERIC_TAGS:
            continue
        if tag not in tags:
            tags.append(tag)
    return tags


def build_frequency_long_df(
    model_to_path: dict[str, str],
    *,
    include_attempted: bool,
    exclude_generic: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model_label, path in model_to_path.items():
        rows = load_rows(path)
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            problem_id = get_problem_key(row, path)
            problem_state = grouped.setdefault(problem_id, {"num_rollouts": 0, "tag_counts": {}})
            problem_state["num_rollouts"] += 1
            tags = extract_tags_from_row(
                row,
                include_attempted=include_attempted,
                exclude_generic=exclude_generic,
            )
            for tag in tags:
                problem_state["tag_counts"][tag] = problem_state["tag_counts"].get(tag, 0) + 1

        for problem_id, state in grouped.items():
            num_rollouts = int(state["num_rollouts"])
            tag_counts = state["tag_counts"]
            if not tag_counts:
                records.append(
                    {
                        "model": model_label,
                        "problem_id": problem_id,
                        "tag": "__no_tag__",
                        "count": 0,
                        "num_rollouts": num_rollouts,
                        "frequency": 0.0,
                    }
                )
            else:
                for tag, count in tag_counts.items():
                    records.append(
                        {
                            "model": model_label,
                            "problem_id": problem_id,
                            "tag": tag,
                            "count": int(count),
                            "num_rollouts": num_rollouts,
                            "frequency": float(count) / float(num_rollouts),
                        }
                    )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No frequency records were created.")
    return df


def complete_frequency_grid(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["tag"] != "__no_tag__"].copy()
    models = sorted(filtered["model"].unique())
    problems = sorted(filtered["problem_id"].unique())
    tags = sorted(filtered["tag"].unique())

    rollout_df = (
        df.groupby(["model", "problem_id"], as_index=False)["num_rollouts"]
        .max()
        .rename(columns={"num_rollouts": "num_rollouts"})
    )
    full_index = pd.MultiIndex.from_product([models, problems, tags], names=["model", "problem_id", "tag"])
    completed = (
        filtered.set_index(["model", "problem_id", "tag"])[["count", "frequency"]]
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    completed = completed.merge(rollout_df, on=["model", "problem_id"], how="left")
    completed["count"] = completed["count"].astype(int)
    completed["frequency"] = completed["frequency"].astype(float)
    completed["num_rollouts"] = completed["num_rollouts"].astype(int)
    return completed


def compute_delta_df(freq_df: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    baseline = (
        freq_df[freq_df["model"] == baseline_model][["problem_id", "tag", "count", "frequency"]]
        .rename(columns={"count": "baseline_count", "frequency": "baseline_frequency"})
    )
    merged = freq_df.merge(baseline, on=["problem_id", "tag"], how="left")
    merged["baseline_count"] = merged["baseline_count"].fillna(0).astype(int)
    merged["baseline_frequency"] = merged["baseline_frequency"].fillna(0.0)
    merged["count_delta"] = merged["count"] - merged["baseline_count"]
    merged["delta"] = merged["frequency"] - merged["baseline_frequency"]
    return merged


def choose_technique_order(delta_df: pd.DataFrame, baseline_model: str, top_k: int | None) -> list[str]:
    comparison_df = delta_df[delta_df["model"] != baseline_model].copy()
    summary = (
        comparison_df.groupby("tag", as_index=False)
        .agg(
            mean_abs_delta=("delta", lambda s: float(np.mean(np.abs(s)))),
            max_abs_delta=("delta", lambda s: float(np.max(np.abs(s)))),
            mean_frequency=("frequency", "mean"),
        )
        .sort_values(["mean_abs_delta", "max_abs_delta", "mean_frequency", "tag"], ascending=[False, False, False, True])
    )
    tags = summary["tag"].tolist()
    if top_k is not None:
        tags = tags[:top_k]
    return tags


def choose_problem_order(delta_df: pd.DataFrame, model_label: str) -> list[str]:
    model_df = delta_df[delta_df["model"] == model_label].copy()
    summary = (
        model_df.groupby("problem_id", as_index=False)
        .agg(mean_abs_delta=("delta", lambda s: float(np.mean(np.abs(s)))))
        .sort_values(["mean_abs_delta", "problem_id"], ascending=[False, True])
    )
    return summary["problem_id"].tolist()


def plot_delta_heatmap(
    delta_df: pd.DataFrame,
    *,
    model_label: str,
    baseline_model: str,
    tags: list[str],
    output_path: Path,
    title_suffix: str,
) -> None:
    model_df = delta_df[(delta_df["model"] == model_label) & (delta_df["tag"].isin(tags))].copy()
    problem_order = choose_problem_order(delta_df, model_label)
    pivot = (
        model_df.pivot(index="problem_id", columns="tag", values="delta")
        .reindex(index=problem_order, columns=tags)
        .fillna(0.0)
    )
    if pivot.empty:
        raise ValueError(f"No data to plot for model {model_label}.")

    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    if vmax == 0.0:
        vmax = 1.0

    width = max(12.0, 0.42 * len(tags) + 4.0)
    height = max(10.0, 0.18 * len(pivot.index) + 3.0)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        pivot,
        cmap="bwr",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.0,
        cbar_kws={"label": f"Technique frequency delta vs {baseline_model}"},
    )
    plt.title(f"{model_label} vs {baseline_model}: proof-technique frequency delta{title_suffix}")
    plt.xlabel("Technique")
    plt.ylabel("Problem")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, int(200 / max(len(tags), 1))))
    plt.yticks(fontsize=max(5, int(500 / max(len(pivot.index), 1))))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_technique_count_sign_summary(delta_df: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    comparison_df = delta_df[delta_df["model"] != baseline_model].copy()
    summary = (
        comparison_df.groupby(["tag", "model"], as_index=False)
        .agg(
            num_positive_problems=("count_delta", lambda s: int((s > 0).sum())),
            num_negative_problems=("count_delta", lambda s: int((s < 0).sum())),
            num_zero_problems=("count_delta", lambda s: int((s == 0).sum())),
            mean_count_delta=("count_delta", "mean"),
            num_problems=("problem_id", "nunique"),
        )
        .sort_values(["tag", "model"], ascending=[True, True])
    )
    return summary


def plot_technique_count_sign_barplots(
    technique_count_sign_df: pd.DataFrame,
    *,
    model_order: list[str],
    output_dir: Path,
    baseline_model: str,
    title_suffix: str,
) -> None:
    barplot_dir = output_dir / "count_delta_sign_barplots_by_technique"
    barplot_dir.mkdir(parents=True, exist_ok=True)
    comparison_model_order = [model for model in model_order if model != baseline_model]

    for tag in sorted(technique_count_sign_df["tag"].unique()):
        tag_df = technique_count_sign_df[technique_count_sign_df["tag"] == tag].copy()
        tag_df["model"] = pd.Categorical(tag_df["model"], categories=comparison_model_order, ordered=True)
        tag_df = tag_df.sort_values("model")

        width = max(7.0, 1.6 * len(comparison_model_order) + 2.5)
        plt.figure(figsize=(width, 5.0))
        ax = plt.gca()
        x = np.arange(len(tag_df))
        bar_width = 0.36
        positive_bars = ax.bar(
            x - bar_width / 2.0,
            tag_df["num_positive_problems"],
            width=bar_width,
            color="red",
            label=f"> 0 vs {baseline_model}",
        )
        negative_bars = ax.bar(
            x + bar_width / 2.0,
            -tag_df["num_negative_problems"],
            width=bar_width,
            color="blue",
            label=f"< 0 vs {baseline_model}",
        )

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Model")
        ax.set_ylabel(f"Number of problems with raw count delta sign vs {baseline_model}")
        ax.set_title(f"{tag}")
        ax.set_xticks(x)
        ax.set_xticklabels(tag_df["model"].astype(str), rotation=25, ha="right")
        ax.legend()
        ax.bar_label(positive_bars, labels=[str(int(v)) for v in tag_df["num_positive_problems"]], padding=3, fontsize=9)
        ax.bar_label(negative_bars, labels=[str(int(v)) for v in tag_df["num_negative_problems"]], padding=3, fontsize=9)

        plt.tight_layout()
        plt.savefig(barplot_dir / f"count_delta_sign__{slugify(tag)}.png", dpi=220)
        plt.close()


def write_summary_outputs(
    freq_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    *,
    baseline_model: str,
    model_order: list[str],
    output_dir: Path,
    top_k: int | None,
    title_suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    freq_df.to_csv(output_dir / "technique_frequencies_long.csv", index=False)
    freq_wide = (
        freq_df.pivot_table(
            index=["model", "problem_id"],
            columns="tag",
            values="frequency",
            fill_value=0.0,
        )
        .reset_index()
    )
    freq_wide.to_csv(output_dir / "technique_frequencies_wide.csv", index=False)

    delta_df.to_csv(output_dir / "technique_deltas_long.csv", index=False)

    technique_count_sign_df = build_technique_count_sign_summary(delta_df, baseline_model)
    technique_count_sign_df.to_csv(output_dir / "technique_count_delta_sign_summary_by_model.csv", index=False)
    plot_technique_count_sign_barplots(
        technique_count_sign_df,
        model_order=model_order,
        output_dir=output_dir,
        baseline_model=baseline_model,
        title_suffix=title_suffix,
    )

    tags = choose_technique_order(delta_df, baseline_model, top_k=top_k)
    for model_label in sorted(delta_df["model"].unique()):
        if model_label == baseline_model:
            continue
        model_delta = delta_df[delta_df["model"] == model_label].copy()
        model_delta.to_csv(output_dir / f"delta_vs_{slugify(baseline_model)}__{slugify(model_label)}.csv", index=False)
        plot_delta_heatmap(
            delta_df,
            model_label=model_label,
            baseline_model=baseline_model,
            tags=tags,
            output_path=output_dir / f"delta_heatmap__{slugify(model_label)}_vs_{slugify(baseline_model)}.png",
            title_suffix=title_suffix,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze proof-technique deltas vs a baseline model.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Annotated JSONL input as LABEL=PATH. Repeat once per model.",
    )
    parser.add_argument(
        "--baseline",
        default="instruct",
        help="Baseline model label from the provided --input specs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSVs and heatmaps.",
    )
    parser.add_argument(
        "--include-attempted",
        action="store_true",
        help="Count `status=attempted` tags in addition to `used` tags.",
    )
    parser.add_argument(
        "--exclude-generic",
        action="store_true",
        help="Exclude general proof-technique tags like contradiction and casework.",
    )
    parser.add_argument(
        "--top-k-techniques",
        type=int,
        default=None,
        help="Restrict heatmaps to the top K techniques by mean absolute delta.",
    )
    parser.add_argument(
        "--use-default-model-paths",
        "--use-default-model-path-template",
        dest="use_default_model_paths",
        action="store_true",
        help=(
            "Use the five default ProofBench annotation JSONLs in notebook/model_solution_annotations/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_to_path = parse_input_specs(args.input)
    if args.use_default_model_paths:
        for label, path in DEFAULT_MODEL_PATHS.items():
            model_to_path.setdefault(label, str(path))

    if not model_to_path:
        raise ValueError("No inputs provided. Use --input LABEL=PATH or --use-default-model-paths.")
    if args.baseline not in model_to_path:
        raise ValueError(f"Baseline label {args.baseline!r} not found in inputs: {sorted(model_to_path)}")

    missing = [f"{label}={path}" for label, path in model_to_path.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing annotated JSONL files:\n"
            + "\n".join(missing)
            + "\nRun notebook/annotate_proof_techniques_async.py first for those datasets."
        )

    freq_long = build_frequency_long_df(
        model_to_path,
        include_attempted=args.include_attempted,
        exclude_generic=args.exclude_generic,
    )
    freq_long = complete_frequency_grid(freq_long)
    delta_df = compute_delta_df(freq_long, baseline_model=args.baseline)

    title_suffix = ""
    if args.include_attempted:
        title_suffix += " (used + attempted)"
    else:
        title_suffix += " (used only)"
    if args.exclude_generic:
        title_suffix += ", generic excluded"

    write_summary_outputs(
        freq_long,
        delta_df,
        baseline_model=args.baseline,
        model_order=list(model_to_path),
        output_dir=Path(args.output_dir),
        top_k=args.top_k_techniques,
        title_suffix=title_suffix,
    )

    print(f"Wrote analysis outputs to {args.output_dir}")
    print(f"Baseline: {args.baseline}")
    print("Inputs:")
    for label, path in sorted(model_to_path.items()):
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
