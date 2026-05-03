import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_utils import (
    bootstrap_passk_curve,
    infer_dataset_label,
    infer_model_label,
    load_problem_success_rates,
    plot_bootstrap_passk_dataset,
)


DEFAULT_IMOPROOFBENCH_INPUT_PATHS = [
    "violetxi/qwen3-4b-instruct-imoproofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-dense-process-imoproofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-grpo-imoproofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-sft-imoproofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-self-distill-imoproofbench-summary-graded",
]

DEFAULT_PROOFBENCH_INPUT_PATHS = [
    "violetxi/qwen3-4b-instruct-proofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-dense-process-proofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-grpo-proofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-sft-proofbench-summary-graded",
    "violetxi/stage1_proof-qwen3-4b-self-distill-proofbench-summary-graded",
]

DEFAULT_INPUT_PATHS = DEFAULT_IMOPROOFBENCH_INPUT_PATHS + DEFAULT_PROOFBENCH_INPUT_PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bootstrap pass@k curves with confidence intervals.")
    parser.add_argument(
        "--input-path",
        action="append",
        dest="input_paths",
        help="Input graded dataset repo id, JSONL path, or local dataset directory. Repeat to override defaults.",
    )
    parser.add_argument(
        "--dataset-set",
        choices=("all", "imoproofbench", "proofbench"),
        default="all",
        help="Which built-in dataset preset to use when --input-path is not provided.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load when using Hugging Face datasets.")
    parser.add_argument("--output-dir", default="notebook/passk_bootstrap_plots", help="Directory for plots and CSV.")
    parser.add_argument("--success-threshold", type=float, default=6.0, help="Score threshold treated as success.")
    parser.add_argument("--max-k", type=int, default=64, help="Maximum k to plot.")
    parser.add_argument("--num-bootstrap", type=int, default=2000, help="Number of bootstrap replicates.")
    parser.add_argument("--ci-alpha", type=float, default=0.05, help="Two-sided CI alpha level.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_paths:
        input_paths = args.input_paths
    elif args.dataset_set == "imoproofbench":
        input_paths = DEFAULT_IMOPROOFBENCH_INPUT_PATHS
    elif args.dataset_set == "proofbench":
        input_paths = DEFAULT_PROOFBENCH_INPUT_PATHS
    else:
        input_paths = DEFAULT_INPUT_PATHS
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    dataset_rows: dict[str, list[dict[str, object]]] = {}
    csv_rows: list[dict[str, object]] = []
    for input_path in input_paths:
        dataset_label = infer_dataset_label(input_path)
        model_label = infer_model_label(input_path)
        success_counts, attempt_counts = load_problem_success_rates(
            input_path,
            split=args.split,
            success_threshold=args.success_threshold,
        )
        point, lower, upper = bootstrap_passk_curve(
            success_counts,
            attempt_counts,
            max_k=args.max_k,
            num_bootstrap=args.num_bootstrap,
            ci_alpha=args.ci_alpha,
            rng=rng,
        )
        dataset_rows.setdefault(dataset_label, []).append(
            {
                "model": model_label,
                "input_path": input_path,
                "point": point,
                "lower": lower,
                "upper": upper,
                "num_problems": int(len(success_counts)),
                "mean_attempts": float(np.mean(attempt_counts)),
            }
        )
        for k_idx, k in enumerate(range(1, args.max_k + 1)):
            csv_rows.append(
                {
                    "dataset": dataset_label,
                    "model": model_label,
                    "input_path": input_path,
                    "k": k,
                    "pass_at_k": float(point[k_idx]),
                    "ci_lower": float(lower[k_idx]),
                    "ci_upper": float(upper[k_idx]),
                    "num_problems": int(len(success_counts)),
                    "mean_attempts_per_problem": float(np.mean(attempt_counts)),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bootstrap_passk_curves.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("Inputs:")
    for dataset_label, rows in sorted(dataset_rows.items()):
        print(f"  {dataset_label}:")
        for row in sorted(rows, key=lambda item: str(item['model'])):
            print(
                f"    {row['model']}: {row['input_path']} "
                f"(problems={row['num_problems']}, mean_attempts={row['mean_attempts']:.2f})"
            )

    print("\nWrote:")
    print(f"  {csv_path}")
    for dataset_label, rows in sorted(dataset_rows.items()):
        plot_path = plot_bootstrap_passk_dataset(
            dataset_label,
            rows,
            output_dir=output_dir,
            success_threshold=args.success_threshold,
            max_k=args.max_k,
        )
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
