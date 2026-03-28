"""
Two-pass evaluation: generate reasoning trace, then summarize into a clean proof.

Pass 1 (generation): The model produces a full response (long reasoning trace).
Pass 2 (summarization): Conditioned on the problem + trace, the model produces
         a clean, rigorous proof suitable for submission (temperature 0, n=1).

Usage:
    # Full two-pass run:
    uv run python scripts/run_summary.py \
        --model-config vllm/vllm-violetxi-stage2-qwen3-4b \
        --output-path outputs/stage2-qwen3-4b-imoproofbench-summary.jsonl

    # With a different model for summarization:
    uv run python scripts/run_summary.py \
        --model-config vllm/vllm-violetxi-stage2-qwen3-4b \
        --summary-model-config openai/gpt-5-mini \
        --output-path outputs/stage2-qwen3-4b-imoproofbench-summary.jsonl

    # Skip pass 1 (reuse existing generation output):
    uv run python scripts/run_summary.py \
        --model-config vllm/vllm-violetxi-stage2-qwen3-4b \
        --generation-path outputs/stage2-qwen3-4b-imoproofbench.jsonl \
        --output-path outputs/stage2-qwen3-4b-imoproofbench-summary.jsonl
"""

import argparse
import yaml
import os
from datasets import load_dataset, concatenate_datasets
from imobench.evaluation import run_bench
from loguru import logger

DEFAULT_SUMMARY_PROMPT = """\
You are given a mathematical problem and a detailed reasoning trace produced by a problem-solving model. The reasoning trace may be long, meandering, contain false starts, backtracking, or informal language. Your job is to distill it into a single, clean, rigorous mathematical proof.

**Instructions:**
- Extract the correct logical thread from the reasoning trace and present it as a polished proof.
- Remove any false starts, dead ends, self-corrections, or redundant exploration.
- Use standard mathematical notation and proof conventions (e.g. "Let", "Suppose", "Then", "Therefore", "QED").
- If the reasoning trace contains multiple approaches, select the one that leads to a complete, correct proof.
- Do NOT add new mathematical ideas or steps that are not supported by the reasoning trace. You are summarizing, not solving.
- If the reasoning trace does not contain a complete proof, produce the best partial proof you can from what is given.

---

**Problem:**
{problem_statement}

---

**Reasoning Trace:**
{reasoning_trace}

---

Now write a clean, rigorous proof based on the above reasoning trace."""


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass eval: generate reasoning trace, then summarize into a proof."
    )
    # Pass 1 (generation) args
    parser.add_argument("--model-config", type=str, required=True,
                        help="Model config for generation (pass 1). Relative to configs/models/")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt template for pass 1. Default: proofbench_run.txt")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset path. Default: lm-provers/IMOProofBench")
    parser.add_argument("--problem-column", type=str, default="problem",
                        help="Column name for problems in the dataset.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of times to run the dataset (for pass@n in pass 1).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore cached intermediate results and re-run.")

    # Pass 1 skip option
    parser.add_argument("--generation-path", type=str, default=None,
                        help="Path to existing pass 1 output. If provided, skip generation "
                             "and load problems + traces directly from this file.")

    # Pass 2 (summarization) args
    parser.add_argument("--summary-model-config", type=str, default=None,
                        help="Model config for summarization (pass 2). "
                             "Defaults to --model-config if not set.")
    parser.add_argument("--summary-prompt", type=str, default=None,
                        help="Path to a custom summary prompt template file. "
                             "If not set, uses the built-in default.")
    parser.add_argument("--summary-max-tokens", type=int, default=None,
                        help="Override max_tokens for the summarization pass.")

    # Output
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the final results.")
    parser.add_argument("--final-answer", action="store_true",
                        help="Whether its final answer or proof grading.")

    args = parser.parse_args()

    # Defaults
    if args.prompt is None and args.final_answer:
        args.prompt = "configs/prompts/answerbench_run.txt"
    elif args.prompt is None:
        args.prompt = "configs/prompts/proofbench_run.txt"
    if args.data_path is None and args.final_answer:
        args.data_path = "lm-provers/IMOBench-FinalAnswer"
    elif args.data_path is None:
        args.data_path = "lm-provers/IMOProofBench"
    if args.summary_model_config is None:
        args.summary_model_config = args.model_config

    # ---- Pass 1: Generation ----
    if args.generation_path is not None:
        logger.info(f"Skipping pass 1, loading from {args.generation_path}")
        # Load everything from the generation file — it already has problems,
        # model_solution, grading_guidelines, etc. No need to load dataset separately.
        dataset = load_dataset("json", data_files=args.generation_path)[args.split]
        questions = list(dataset[args.problem_column])
        reasoning_traces = list(dataset["model_solution"])
        gen_costs = [{"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}] * len(questions)
        logger.info(f"Loaded {len(questions)} entries from generation file")

        # Drop columns that will be re-created by pass 2
        for col in ["model_solution", "cost_run", "history"]:
            if col in dataset.column_names:
                dataset = dataset.remove_columns(col)
    else:
        logger.info("=== Pass 1: Generation ===")
        # Load dataset
        if os.path.exists(args.data_path):
            dataset = load_dataset("json", data_files=args.data_path)[args.split]
        else:
            dataset = load_dataset(args.data_path)[args.split]

        if args.n > 1:
            dataset = concatenate_datasets([dataset] * args.n)

        questions = list(dataset[args.problem_column])

        with open(os.path.join("configs/models", args.model_config + ".yaml")) as f:
            gen_model_config = yaml.safe_load(f)
        with open(args.prompt) as f:
            gen_prompt_template = f.read()

        gen_results = run_bench(
            gen_model_config, gen_prompt_template, questions,
            overwrite=args.overwrite,
            other_params={
                "output_path": args.output_path,
                "n": args.n,
                "model_config_name": args.model_config,
                "pass": "generation",
            },
            path=args.model_config,
        )
        reasoning_traces = [row["response"] for row in gen_results]
        gen_costs = [row["cost"] for row in gen_results]

        gen_cost_total = sum(c["cost"] for c in gen_costs)
        logger.info(f"Pass 1 cost: ${gen_cost_total:.4f}")

    # ---- Pass 2: Summarization (temperature=0, n=1) ----
    logger.info("=== Pass 2: Summarization ===")
    with open(os.path.join("configs/models", args.summary_model_config + ".yaml")) as f:
        summary_model_config = yaml.safe_load(f)

    # Force deterministic summarization
    summary_model_config["temperature"] = 0

    if args.summary_max_tokens is not None:
        summary_model_config["max_tokens"] = args.summary_max_tokens

    if args.summary_prompt is not None:
        with open(args.summary_prompt) as f:
            summary_prompt_template = f.read()
    else:
        summary_prompt_template = DEFAULT_SUMMARY_PROMPT

    # Build summary prompts by formatting with both problem and reasoning trace.
    # We pass the fully formatted prompt as the "question" and use a passthrough
    # template so run_bench sends it as-is.
    summary_questions = []
    for question, trace in zip(questions, reasoning_traces):
        summary_questions.append(
            summary_prompt_template.format(
                problem_statement=question,
                reasoning_trace=trace,
            )
        )
    passthrough_template = "{problem_statement}"

    summary_results = run_bench(
        summary_model_config, passthrough_template, summary_questions,
        overwrite=args.overwrite,
        other_params={
            "output_path": args.output_path,
            "model_config_name": args.summary_model_config,
            "pass": "summarization",
        },
        path=args.summary_model_config,
    )

    summary_costs = [row["cost"] for row in summary_results]
    summary_cost_total = sum(c["cost"] for c in summary_costs)
    logger.info(f"Pass 2 cost: ${summary_cost_total:.4f}")

    # ---- Save results ----
    if args.final_answer:
        if "schema_0" not in dataset.column_names:
            dataset = dataset.add_column(
                "schema_0",
                [[{"desc": "Whether the answer is correct.", "title": "Answer Correctness", "points": 1}]]
                * len(dataset),
            )
    elif "grading_guidelines" in dataset.column_names and "schema_0" not in dataset.column_names:
        dataset = dataset.add_column(
            "schema_0",
            [[{"desc": gg, "title": "Proof Grade", "points": 7}] for gg in dataset["grading_guidelines"]],
        )

    # model_solution is the summarized proof (what gets graded)
    dataset = dataset.add_column("model_solution", [row["response"] for row in summary_results])
    # also keep the raw reasoning trace for analysis
    dataset = dataset.add_column("reasoning_trace", reasoning_traces)

    # combine costs from both passes
    total_costs = []
    for gc, sc in zip(gen_costs, summary_costs):
        total_costs.append({
            "cost": gc["cost"] + sc["cost"],
            "input_tokens": gc["input_tokens"] + sc["input_tokens"],
            "output_tokens": gc["output_tokens"] + sc["output_tokens"],
            "time": gc["time"] + sc["time"],
        })
    dataset = dataset.add_column("cost_run", total_costs)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_json(args.output_path)

    total_cost = sum(c["cost"] for c in total_costs)
    logger.info(f"Total cost (pass 1 + pass 2): ${total_cost:.4f}")
    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
