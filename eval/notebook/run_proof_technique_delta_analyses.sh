#!/usr/bin/env bash

##### Model solution annotations #####
python notebook/analyze_proof_technique_deltas.py \
  --use-default-model-paths \
  --output-dir notebook/proof_technique_delta_analysis/proofbench

python notebook/analyze_proof_technique_deltas.py \
  --input instruct=notebook/model_solution_annotations/qwen3-4b-instruct-imoproofbench-summary-graded-techniques.jsonl \
  --input grpo=notebook/model_solution_annotations/stage1_proof-qwen3-4b-grpo-imoproofbench-summary-graded-techniques.jsonl \
  --input dense_process=notebook/model_solution_annotations/stage1_proof-qwen3-4b-dense-process-imoproofbench-summary-graded-techniques.jsonl \
  --input sft=notebook/model_solution_annotations/stage1_proof-qwen3-4b-sft-imoproofbench-summary-graded-techniques.jsonl \
  --input self_distill=notebook/model_solution_annotations/stage1_proof-qwen3-4b-self-distill-imoproofbench-summary-graded-techniques.jsonl \
  --baseline instruct \
  --output-dir notebook/proof_technique_delta_analysis/imoproofbench

##### Reasoning trace annotations #####
python notebook/analyze_proof_technique_deltas.py \
  --input instruct=notebook/reasoning_trace_annotations/qwen3-4b-instruct-imoproofbench-summary-graded-techniques.jsonl \
  --input grpo=notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-grpo-imoproofbench-summary-graded-techniques.jsonl \
  --input dense_process=notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-dense-process-imoproofbench-summary-graded-techniques.jsonl \
  --input sft=notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-sft-imoproofbench-summary-graded-techniques.jsonl \
  --input self_distill=notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-self-distill-imoproofbench-summary-graded-techniques.jsonl \
  --baseline instruct \
  --output-dir notebook/proof_technique_delta_analysis/reasoning_trace