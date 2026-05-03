#!/usr/bin/env bash

# python notebook/annotate_proof_techniques_async.py \
#   --data-path violetxi/qwen3-4b-instruct-imoproofbench-summary-graded \
#   --solution-column reasoning_trace \
#   --output-path notebook/reasoning_trace_annotations/qwen3-4b-instruct-imoproofbench-summary-graded-techniques.jsonl \
#   --concurrency 32 \
#   --api-key-env GEMINI_API_KEY

# python notebook/annotate_proof_techniques_async.py \
#   --data-path violetxi/stage1_proof-qwen3-4b-grpo-imoproofbench-summary-graded \
#   --solution-column reasoning_trace \
#   --output-path notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-grpo-imoproofbench-summary-graded-techniques.jsonl \
#   --concurrency 32 \
#   --api-key-env GEMINI_API_KEY

# python notebook/annotate_proof_techniques_async.py \
#   --data-path violetxi/stage1_proof-qwen3-4b-dense-process-imoproofbench-summary-graded \
#   --solution-column reasoning_trace \
#   --output-path notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-dense-process-imoproofbench-summary-graded-techniques.jsonl \
#   --concurrency 32 \
#   --api-key-env GEMINI_API_KEY

python notebook/annotate_proof_techniques_async.py \
  --data-path violetxi/stage1_proof-qwen3-4b-sft-imoproofbench-summary-graded \
  --solution-column reasoning_trace \
  --output-path notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-sft-imoproofbench-summary-graded-techniques.jsonl \
  --concurrency 32 \
  --api-key-env GEMINI_API_KEY

python notebook/annotate_proof_techniques_async.py \
  --data-path violetxi/stage1_proof-qwen3-4b-self-distill-imoproofbench-summary-graded \
  --solution-column reasoning_trace \
  --output-path notebook/reasoning_trace_annotations/stage1_proof-qwen3-4b-self-distill-imoproofbench-summary-graded-techniques.jsonl \
  --concurrency 32 \
  --api-key-env GEMINI_API_KEY
