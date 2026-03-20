#!/bin/bash
# Train Qwen3-4B-Instruct-2507 with OpenRouter gpt-oss-20b as the LLM judge.

set -euo pipefail

export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

python -m pipelinerl.launch \
  --config-name=dense_outcome \
  wandb.wandb_project_name=prl-proof-qwen4b-instruct \
  output_dir=results/dense_outcome
