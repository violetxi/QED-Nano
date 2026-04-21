#!/bin/bash
# Train Qwen3-4B-Instruct-2507 with OpenRouter gpt-oss-20b as the LLM judge.

set -euo pipefail

# Marlowe nodes: strip NVHPC CUDA compat libs from LD_LIBRARY_PATH — they
# override the system driver's libcuda.so and cause CUDA error 803.
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/.*/compat' | paste -sd ':')

export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

python -m pipelinerl.launch \
  --config-name=dense_outcome_cont \
  wandb.wandb_project_name=prl-proof-qwen4b-instruct \
  output_dir=results/dense_outcome \
  world.actor_start_port=18080
