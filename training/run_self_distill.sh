#!/bin/bash
# Train Qwen3-4B-Instruct-2507 with OpenRouter gpt-oss-20b as the LLM judge.

set -euo pipefail

# Activate the qed conda environment
eval "$(conda shell.bash hook)"
conda activate qed

# Remove nvhpc CUDA compat dir from LD_LIBRARY_PATH to avoid loading a stale
# libcuda.so (555.42.02) that conflicts with the installed driver (565.57.01).
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda/12.5/compat' | paste -sd ':')

export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before running}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

python -m pipelinerl.launch \
  --config-name=self_distill_cont \
  wandb.wandb_project_name=prl-proof-qwen4b-instruct \
  output_dir=results/self_distill_cont
