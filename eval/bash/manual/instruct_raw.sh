#!/bin/bash
# === Manual eval: instruct ===
# Run from the eval/ directory: cd /hai/scratch/ziyxiang/QED-Nano/eval
# Activate env first: conda activate qed

# Step 0: Env vars (run once per shell session)
export VLLM_API_KEY=token-abc123
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export UV_PYTHON=3.13
# Remove nvhpc compat path from LD_LIBRARY_PATH (ONLY ON MARLOWE for hosting vLLM server)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda/12.5/compat' | tr '\n' ':' | sed 's/:$//')
export CC=/cm/local/apps/gcc/13.1.0/bin/gcc

# Step 1: Start vLLM server (runs in foreground — use a separate tmux pane)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.8 \
  --data-parallel-size 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 40000 \
  --max-num-seqs 2048 \
  --api-key token-abc123

# Step 2: In another pane, run these one by one (from eval/ directory)

# 2a: IMOProofBench generation
uv run python scripts/run.py \
  --model-config vllm/vllm-qwen-qwen3-4b-instruct-4b-2507 \
  --output-path outputs/stage2-qwen3-4b-instruct-raw-imoproofbench.jsonl \
  --overwrite \
  --n 32

# 2b: Grade IMOProofBench
uv run python scripts/eval.py \
  --model-config  openai/gpt-5-nano \
  --data-path outputs/stage2-qwen3-4b-instruct-raw-imoproofbench.jsonl \
  --output-path outputs/stage2-qwen3-4b-instruct-raw-imoproofbench-graded.jsonl

# 2c: IMOProofBench stats
uv run python scripts/stats.py outputs/stage2-qwen3-4b-instruct-raw-imoproofbench-graded.jsonl

# # 2d: ProofBench generation
uv run python scripts/run.py \
  --model-config vllm/vllm-qwen-qwen3-4b-instruct-4b-2507 \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage2-qwen3-4b-instruct-raw-proofbench.jsonl \
  --overwrite \
  --n 32

# 2e: Grade ProofBench
uv run python scripts/eval.py \
  --model-config  openai/gpt-5-nano \
  --data-path outputs/stage2-qwen3-4b-instruct-raw-proofbench.jsonl \
  --output-path outputs/stage2-qwen3-4b-instruct-raw-proofbench-graded.jsonl \
  --proofbench

# 2f: ProofBench stats
uv run python scripts/stats.py outputs/stage2-qwen3-4b-instruct-raw-proofbench-graded.jsonl

# Step 3: Kill the vLLM server (Ctrl+C in pane 1, or:)
pkill -f "vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B-Instruct-2507"
