#!/bin/bash
# === Manual summary eval: grpo ===
# Run from the eval/ directory: cd /hai/scratch/ziyxiang/QED-Nano/eval
# Activate env first: conda activate qed

# Step 0: Env vars (run once per shell session)
export VLLM_API_KEY=token-abc123
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export UV_PYTHON=3.13

# Step 1: Start vLLM server (runs in foreground — use a separate tmux pane)
python -m vllm.entrypoints.openai.api_server \
  --model violetxi/exp_stage2_proof_qwen3-4b_grpo \
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

# 2a: Summarize IMOProofBench
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage2-qwen3-4b-grpo \
  --generation-path outputs/stage2-qwen3-4b-grpo-imoproofbench.jsonl \
  --output-path outputs/stage2-qwen3-4b-grpo-imoproofbench-summary.jsonl

# 2b: Grade summarized IMOProofBench
uv run python scripts/eval.py \
  --model-config google/gemini-3-pro \
  --data-path outputs/stage2-qwen3-4b-grpo-imoproofbench-summary.jsonl \
  --output-path outputs/stage2-qwen3-4b-grpo-imoproofbench-summary-graded.jsonl

# 2c: Stats
uv run python scripts/stats.py outputs/stage2-qwen3-4b-grpo-imoproofbench-summary-graded.jsonl

# Step 3: Kill the vLLM server (Ctrl+C in pane 1, or:)
# pkill -f "vllm.entrypoints.openai.api_server --model violetxi/exp_stage2_proof_qwen3-4b_grpo"
