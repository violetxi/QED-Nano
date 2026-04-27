#!/bin/bash
# === Manual two-step eval (generate + summarize): stage1 dense_process ===
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
  --model violetxi/exp_stage1_qwen3-4b_pr_delta_process \
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

# # 2a: IMOProofBench (generate + summarize)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process \
#   --output-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-summary.jsonl \
#   --overwrite \
#   --n 16

# # 2b: Grade summarized IMOProofBench
# uv run python scripts/eval.py \
#   --model-config google/gemini-3-pro \
#   --data-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-summary.jsonl \
#   --output-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-summary-graded.jsonl

# 2c: IMOProofBench stats
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-process-imoproofbench-summary-graded.jsonl

# # 2d: ProofBench (generate + summarize)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process \
#   --data-path lm-provers/ProofBench \
#   --output-path outputs/stage1-qwen3-4b-dense-process-proofbench-summary.jsonl \
#   --overwrite \
#   --n 16

# # 2e: Grade summarized ProofBench
# uv run python scripts/eval.py \
#   --model-config google/gemini-3-pro \
#   --data-path outputs/stage1-qwen3-4b-dense-process-proofbench-summary.jsonl \
#   --output-path outputs/stage1-qwen3-4b-dense-process-proofbench-summary-graded.jsonl \
#   --proofbench

# 2f: ProofBench stats
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-process-proofbench-summary-graded.jsonl

# ============================================================
# 128 samples, 16k response length
# ============================================================

# # 4a: IMOProofBench (generate + summarize, n128)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process \
#   --output-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-n128-summary.jsonl \
#   --overwrite \
#   --n 128

# 4b: Grade summarized IMOProofBench (n128)
uv run python scripts/eval.py \
  --model-config google/gemini-3-pro \
  --data-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-n128-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-process-imoproofbench-n128-summary-graded.jsonl

# 4c: IMOProofBench stats (n128)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-process-imoproofbench-n128-summary-graded.jsonl

# # 4d: ProofBench (generate + summarize, n128)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process \
#   --data-path lm-provers/ProofBench \
#   --output-path outputs/stage1-qwen3-4b-dense-process-proofbench-n128-summary.jsonl \
#   --overwrite \
#   --n 128

# 4e: Grade summarized ProofBench (n128)
uv run python scripts/eval.py \
  --model-config google/gemini-3-pro \
  --data-path outputs/stage1-qwen3-4b-dense-process-proofbench-n128-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-process-proofbench-n128-summary-graded.jsonl \
  --proofbench

# 4f: ProofBench stats (n128)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-process-proofbench-n128-summary-graded.jsonl

# Step 6: Kill the vLLM server (Ctrl+C in pane 1, or:)
# pkill -f "vllm.entrypoints.openai.api_server --model violetxi/exp_stage1_qwen3-4b_pr_delta_process"
