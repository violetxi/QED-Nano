#!/bin/bash
# === Manual two-step eval (generate + summarize): stage1 dense_outcome ===
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
  --model violetxi/exp_stage1_qwen3-4b_dense_outcome \
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

# 2a: IMOProofBench (generate + summarize)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary.jsonl \
  --overwrite \
  --n 16

# 2b: Grade summarized IMOProofBench
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-graded.jsonl

# 2c: IMOProofBench stats
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-graded.jsonl

# 2d: ProofBench (generate + summarize)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary.jsonl \
  --overwrite \
  --n 16

# 2e: Grade summarized ProofBench
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-graded.jsonl \
  --proofbench

# 2f: ProofBench stats
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-graded.jsonl

# ============================================================
# 24k response length
# ============================================================

# # 3a: IMOProofBench (generate + summarize, 24k)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome-24k \
#   --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-24k.jsonl \
#   --overwrite \
#   --n 16 \
#   --summary-max-tokens 16384

# # 3b: Grade summarized IMOProofBench (24k)
# uv run python scripts/eval.py \
#   --model-config openai/gpt-5-nano \
#   --data-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-24k.jsonl \
#   --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-24k-graded.jsonl

# 3c: IMOProofBench stats (24k)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-summary-24k-graded.jsonl

# 3d: ProofBench (generate + summarize, 24k)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome-24k \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-24k.jsonl \
  --overwrite \
  --n 16 \
  --summary-max-tokens 16384

# 3e: Grade summarized ProofBench (24k)
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-24k.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-24k-graded.jsonl \
  --proofbench

# 3f: ProofBench stats (24k)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-proofbench-summary-24k-graded.jsonl

# ============================================================
# 128 samples, 16k response length
# ============================================================

# 4a: IMOProofBench (generate + summarize, n128)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary.jsonl \
  --overwrite \
  --n 128

# 4b: Grade summarized IMOProofBench (n128)
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-graded.jsonl

# 4c: IMOProofBench stats (n128)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-graded.jsonl

# 4d: ProofBench (generate + summarize, n128)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary.jsonl \
  --overwrite \
  --n 128

# 4e: Grade summarized ProofBench (n128)
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-graded.jsonl \
  --proofbench

# 4f: ProofBench stats (n128)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-graded.jsonl

# ============================================================
# 128 samples, 24k response length
# ============================================================

# 5a: IMOProofBench (generate + summarize, n128, 24k)
# uv run python scripts/run_summary.py \
#   --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome-24k \
#   --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-24k.jsonl \
#   --overwrite \
#   --n 128 \
#   --summary-max-tokens 16384

# 5b: Grade summarized IMOProofBench (n128, 24k)
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-24k.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-24k-graded.jsonl

# 5c: IMOProofBench stats (n128, 24k)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-imoproofbench-n128-summary-24k-graded.jsonl

# 5d: ProofBench (generate + summarize, n128, 24k)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-outcome-24k \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-24k.jsonl \
  --overwrite \
  --n 128 \
  --summary-max-tokens 16384

# 5e: Grade summarized ProofBench (n128, 24k)
uv run python scripts/eval.py \
  --model-config openai/gpt-5-nano \
  --data-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-24k.jsonl \
  --output-path outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-24k-graded.jsonl \
  --proofbench

# 5f: ProofBench stats (n128, 24k)
uv run python scripts/stats.py outputs/stage1-qwen3-4b-dense-outcome-proofbench-n128-summary-24k-graded.jsonl

# Step 6: Kill the vLLM server (Ctrl+C in pane 1, or:)
# pkill -f "vllm.entrypoints.openai.api_server --model violetxi/exp_stage1_qwen3-4b_dense_outcome"
