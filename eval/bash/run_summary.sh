#!/bin/bash
# Run summarization (pass 2 only) for all existing generation outputs.
# Each model block: start vLLM, summarize IMOProofBench + ProofBench, stop vLLM.
set -e

PORT=8000

export VLLM_API_KEY=token-abc123
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export UV_PYTHON=3.13

cd "$(dirname "$0")/.."

start_vllm() {
  local model=$1
  local log_tag=$2

  pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  sleep 3

  echo "=== Starting vLLM server for ${model} ==="
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 49152 \
    --gpu-memory-utilization 0.8 \
    --data-parallel-size 8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 40000 \
    --max-num-seqs 2048 \
    --api-key "${VLLM_API_KEY}" > /tmp/vllm_summary_${log_tag}.log 2>&1 &
  VLLM_PID=$!

  echo "Waiting for vLLM server (PID: ${VLLM_PID}) to be ready..."
  for i in $(seq 1 300); do
    if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" http://localhost:${PORT}/v1/models | grep -q "${model}"; then
      echo "Server ready after ${i}s"
      return 0
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
      echo "vLLM server died. Log:"
      tail -30 /tmp/vllm_summary_${log_tag}.log
      return 1
    fi
    sleep 1
  done
  echo "vLLM server failed to start within 300s."
  tail -30 /tmp/vllm_summary_${log_tag}.log
  return 1
}

stop_vllm() {
  echo "=== Stopping vLLM server ==="
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  sleep 2
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  sleep 3
}

# ============================================================
# instruct
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_instruct"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b"
PREFIX="outputs/stage2-qwen3-4b-instruct"

start_vllm "${MODEL}" "instruct"

echo "=== Summarizing instruct / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

echo "=== Summarizing instruct / ProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-proofbench.jsonl" \
  --output-path "${PREFIX}-proofbench-summary.jsonl"

stop_vllm

# ============================================================
# sft
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_sft"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-sft"
PREFIX="outputs/stage2-qwen3-4b-sft"

start_vllm "${MODEL}" "sft"

echo "=== Summarizing sft / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

echo "=== Summarizing sft / ProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-proofbench.jsonl" \
  --output-path "${PREFIX}-proofbench-summary.jsonl"

stop_vllm

# ============================================================
# grpo (IMOProofBench only)
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_grpo"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-grpo"
PREFIX="outputs/stage2-qwen3-4b-grpo"

start_vllm "${MODEL}" "grpo"

echo "=== Summarizing grpo / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

stop_vllm

# ============================================================
# dense-outcome
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_dense_outcome"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-dense-outcome"
PREFIX="outputs/stage2-qwen3-4b-dense-outcome"

start_vllm "${MODEL}" "dense-outcome"

echo "=== Summarizing dense-outcome / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

echo "=== Summarizing dense-outcome / ProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-proofbench.jsonl" \
  --output-path "${PREFIX}-proofbench-summary.jsonl"

stop_vllm

# ============================================================
# dense-process
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_dense_process"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-dense-process"
PREFIX="outputs/stage2-qwen3-4b-dense-process"

start_vllm "${MODEL}" "dense-process"

echo "=== Summarizing dense-process / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

echo "=== Summarizing dense-process / ProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-proofbench.jsonl" \
  --output-path "${PREFIX}-proofbench-summary.jsonl"

stop_vllm

# ============================================================
# self-distill
# ============================================================
MODEL="violetxi/exp_stage2_proof_qwen3-4b_self_distill"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-self-distill"
PREFIX="outputs/stage2-qwen3-4b-self-distill"

start_vllm "${MODEL}" "self-distill"

echo "=== Summarizing self-distill / IMOProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-imoproofbench.jsonl" \
  --output-path "${PREFIX}-imoproofbench-summary.jsonl"

echo "=== Summarizing self-distill / ProofBench ==="
uv run python scripts/run_summary.py \
  --model-config "${CONFIG}" \
  --generation-path "${PREFIX}-proofbench.jsonl" \
  --output-path "${PREFIX}-proofbench-summary.jsonl"

stop_vllm

echo "=== All summarization runs complete ==="
