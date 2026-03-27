#!/bin/bash
set -e

MODEL="violetxi/exp_stage2_proof_qwen3-4b_self_distill"
CONFIG="vllm/vllm-violetxi-stage2-qwen3-4b-self-distill"
OUTPUT_PREFIX="outputs/stage2-qwen3-4b-self-distill"
PORT=8000

export VLLM_API_KEY=token-abc123
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export UV_PYTHON=3.13

cleanup() {
  echo "=== Stopping vLLM server ==="
  if [ -n "${VLLM_PID:-}" ]; then
    kill -- -${VLLM_PID} 2>/dev/null || true
    sleep 2
    kill -9 -- -${VLLM_PID} 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.openai.api_server --model ${MODEL}" 2>/dev/null || true
  sleep 1
  pkill -9 -f "vllm.entrypoints.openai.api_server --model ${MODEL}" 2>/dev/null || true
}
trap cleanup EXIT

# Kill any existing vLLM on this port
echo "=== Cleaning up any existing vLLM on port ${PORT} ==="
pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
sleep 3

echo "=== Starting vLLM server for ${MODEL} ==="
setsid python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --dtype bfloat16 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.8 \
  --data-parallel-size 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 40000 \
  --max-num-seqs 2048 \
  --api-key "${VLLM_API_KEY}" > /tmp/vllm_${OUTPUT_PREFIX//\//_}.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID: ${VLLM_PID}) to be ready..."
SERVER_READY=false
for i in $(seq 1 300); do
  if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" http://localhost:${PORT}/v1/models | grep -q "${MODEL}"; then
    echo "Server ready after ${i}s"
    SERVER_READY=true
    break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "vLLM server died. Log:"
    tail -30 /tmp/vllm_${OUTPUT_PREFIX//\//_}.log
    exit 1
  fi
  sleep 1
done

if [ "${SERVER_READY}" = false ]; then
  echo "vLLM server failed to start within 300s. Log:"
  tail -30 /tmp/vllm_${OUTPUT_PREFIX//\//_}.log
  exit 1
fi

echo "=== Running IMOProofBench generation ==="
uv run python scripts/run.py \
  --model-config "${CONFIG}" \
  --output-path "${OUTPUT_PREFIX}-imoproofbench.jsonl" \
  --overwrite \
  --n 4

echo "=== Grading IMOProofBench ==="
uv run python scripts/eval.py \
  --model-config openai/gpt-51 \
  --data-path "${OUTPUT_PREFIX}-imoproofbench.jsonl" \
  --output-path "${OUTPUT_PREFIX}-imoproofbench-graded.jsonl"

echo "=== IMOProofBench Stats ==="
uv run python scripts/stats.py "${OUTPUT_PREFIX}-imoproofbench-graded.jsonl"

echo "=== Running ProofBench generation ==="
uv run python scripts/run.py \
  --model-config "${CONFIG}" \
  --data-path lm-provers/ProofBench \
  --output-path "${OUTPUT_PREFIX}-proofbench.jsonl" \
  --overwrite \
  --n 4

echo "=== Grading ProofBench ==="
uv run python scripts/eval.py \
  --model-config openai/gpt-51 \
  --data-path "${OUTPUT_PREFIX}-proofbench.jsonl" \
  --output-path "${OUTPUT_PREFIX}-proofbench-graded.jsonl" \
  --proofbench

echo "=== ProofBench Stats ==="
uv run python scripts/stats.py "${OUTPUT_PREFIX}-proofbench-graded.jsonl"

echo "Done: ${MODEL}"
