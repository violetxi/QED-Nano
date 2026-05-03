#!/bin/bash
#SBATCH --job-name=generate_pr_rubric
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --requeue

# envs
source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
conda activate qed

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

JOB_WORKING_DIR="/hai/scratch/ziyxiang/QED-Nano/eval"
cd "$JOB_WORKING_DIR"

# Step 0: Env vars (run once per shell session)
export VLLM_API_KEY=token-abc123
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export UV_PYTHON=3.13
# Remove nvhpc compat path from LD_LIBRARY_PATH (ONLY ON MARLOWE for hosting vLLM server)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda/12.5/compat' | tr '\n' ':' | sed 's/:$//')
export CC=/usr/bin/gcc

# Step 1: Start vLLM server in the background
python -m vllm.entrypoints.openai.api_server \
  --model violetxi/stage1_proof_pr_delta_rubric \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 40000 \
  --max-num-seqs 2048 \
  --api-key token-abc123 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# Wait for the server to come up (or die)
echo "Waiting for vLLM server on http://127.0.0.1:8000 ..."
until curl -s -f -H "Authorization: Bearer token-abc123" \
        http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server died before becoming ready" >&2
    exit 1
  fi
  sleep 10
done
echo "vLLM server is ready."

# Step 2: IMOProofBench (generate + summarize)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process-rubric \
  --output-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-imoproofbench-summary.jsonl \
  --overwrite \
  --n 16

# Step 3: ProofBench (generate + summarize)
uv run python scripts/run_summary.py \
  --model-config vllm/vllm-violetxi-stage1-qwen3-4b-dense-process-rubric \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-proofbench-summary.jsonl \
  --overwrite \
  --n 16