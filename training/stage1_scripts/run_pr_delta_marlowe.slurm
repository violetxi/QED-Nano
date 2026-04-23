#!/bin/bash
#SBATCH --job-name=pr_delta_1a1f_clip
#SBATCH -p preempt
#SBATCH -A marlowe-m000237
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

CONDA_BASE="/projects/m000237/ziyxiang/miniconda3"
export PATH="$CONDA_BASE/bin:$CONDA_BASE/condabin:/cm/shared/apps/slurm/current/sbin:/cm/shared/apps/slurm/current/bin:$PATH"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "qed"; then
  conda activate qed
elif [[ -f "qed-train/bin/activate" ]]; then
  source qed-train/bin/activate
else
  echo "Could not find a usable Python environment (expected conda env 'qed' or qed-train/bin/activate)." >&2
  exit 1
fi

JOB_WORKING_DIR="/scratch/m000237/ziyxiang/QED-Nano/training"
cd "$JOB_WORKING_DIR"

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment before sbatch}"

export WORLD_SIZE=$SLURM_NTASKS
export HYDRA_FULL_ERROR=1
export MASTER_PORT=6390

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
HOSTLIST=$(printf '%s\n' "${nodes_array[@]}" | paste -sd, -)

declare -A IP_OF
while read -r host ip; do
  [[ -n "$host" ]] && IP_OF["$host"]="$ip"
done < <(
  srun -w "$HOSTLIST" --ntasks-per-node=1 bash -c '
    h=$(hostname -s)
    ip=$(hostname -I | tr " " "\n" | grep -m1 -E "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
    echo "$h $ip"
  '
)

ip_addr=()
for h in "${nodes_array[@]}"; do
  ip_addr+=("${IP_OF[$h]}")
done

export ALL_ADDR="$(IFS=,; echo "${ip_addr[*]}")"
if ((${#nodes_array[@]} > 0)); then
  export MASTER_ADDR="${ip_addr[0]}"
else
  echo "nodes_array is empty; cannot set MASTER_ADDR" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module unload cudatoolkit/12.5 gcc/13.1.0 nvhpc/24.7 >/dev/null 2>&1 || true
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "Nodes allocated: ${nodes_array[@]}"
for i in "${!nodes_array[@]}"; do
  printf "%s %s\n" "${nodes_array[$i]}" "${ip_addr[$i]}"
done
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "Job ID: $SLURM_JOB_ID  GPUs/node: ${SLURM_GPUS_ON_NODE:-2}  CPUs/task: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "--------------------"

nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

srun -w "$HOSTLIST" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    CONDA_BASE="/projects/m000237/ziyxiang/miniconda3"
    export PATH="$CONDA_BASE/bin:$CONDA_BASE/condabin:/cm/shared/apps/slurm/current/sbin:/cm/shared/apps/slurm/current/bin:$PATH"
    if command -v module >/dev/null 2>&1; then
      module unload cudatoolkit/12.5 gcc/13.1.0 nvhpc/24.7 >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
    fi
    if command -v conda >/dev/null 2>&1 && conda env list | awk '"'"'{print $1}'"'"' | grep -qx "qed"; then
      conda activate qed
    elif [[ -f "'"$JOB_WORKING_DIR"'/qed-train/bin/activate" ]]; then
      source "'"$JOB_WORKING_DIR"'/qed-train/bin/activate"
    else
      echo "Could not find a usable Python environment (expected conda env '\''qed'\'' or qed-train/bin/activate)." >&2
      exit 1
    fi
    export RANK=$SLURM_NODEID
    export WORLD_SIZE='"$WORLD_SIZE"'
    export ALL_ADDR='"$ALL_ADDR"'
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export HYDRA_FULL_ERROR=1
    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ALL_ADDR=$ALL_ADDR"
    cd '"$JOB_WORKING_DIR"'
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    python -c '"'"'import torch; print(f"torch={torch.__version__} cuda_build={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}"); assert torch.cuda.is_available(), "CUDA is not available inside srun job step"'"'"'
    python -m pipelinerl.launch \
      --config-name=exp_rl_pr_stage1 \
      wandb.wandb_project_name=prl-proof-qwen4b-gemini-stage1 \
      output_dir=results/pr_delta_6a2f \
      world.actor_start_port=18120 \
      world.actor_fraction=6 \
      world.finetune_fraction=2 \
      finetune.seq_length=24576 \
      finetune.seq_parallel=2 \
      finetune.save_checkpoint_steps=1 \
      process_grader.is_clip_length=true
  '
