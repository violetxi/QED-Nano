#!/bin/bash
#SBATCH --job-name=dense_process_stage2
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=168:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --requeue

# envs
source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
conda activate qed

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

JOB_WORKING_DIR="/hai/scratch/ziyxiang/QED-Nano/training"
cd "$JOB_WORKING_DIR"

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in ~/.bashrc}"

# Discover node list / IPs
export WORLD_SIZE=$SLURM_NTASKS
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
  export MASTER_PORT=6379
else
  echo "nodes_array is empty; cannot set MASTER_ADDR" >&2
  exit 1
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "Nodes allocated: ${nodes_array[@]}"
for i in "${!nodes_array[@]}"; do
  printf "%s %s\n" "${nodes_array[$i]}" "${ip_addr[$i]}"
done
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "Job ID: $SLURM_JOB_ID  GPUs/node: $SLURM_GPUS_ON_NODE  CPUs/task: $SLURM_CPUS_PER_TASK"
echo "--------------------"

srun -w "$HOSTLIST" --ntasks-per-node=1 \
  bash -lc '
    source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
    conda activate qed
    unset ROCR_VISIBLE_DEVICES
    export RANK=$SLURM_NODEID
    export WORLD_SIZE='"$WORLD_SIZE"'
    export ALL_ADDR='"$ALL_ADDR"'
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export HYDRA_FULL_ERROR=1
    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ALL_ADDR=$ALL_ADDR"
    cd '"$JOB_WORKING_DIR"'
    python -m pipelinerl.launch \
      --config-name=dense_process_stage2 \
      wandb.wandb_project_name=prl-proof-qwen4b-gemini-stage2 \
      output_dir=results/dense_process_10a2f \
      world.actor_start_port=18080 \
      world.actor_fraction=1 \
      world.finetune_fraction=1 \
      finetune.seq_length=24576 \
      finetune.seq_parallel=1
  '
