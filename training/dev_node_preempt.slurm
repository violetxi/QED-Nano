#!/bin/bash
#SBATCH --job-name=exp_rl
#SBATCH -p preempt
#SBATCH -A marlowe-m000237
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/dev_node_%j.out
#SBATCH --error=logs/dev_node_%j.err

mkdir -p logs

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Started: $(date)"

sleep infinity
