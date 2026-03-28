#!/bin/bash
#SBATCH --job-name=exp_rl
#SBATCH -p batch
#SBATCH -A marlowe-m000237-free
#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dev_node_%j.out
#SBATCH --error=logs/dev_node_%j.err

mkdir -p logs

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Started: $(date)"

sleep infinity
