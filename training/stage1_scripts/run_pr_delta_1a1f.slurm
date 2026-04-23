#!/bin/bash
#SBATCH --job-name=pr_delta_1a1f_clip
#SBATCH --account=ingrai
#SBATCH --partition=hai-lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
conda activate qed

unset ROCR_VISIBLE_DEVICES

JOB_WORKING_DIR="/hai/scratch/ziyxiang/QED-Nano/training"
cd "$JOB_WORKING_DIR"

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in ~/.bashrc}"

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname -I | tr " " "\n" | grep -m1 -E "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
export MASTER_PORT=6390
export ALL_ADDR="$MASTER_ADDR"
export RANK=$SLURM_NODEID
export HYDRA_FULL_ERROR=1

echo "WORLD_SIZE=$WORLD_SIZE"
echo "HOST=$(hostname -s)"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "Job ID: $SLURM_JOB_ID  GPUs/node: $SLURM_GPUS_ON_NODE  CPUs/task: $SLURM_CPUS_PER_TASK"
echo "--------------------"

srun --ntasks-per-node=1 \
  bash -lc '
    source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
    conda activate qed
    unset ROCR_VISIBLE_DEVICES
    export RANK='"$RANK"'
    export WORLD_SIZE='"$WORLD_SIZE"'
    export ALL_ADDR='"$ALL_ADDR"'
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export HYDRA_FULL_ERROR=1
    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ALL_ADDR=$ALL_ADDR"
    cd '"$JOB_WORKING_DIR"'
    python -m pipelinerl.launch \
      --config-name=exp_rl_pr_stage1 \
      wandb.use_wandb=true \
      llm.wandb_table.enabled=false \
      llm_grader.wandb_table.enabled=false \
      force_restart=true \
      output_dir=results/pr_delta_1a1f_step5_eval_clip \
      world.actor_start_port=18120 \
      world.actor_fraction=1 \
      world.finetune_fraction=1 \
      world.preprocessor_fraction=0 \
      eval_every_n_versions=1 \
      skip_first_eval=true \
      finetune.seq_parallel=1 \
      finetune.train_batch_size=2 \
      finetune.gradient_accumulation_passes=1 \
      finetune.max_train_steps=5 \
      finetune.interrupt_train_steps=5 \
      finetune.save_checkpoint_steps=1000 \
      finetune.attempts=4 \
      finetune.seq_length=8192 \
      actor.llm_max_rollouts=128 \
      llm.parameters.max_tokens=4096 \
      test_llm.parameters.max_tokens=1024 \
      process_grader.is_clip_length=true \
      process_grader.max_concurrency=8 \
      process_grader.sampling_kwargs.max_output_tokens=4096 \
      process_grader.sampling_kwargs.thinking_level=medium
  '
