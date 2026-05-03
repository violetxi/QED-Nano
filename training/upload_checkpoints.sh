#!/bin/bash
#SBATCH --job-name=upload_checkpoints
#SBATCH --account=ingrai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# envs
source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
conda activate qed

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

JOB_WORKING_DIR="/hai/scratch/ziyxiang/QED-Nano/training"
cd "$JOB_WORKING_DIR"

# # Base (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/20260312-233128/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_instruct \
# --main_checkpoint current

# # Dense Process (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/dense_process/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_dense_process \
# --main_checkpoint current

### Upload to a separate repo bc previous checkpoints were lost but uploaded already
# # Self-Distillation (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/self_distill_cont/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_self_distill_cont \
# --main_checkpoint current

# # SFT (Uploaded at ckpt-392)
# python upload_checkpoints.py \
# --checkpoint_dir results/sft/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_sft \
# --main_checkpoint current

# # GRPO (Uploaded at ckpt-310)
# python upload_checkpoints.py \
# --checkpoint_dir results/grpo/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_grpo \
# --main_checkpoint current

# # Dense Outcome (Uploaded at ckpt-383)
# python upload_checkpoints.py \
# --checkpoint_dir results/dense_outcome/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_dense_outcome \
# --main_checkpoint current


##### Proof Checkpoints #####
python upload_checkpoints.py \
--checkpoint_dir results/grpo_10a2f/finetune \
--hf_repo_id violetxi/proof_stage1_qwen3-4b_grpo \
--main_checkpoint current