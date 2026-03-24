# Base (Done!)
python upload_checkpoints.py \
--checkpoint_dir results/20260312-233128/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_instruct \
--main_checkpoint current

# # SFT 
# python upload_checkpoints.py \
# --checkpoint_dir results/sft/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_sft \
# --main_checkpoint current

# Self-Distillation (Uploaded at ckpt-271)
python upload_checkpoints.py \
--checkpoint_dir results/self_distill/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_self_distill \
--main_checkpoint current

# # GRPO
# python upload_checkpoints.py \
# --checkpoint_dir results/prl-exp-stage2-grpo-16k/finetune/ \
# --hf_repo_id violetxi/exp_stage2_qwen3-4b_grpo \
# --main_checkpoint current

# # Dense Outcome
# python upload_checkpoints.py \
# --checkpoint_dir results/dense-outcome-16k/finetune/ \
# --hf_repo_id violetxi/exp_stage2_qwen3-4b_dense_outcome \
# --main_checkpoint current

# # Dense Process
# python upload_checkpoints.py \
# --checkpoint_dir results/dense-process-16k/finetune/ \
# --hf_repo_id violetxi/exp_stage2_qwen3-4b_dense_process \
# --main_checkpoint current