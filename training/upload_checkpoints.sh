# # Base (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/20260312-233128/finetune/ \
# --hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_instruct \
# --main_checkpoint current

# SFT (Uploaded at ckpt-327)
python upload_checkpoints.py \
--checkpoint_dir results/sft/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_sft \
--main_checkpoint current

# Self-Distillation (Uploaded at ckpt-305)
python upload_checkpoints.py \
--checkpoint_dir results/self_distill/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_self_distill \
--main_checkpoint current

# GRPO (Uploaded at ckpt-196)
python upload_checkpoints.py \
--checkpoint_dir results/grpo/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_grpo \
--main_checkpoint current

# Dense Outcome (Uploaded at ckpt-298)
python upload_checkpoints.py \
--checkpoint_dir results/dense_outcome/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_dense_outcome \
--main_checkpoint current

# Dense Process (Uploaded at ckpt-380)
python upload_checkpoints.py \
--checkpoint_dir results/dense_process/finetune/ \
--hf_repo_id violetxi/exp_stage2_proof_qwen3-4b_dense_process \
--main_checkpoint current