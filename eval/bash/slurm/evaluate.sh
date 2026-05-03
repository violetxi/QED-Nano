##### Dense Process Rubric #####
# # IMOProofBench grading
# uv run python scripts/eval.py \
#   --model-config google/gemini-3.1-flash-medium \
#   --data-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-imoproofbench-summary.jsonl \
#   --output-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-imoproofbench-summary-graded.jsonl

# IMOProofBench stats
uv run python scripts/stats.py outputs/stage1_proof-qwen3-4b-dense-process-rubric-imoproofbench-summary-graded.jsonl

# ProofBench grading
uv run python scripts/eval.py \
  --model-config google/gemini-3.1-flash-medium \
  --data-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-proofbench-summary.jsonl \
  --output-path outputs/stage1_proof-qwen3-4b-dense-process-rubric-proofbench-summary-graded.jsonl \
  --proofbench

# ProofBench stats
uv run python scripts/stats.py outputs/stage1_proof-qwen3-4b-dense-process-rubric-proofbench-summary-graded.jsonl