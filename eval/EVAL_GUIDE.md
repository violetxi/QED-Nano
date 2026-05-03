# Evaluation Guide: violetxi/exp_stage2_proof_qwen3-4b_instruct

## Prerequisites

- Python 3.12, uv
- GPU with enough VRAM for 4B model (single A100/H100 is fine)
- `OPENAI_API_KEY` set in your environment (for grading)
- vLLM installed (`uv pip install vllm`)

## 1. Serve the model with vLLM

```bash
VLLM_API_KEY=token-abc123 vllm serve violetxi/exp_stage2_proof_qwen3-4b_instruct \
  --port 8000 \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.8 \
  --data-parallel-size $(nvidia-smi -L | wc -l)
```

Adjust `--max-model-len` based on your GPU memory. Keep this running in a separate terminal.

## 2. Generation

All commands below should be run from the `eval/` directory:

```bash
cd eval
export VLLM_API_KEY=token-abc123
```

### IMOProofBench

```bash
uv run python scripts/run.py \
  --model-config vllm/vllm-violetxi-stage2-qwen3-4b \
  --output-path outputs/stage2-qwen3-4b-imoproofbench.jsonl
```

### ProofBench

```bash
uv run python scripts/run.py \
  --model-config vllm/vllm-violetxi-stage2-qwen3-4b \
  --data-path lm-provers/ProofBench \
  --output-path outputs/stage2-qwen3-4b-proofbench.jsonl
```

### Options

- `--n N` — run N times over the dataset (for pass@n)
- `--overwrite` — ignore cached intermediate results and re-run

## 3. Grading (OpenAI judge)

The `--model-config` here is the **judge** model, not your model.

### IMOProofBench

```bash
uv run python scripts/eval.py \
  --model-config google/gemini-3-pro \
  --data-path outputs/stage2-qwen3-4b-imoproofbench.jsonl \
  --output-path outputs/stage2-qwen3-4b-imoproofbench-graded.jsonl
```

### ProofBench

```bash
uv run python scripts/eval.py \
  --model-config openai/gpt-51 \
  --data-path outputs/stage2-qwen3-4b-proofbench.jsonl \
  --output-path outputs/stage2-qwen3-4b-proofbench-graded.jsonl \
  --proofbench
```

The `--proofbench` flag switches to the ProofBench-specific grading prompt and uses the `grading_scheme` column.

### Other judge options

Any OpenAI config under `configs/models/openai/` works. Examples:
- `openai/gpt-51` — GPT-5.1 (high reasoning)
- `openai/gpt-5-mini` — GPT-5 Mini (cheaper)
- `openai/oss-120b` — GPT-OSS-120B

## 4. Results

```bash
uv run python scripts/stats.py outputs/stage2-qwen3-4b-imoproofbench-graded.jsonl
uv run python scripts/stats.py outputs/stage2-qwen3-4b-proofbench-graded.jsonl
```

## 5. SLURM (if on a cluster)

Single command that chains inference + grading:

```bash
bash slurm/queue_launch_imobench.sh \
  --model violetxi/exp_stage2_proof_qwen3-4b_instruct \
  --judge openai/gpt-51
```

## Notes

- Intermediate results are cached (MD5-based). Interrupted runs resume automatically.
- IMOProofBench scores on a 0-7 scale per problem.
- ProofBench also scores 0-7 but uses a different grading prompt and `grading_scheme` column.
- vLLM serves an OpenAI-compatible API at `localhost:8000/v1`.
