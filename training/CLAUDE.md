# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is the `training/` directory of **QED-Nano**, a 4B model for Olympiad-level proof generation. It is built on top of **PipelineRL** — a scalable asynchronous RL framework with in-flight weight updates. The package name is `pipelinerl`.

## Environment and install

Two install paths coexist — they use different env names and different Python/vLLM pins. Check which the current scripts expect before running anything.

- **conda (README):** env `pipeline-rl`, Python 3.11, `pip install torch==2.6.0 && pip install -e . --no-build-isolation`.
- **uv (`install.sh`):** creates a `uv` venv at `./qed-train` and installs `vllm==0.8.5.post1`.
- **Cluster scripts in `stage1_scripts/` currently use `conda activate qed`** (see `run_pr_delta.slurm`) — different from both above. Don't assume one env; read the launch script.
- Optional: `conda install redis-server==7.4.0 -c conda-forge` to use Redis instead of filesystem streams.

## Common commands

Launches always go through Hydra via `python -m pipelinerl.launch`. Configs live in `conf/`; pick one with `--config-name=<name>` (omit `.yaml`) and pass overrides as `key=value`.

```bash
# Default single-node run (8× H100 assumed)
python -m pipelinerl.launch output_dir=results/base1

# Named config (base.yaml is the default; others extend it via Hydra defaults)
python -m pipelinerl.launch --config-name=grpo_stage2 output_dir=results/grpo

# 4-GPU node
python -m pipelinerl.launch --config-name=base_4gpu output_dir=results/base1

# Redis-backed streams instead of filesystem
python -m pipelinerl.launch streams=redis output_dir=results/base1

# Cluster entrypoints
sbatch run.slurm --config qed_nano_rc --job-name myrun        # multi-node, generic
sbatch stage1_scripts/run_pr_delta.slurm                      # stage-1 process-reward proof training
bash   stage2_scripts/run_grpo.sh                             # stage-2 scripted launches
```

> Always use a timestamped `output_dir` — PipelineRL crashes the whole job on WandB run-name collisions.

### Tests

There is no pytest suite wired into the repo; tests under `tests/` are standalone `unittest` modules that import `pipelinerl` files by path. Run individually:

```bash
python -m unittest tests.test_verifier_api_process_reward -v
# or a single test:
python -m unittest tests.test_verifier_api_process_reward.ProcessJudgeFormattingTest.test_format_variants_block_numbers_variants
```

Per `AGENTS.md`: prefer a short `python -m pipelinerl.launch ...` run with a timestamped `output_dir` as the smoke check for any nontrivial change.

## Architecture

PipelineRL is a Hydra-driven multi-process pipeline. The key trick is **in-flight weight updates**: after each optimizer step, the trainer broadcasts new weights to the inference (vLLM) servers over NCCL without halting sampling. This keeps batches large and data near on-policy. By default it runs a simplified GRPO — no value net, no trust-region clip, no KL/entropy bonuses (KL is available via config).

### Stages (what to read first when tracing a run)

1. **Orchestrator — `pipelinerl/launch.py`**
   Parses Hydra config, builds a `WorldMap` (`pipelinerl/world.py`) from `WORLD_SIZE`/`RANK`/`MASTER_ADDR`, allocates each node's GPUs into actor / preprocessor / trainer pools by `cfg.world.*_fraction`, and spawns all subprocesses (`actor_llm`, `preprocessor_llm`, `actor`, `preprocessor`, `verifier`, `finetune`). `validate_config` here enforces cross-section invariants (e.g. `preprocessor_fraction>0` if KL, `seq_parallel>1` requires `seq_packing`, VLM restrictions).

2. **Inference servers**
   - **Actor LLMs** (`pipelinerl/entrypoints/run_vllm1.py` → `pipelinerl/run_llm.py`): subclass vLLM's `Worker` to add NCCL process-group setup, `receive_weight_update` (pauses inference, broadcasts weights, reloads params), and HTTP endpoints `POST /v1/chat/completion` and `POST /receive_weight_update`.
   - **Reference LLMs**: plain `vllm.entrypoints.openai.api_server` for reference log-probs.

3. **Actors — `pipelinerl/entrypoints/run_actor.py`** (legacy: `rc_actor.py` / `run_rc_actor.py` for reasoning-cache variants)
   Loads datasets, waits for inference, runs `ActorLoop` with `problem_queue` + `result_queue`; workers run a uvloop asyncio loop that issues concurrent HTTP requests to actor LLMs and listens for weight-update broadcasts. Writes rollouts to the `actor` stream; publishes metrics to the `stats` stream and WandB. Backpressure via `cfg.finetune.max_lag` and `cfg.finetune.weight_update_interval`.

4. **Preprocessor — `pipelinerl/entrypoints/run_preprocess.py`**
   Reads raw traces from `cfg.preprocess.input` (default `actor`), tokenizes + preprocesses in a `ProcessPoolExecutor`, optionally attaches reference log-probs, writes micro-batches to `cfg.preprocess.output` (default `training_data`).

5. **Trainer — `pipelinerl/entrypoints/run_finetune.py`** (→ `pipelinerl/finetune_loop.py`, with RL logic in `pipelinerl/finetune/rl/`)
   Background threads collate preprocessed micro-batches into tensors; main loop: pull batch → `rl_step` (policy-gradient + optional KL) → `optimizer.step` → `lr_scheduler.step`. On rank 0, `WeightUpdateManager.send_weight_update` gathers params, posts `WeightUpdateRequest` to actor LLMs, broadcasts tensors over NCCL, and writes `WeightUpdateSuccess` to the update stream.

6. **Verifier — `pipelinerl/entrypoints/run_environment.py`**
   FastAPI service: `POST /` checks model outputs (math via `math_verify`, countdown via `countdown_utils`, proofs via LLM grader); `GET /health` for readiness.

### Streams (`pipelinerl/streams.py`)

JSON-line message bus between processes. Two backends (files or Redis) selected via `streams=` override. Core streams:

| Stream | Direction | Purpose |
|---|---|---|
| `actor` | Actor → Preprocessor | Raw rollout samples |
| `training_data` | Preprocessor → Trainer | Processed training micro-batches |
| `stats` / `stats_test` | Actor → Monitoring | Sliding-window metrics for WandB |
| `actor_test` | Actor → Monitoring | Eval samples |

### Domains (`pipelinerl/domains/`)

Each domain supplies its own `load_datasets.py`, `rollouts.py`, and a verifier. `math/` is the main domain for QED-Nano and includes the **process-reward** machinery (`process_reward_utils.py`, `process_reward_logging.py`, `verifier_api.py`) used by stage-1 configs like `exp_rl_pr_stage1.yaml`. Other domains (`chartqa/`, `counting/`, `guessing/`) follow the same shape.

### LLM grader (proof pipeline)

Proof-verification configs call out to an external LLM grader — configured under `llm_grader.*` in the config. The grader is either a local vLLM server started automatically when training launches (set `llm_grader.vllm_kwargs` for Slurm geometry; `data-parallel-size × tensor-parallel-size == total GPUs`) or a deployed endpoint (e.g. `gpt-oss-120b-twj`). `prompt_name` picks a template from `conf/evaluator_prompts/`; `reasoning_delimiters` determines where to split the response for the final answer. For Responses-API models, `max_output_tokens` is the *total* (prompt + output) budget.

## Config layout (`conf/`)

- `base.yaml` — root config; most experiment configs extend it via Hydra defaults.
- Domain/recipe configs at top level: `qed_nano_rc.yaml`, `qed_nano_rl.yaml`, `grpo_stage1_proof.yaml`, `grpo_stage2.yaml`, `exp_rl_pr_stage1.yaml`, `dense_process_stage2.yaml`, `dense_outcome_stage2.yaml`, `self_distill_stage2.yaml`, `sft_stage2.yaml`, `qwen4b_instruct.yaml`.
- Subdirs `accelerate/`, `deepspeed/`, `finetune/` (`grpo.yaml`, `ppo.yaml`, `actor_critic.yaml`, `base.yaml`), `rewards/` (`pure_success`, `success_and_format`, `format`, etc.), `streams/` (`files.yaml`, `redis.yaml`), `actor/`, `mcp/`.
- File naming is lowercase snake_case (e.g. `dense_process_stage2.yaml`). Keep this pattern for new configs.

## Operational entrypoints

- `run.slurm` — multi-node template; accepts `--config` and `--job-name`, builds `ALL_ADDR`/`MASTER_ADDR` from `scontrol`, calls `python -m pipelinerl.launch`. Contains `<FIXME>` placeholders for `OPENAI_API_KEY`, `HF_TOKEN`, `OPENAI_BASE_URL` that must be set.
- `stage1_scripts/` — stage-1 proof-training launches (including the current process-reward flow in `run_pr_delta.slurm`, which activates `conda activate qed`).
- `stage2_scripts/` — stage-2 scripted launches (`run_grpo.sh`, `run_sft.sh`, `run_dense_outcome.sh`, `run_dense_process.sh`, `run_self_distill.sh`, `run_qwen4b_instruct.sh`).
- `preprocess_data/` — dataset builders (e.g. `build_fineproofs_stage1.py`).
- `prompts/` + `conf/evaluator_prompts/` — grader and reasoning prompts.

## Conventions

- Treat `results/`, `runs/`, `slurm_logs/`, `outputs/`, `.venv/`, and `qed-train/` as generated output; do not commit them.
- Required env vars for various recipes: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GEMINI_API_KEY`, `HF_TOKEN`. Some stage-1 scripts `assert` `GEMINI_API_KEY` up front.
- Commit subjects are short, lowercase, often stage-prefixed (`stage1: ...`). Avoid bare `minor`.
- PRs should list touched configs/scripts, required env vars, and the validation command used; for training changes include before/after metrics.
