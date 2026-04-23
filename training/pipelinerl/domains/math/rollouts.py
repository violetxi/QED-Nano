import time
import random

import aiohttp
import os
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .process_reward_utils import (
    DEFAULT_STEP_DELIMITER,
    assign_chunk_values_to_output_tokens,
    compute_chunk_advantages,
    compute_chunk_rewards,
    expand_output_token_values_to_labels,
    split_reward_chunks,
)
from .verifier_api import (
    verify_answer_rpc,
    verify_proof,
    parse_schema,
    score_proof_prefixes,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _strip_trailing_chat_tokens(text: str) -> str:
    stripped = text
    for token in ("<|im_end|>", "</s>", "<|endoftext|>"):
        while stripped.endswith(token):
            stripped = stripped[: -len(token)].rstrip()
    return stripped

def remove_reasoning(completion: str, reasoning_delimiters: list[str] = None) -> str:
    # Treat empty lists like None (no delimiter-based stripping).
    if not reasoning_delimiters:
        return completion
    else:
        # Split final answer from reasoning content
        for delim in reasoning_delimiters:
            if delim in completion:
                completion = completion.split(delim)[-1]
                return completion.strip()
        return ""
    

class Metrics(BaseMetrics):
    penalty: float

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping

def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.

def apply_score_threshold(score: float) -> float:
    """
    Apply reward thresholding based on verification score.
    
    Args:
        score: The verification score (0-7 for proof grading)
    
    Returns:
        Thresholded reward.
    """

    # custom thresholding for proof grading
    if score < 1.0:
        return score
    if score < 6.0:
        return 1.0
    return score

async def generate_math_rollout_rc(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    return await generate_math_rollout(cfg, llm, problem, session, rc_actor=True)


async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    rc_actor=False,
) -> RolloutResult:
    messages = []
    actor_cfg = cfg.rc_actor if rc_actor else cfg.actor
    if actor_cfg.system_prompt is not None:
        messages.append({"role": "system", "content": actor_cfg.system_prompt})
    if "task" not in problem:
        logger.error(
            "Rollout received a problem dict without 'task' key; keys present: %s",
            sorted(problem.keys()),
        )
    messages.append({"role": "user", "content": actor_cfg.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)
    # logger.info(f"Reasoning prompt: {prompt}")
    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    generation_raw = llm_call.output.content
    reasoning_delimiters = (
        cfg.llm_grader.reasoning_delimiters
        if "reasoning_delimiters" in cfg.llm_grader
        else None
    )
    generation_final_answer = remove_reasoning(generation_raw, reasoning_delimiters=reasoning_delimiters)
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = actor_cfg.discount_factor

    trace = make_training_text(llm, llm_call)
    process_grader_cfg = cfg.get("process_grader", None)
    process_reward_enabled = bool(
        process_grader_cfg is not None
        and process_grader_cfg.get("enabled", False)
        and problem.get("variants")
    )

    # logger.info(f"Generated training text. Now verifying with verifier API.")
    # logger.info(f"Verifying generated reasoning: {generation_final_answer[:100]}")

    # ===========================================================
    # PROOF-BASED SCORING BRANCH
    # ===========================================================
    verifier_metrics: dict[str, float | int] = {}
    verifier_table_entry: dict[str, str | int] | None = None
    if process_reward_enabled:
        output_token_ids = [token_id for token_id in trace.labels if token_id != -100]
        if output_token_ids:
            decode_token = lambda token_id: llm.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            delimiter = process_grader_cfg.get("delimiter", DEFAULT_STEP_DELIMITER)
            chunks = split_reward_chunks(
                output_token_ids,
                decode_token=decode_token,
                delimiter=delimiter,
            )
            chunk_token_spans = [chunk.token_span for chunk in chunks]
            prefix_texts = [
                _strip_trailing_chat_tokens(
                    llm.tokenizer.decode(
                        output_token_ids[:chunk.token_span[1]],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
                for chunk in chunks
            ]
            prefix_results = await score_proof_prefixes(
                problem=problem["task"],
                ref_solution=problem["answer"],
                variants=problem.get("variants", []),
                prefix_texts=prefix_texts,
                prompt_name=process_grader_cfg.get("prompt_name", None),
                model=process_grader_cfg.get("name", None),
                sampling_kwargs=process_grader_cfg.get("sampling_kwargs", None),
                timeout_seconds=process_grader_cfg.get("timeout_seconds", 900),
                max_retries=process_grader_cfg.get("max_retries", 3),
                retry_backoff=list(process_grader_cfg.get("retry_backoff", [15, 30, 60, 90, 120])),
                provider=process_grader_cfg.get("provider", None),
                max_concurrency=process_grader_cfg.get("max_concurrency", None),
            )
            prefix_scores = [float(result.score) for result in prefix_results]
            chunk_rewards = compute_chunk_rewards(prefix_scores)
            chunk_advantages = compute_chunk_advantages(prefix_scores)
            normalize_by_token_count = bool(process_grader_cfg.get("normalize_by_token_count", True))
            token_rewards_output = assign_chunk_values_to_output_tokens(
                num_output_tokens=len(output_token_ids),
                chunk_token_spans=chunk_token_spans,
                chunk_values=chunk_rewards,
                normalize_by_token_count=normalize_by_token_count,
            )
            token_advantages_output = assign_chunk_values_to_output_tokens(
                num_output_tokens=len(output_token_ids),
                chunk_token_spans=chunk_token_spans,
                chunk_values=chunk_advantages,
                normalize_by_token_count=normalize_by_token_count,
            )
            trace.token_rewards = expand_output_token_values_to_labels(
                labels=trace.labels,
                output_token_values=token_rewards_output,
            )
            trace.token_advantages = expand_output_token_values_to_labels(
                labels=trace.labels,
                output_token_values=token_advantages_output,
            )
            trace.reward = prefix_scores[-1] if prefix_scores else 0.0
            trace.metadata["process_reward"] = {
                "delimiter": delimiter,
                "chunk_token_spans": chunk_token_spans,
                "prefix_scores": prefix_scores,
                "chunk_rewards": chunk_rewards,
                "chunk_advantages": chunk_advantages,
                "judge_outputs": [
                    {
                        "prefix_index": result.prefix_index,
                        "score": result.score,
                        "aligned_path": result.aligned_path,
                        "why": result.why,
                        "output_text": result.output_text,
                    }
                    for result in prefix_results
                ],
            }
            verifier_metrics = {
                "process_reward/final_prefix_score": trace.reward,
                "process_reward/mean_prefix_score": (
                    sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0.0
                ),
                "process_reward/num_prefixes": len(prefix_scores),
            }
        else:
            trace.token_rewards = [0.0] * len(trace.labels)
            trace.token_advantages = [0.0] * len(trace.labels)
            trace.reward = 0.0
            trace.metadata["process_reward"] = {
                "delimiter": process_grader_cfg.get("delimiter", DEFAULT_STEP_DELIMITER),
                "chunk_token_spans": [],
                "prefix_scores": [],
                "chunk_rewards": [],
                "chunk_advantages": [],
                "judge_outputs": [],
            }

        metrics = Metrics(
            reward=trace.reward,
            success=trace.reward == 5,
            no_error=True,
            no_answer=not bool(output_token_ids),
            penalty=0.0,
        )

    elif "schema" in problem:
        llm_grader_cfg = cfg.get("llm_grader", None)
        wandb_table_cfg = llm_grader_cfg.get("wandb_table", None) if llm_grader_cfg is not None else None
        wandb_table_enabled = True
        if wandb_table_cfg is not None:
            wandb_table_enabled = bool(wandb_table_cfg.get("enabled", True))

        schema_text = parse_schema(problem["schema"])
        
        # make sure original_problem is present when using RC stream, since generation prompt is not the same as the original problem but
        # we need to use the original problem for verification
        if rc_actor or cfg.actor.get("use_rc_stream", False):
            assert "original_problem" in problem, "original_problem must be present when using RC stream"
        verification = await verify_proof(
            problem=problem["original_problem"] if "original_problem" in problem else problem["task"],
            ref_solution=problem["answer"],
            schema=schema_text,
            generation=generation_final_answer,
            prompt_name=getattr(cfg.llm_grader, "prompt_name", None),
            model=getattr(cfg.llm_grader, "name", None) if "/" in getattr(cfg.llm_grader, "name", "") else os.getenv("HF_ENDPOINT_REPO"),
            sampling_kwargs=getattr(cfg.llm_grader, "sampling_kwargs", None),
            provider=getattr(cfg.llm_grader, "provider", None),
            log_wandb_metrics=cfg.wandb.use_wandb,
            collect_table_entry=bool(cfg.wandb.use_wandb and wandb_table_enabled),
        )
        score = verification.score
        verifier_metrics = verification.metrics
        verifier_table_entry = verification.table_entry
        # normalize score to [0, 1]
        if cfg.llm_grader.get("custom_reward_threshold", False):
            reward = apply_score_threshold(score) / 7.0
        else:
            reward = (score / 7.0) 
        
        reward = reward * (discount_factor ** llm_call.output_length_tokens)

        # Overlong penalty if configured
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            overlong_penalty = length_penalty(
                llm.parameters["max_tokens"],
                llm_call.output_length_tokens,
                rewards.buffer_tokens,
            )
        reward += overlong_penalty
        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=score == 7,          # treat 6–7 as success
            no_error=True,               # we don't track parse errors here
            no_answer=False,             # proof always produces output
            penalty=overlong_penalty,
        )

    # ===========================================================
    # STANDARD VERIFIABLE-MATH BRANCH
    # ===========================================================
    else:
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        env_job = random.choice(env_jobs)
        assert env_job.port is not None

        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )

        match (answer_status, trace.finished):
            case ("wrong", False):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", True):
                reward = rewards.wrong_answer_finished
            case ("no_answer", False):
                reward = rewards.no_answer_not_finished
            case ("no_answer", True):
                reward = rewards.no_answer_finished
            case ("unparsable", False):
                reward = rewards.unparsable_not_finished
            case ("unparsable", True):
                reward = rewards.unparsable_finished
            case ("correct", False):
                reward = rewards.correct_answer_not_finished
            case ("correct", True):
                reward = rewards.correct_answer_finished
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

        reward *= discount_factor ** llm_call.output_length_tokens
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            overlong_penalty = length_penalty(
                llm.parameters["max_tokens"],
                llm_call.output_length_tokens,
                rewards.buffer_tokens,
            )
        reward += overlong_penalty
        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=answer_status == "correct",
            no_error=answer_status != "unparsable",
            no_answer=answer_status == "no_answer",
            penalty=overlong_penalty,
        )

    # ===========================================================
    # COMMON RETURN BLOCK
    # ===========================================================
    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics=verifier_metrics,
        verifier_table_entry=verifier_table_entry,
    )


async def generate_summarization_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """
    Simplified rollout for summarization tasks.
    No system prompt, no scoring - just generates text with reward=0.
    """
    # Create prompt without system message
    messages = [{"role": "user", "content": problem.get("task", "")}]
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    
    # Create training trace with zero reward
    trace = make_training_text(llm, llm_call)
    trace.reward = 0.0

    # Simple metrics - no scoring
    metrics = Metrics(
        reward=0.0,
        success=True,  # Always "successful" since we're not verifying
        no_error=True,
        no_answer=False,
        penalty=0.0,
    )

   
    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics={},
        verifier_table_entry=None,
    )
