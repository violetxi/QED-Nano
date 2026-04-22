import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import signal
import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from datetime import datetime

import math_verify  # Ensure math_verify is installed

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

import pipelinerl.countdown_utils

import re
import asyncio
import os
import openai
from datasets import load_dataset

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)


logger = logging.getLogger(__name__)


def _timestamp() -> str:
    """Return a formatted timestamp like '2026-01-11 12:30:25,750'."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S,") + f"{now.microsecond // 1000:03d}"


class TimeoutException(Exception):
    pass


class UnparsableException(Exception):
    pass


class NoAnswerException(Exception):
    pass


class EmptyBoxedException(Exception):
    pass


@contextmanager
def timeout(seconds=1):
    def timeout_handler(signum, frame):
        raise TimeoutException("Computation timed out")

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield  # This is the key addition - context managers must yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def verify_answer(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    """
    Checks if a predicted answer matches a gold (correct) answer by making a request to the math_verify package.

    Args:
        prediction (str): The predicted answer to validate
        gold (str): The gold (correct) answer to compare against
        strict (bool): Whether to enforce strict comparison mode.
        - In strict mode: Variables matter and sets are not comparable with tuples
        - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        url (str): URL of the validation service endpoint

    Returns:
        str: The status of the answer, which can be one of the following:
        - "correct": The prediction is correct
        - "wrong": The prediction is incorrect
        - "no_answer": The prediction is empty
        - "unparsable": The prediction cannot be parsed

    """
    if prediction.startswith("countdown"):
        return verify_countdown(prediction, gold)
    else:
        return verify_math(prediction, gold, strict=strict, max_prediction_length=max_prediction_length)


def verify_math(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    try:
        # Input Sanitization / Validation (very important)
        if not isinstance(prediction, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")

        boxed_start = prediction.rfind("\\boxed{")

        if boxed_start < 0:
            raise NoAnswerException()

        boxed_prediction = prediction[boxed_start:]
        if "\\boxed{}" in boxed_prediction:
            raise EmptyBoxedException()

        if len(boxed_prediction) > max_prediction_length:
            raise UnparsableException()

        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)
        if not boxed_prediction_parsed:
            raise ValueError("Failed to parse prediction.")

        with timeout(1):
            equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed, strict=strict, timeout_seconds=1)
        if equivalent:
            answer_status = "correct"
        else:
            answer_status = "wrong"

    except Exception as e:
        match e:
            case NoAnswerException():
                answer_status = "no_answer"
            case _:
                answer_status = "unparsable"
    return answer_status


def verify_countdown(prediction: str, gold: str) -> str:
    target = eval(gold.split("-")[1])
    numbers = eval(gold.split("-")[2])

    equation = pipelinerl.countdown_utils.extract_solution(solution_str=prediction)

    if equation is None:
        return "no_answer"

    format_correct = pipelinerl.countdown_utils.validate_format(prediction)
    if not format_correct:
        return "unparsable"

    # Validate equation uses correct numbers
    if not pipelinerl.countdown_utils.validate_equation(equation, numbers):
        return "wrong"

    # Evaluate equation
    try:
        result = pipelinerl.countdown_utils.evaluate_equation(equation)
        if result is None:
            return "wrong"

        if abs(result - target) < 1e-5:  # Account for floating point precision
            return "correct"
        else:
            return "wrong"
    except Exception as _:
        return "wrong"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    gold: str,
    strict: bool = True,
    max_prediction_length: int = 1000,
):
    """
    Verify the answer using the verifier API.
    """
    json = {
        "prediction": prediction,
        "gold": gold,
        "strict": strict,
        "max_prediction_length": max_prediction_length,
    }
    # logger.info(f"Verifying answer with request: prediction: {prediction[:100]}, gold: {gold[:100]}, strict: {strict}, max_prediction_length: {max_prediction_length}")
    async with session.post(
        f"http://{host}:{port}/verify_answer",
        json=json,
    ) as response:
        if response.status == 200:
            data = await response.json()
            return data["answer_status"]
        else:
            logger.error(f"Error verifying answer: {response.status}")
            logger.error(f"Response: {await response.text()}")
            raise ValueError("Error verifying answer")


class MathEnvironment:

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                gold = request["gold"]
                strict = request["strict"]
                max_prediction_length = request["max_prediction_length"]

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(verify_answer, prediction, gold, strict, max_prediction_length)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)


def _infer_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "prompts").is_dir():
            return parent
    return module_path.parent


_REPO_ROOT = _infer_repo_root()
_EVALUATOR_PROMPT_DIR = (_REPO_ROOT / "prompts" / "evaluator_prompts").resolve()


def load_evaluator_prompt(prompt_name: str | os.PathLike) -> str:
    """
    Load an evaluator prompt file from prompts/evaluator_prompts.
    """
    if prompt_name is None:
        raise ValueError("llm_grader.prompt_name must name a Markdown file in prompts/evaluator_prompts")

    prompt_str = str(prompt_name).strip()
    if not prompt_str:
        raise ValueError("llm_grader.prompt_name cannot be empty")

    filename = Path(prompt_str).name
    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    prompt_path = (_EVALUATOR_PROMPT_DIR / filename).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Evaluator prompt '{filename}' not found in {_EVALUATOR_PROMPT_DIR}")
    return prompt_path.read_text(encoding="utf-8")


def parse_schema(schema: Any) -> str:
    """
    Normalize a schema payload into the grader-compatible string format.

    The schema is often provided as a list of dicts with keys "title", "points",
    and "desc". This helper converts that structure into the Markdown string
    expected by the evaluator prompt.
    """
    if isinstance(schema, str):
        return schema
    if not isinstance(schema, list):
        raise TypeError("Schema must be a string or a list of dicts")

    sections: list[str] = []
    for idx, entry in enumerate(schema):
        if not isinstance(entry, dict):
            raise ValueError(f"Schema entry at index {idx} must be a dict")
        title = entry.get("title")
        points = entry.get("points")
        description = entry.get("desc") or entry.get("description")
        if title is None or points is None or description is None:
            raise ValueError(f"Schema entry at index {idx} missing title, points, or description")
        section = f"# {title} ({points} points)\nDescription: {description}".strip()
        sections.append(section)

    return "\n\n".join(sections)

_grader_clients: dict[str, Any] = {}

@dataclass
class ProofVerificationResult:
    score: int
    metrics: dict[str, float | int] = field(default_factory=dict)
    table_entry: dict[str, str | int] | None = None


@dataclass
class GraderResponse:
    output_text: str
    reasoning_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def resolve_provider(model: str | None, explicit: str | None = None) -> str:
    """
    Pick grader provider. Explicit value wins; otherwise infer from model name.

    - `google/...` prefix or anything containing `gemini` -> `gemini`
    - everything else -> `openai` (OpenAI-compatible HTTP API, incl. OpenRouter)
    """
    if explicit and explicit != "auto":
        return explicit
    if not model:
        return "openai"
    name = model.lower()
    if name.startswith("google/") or "gemini" in name:
        return "gemini"
    return "openai"


def _strip_google_prefix(model: str) -> str:
    """Drop a leading `google/` on Gemini model IDs for the genai SDK."""
    return model.split("/", 1)[1] if model.lower().startswith("google/") else model


def _extract_reasoning_from_response(response: Any) -> str:
    """
    Extract reasoning content from an OpenAI Responses API Response object.

    The Response object has an `output` list containing output items.
    For reasoning models, this includes items with type="reasoning" that have
    a `content` list of text objects.

    See: https://platform.openai.com/docs/api-reference/responses/object
    """
    reasoning_chunks: list[str] = []
    for item in response.output or []:
        if getattr(item, "type", None) == "reasoning":
            for content_item in getattr(item, "content", []) or []:
                text = getattr(content_item, "text", None)
                if text:
                    reasoning_chunks.append(text)
    return "\n\n".join(reasoning_chunks)


def _normalize_openai_response(response: Any) -> GraderResponse:
    output_text = getattr(response, "output_text", None) or ""
    reasoning_text = _extract_reasoning_from_response(response)
    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
        if output_tokens is None and isinstance(usage, dict):
            output_tokens = usage.get("output_tokens")
    return GraderResponse(
        output_text=output_text,
        reasoning_text=reasoning_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _build_gemini_config(api_kwargs: dict[str, Any] | None):
    """Translate our sampling_kwargs dict into a google.genai GenerateContentConfig."""
    from google.genai import types

    remaining = dict(api_kwargs or {})
    thinking_kwargs: dict[str, Any] = {"include_thoughts": True}
    if "thinking_level" in remaining:
        thinking_kwargs["thinking_level"] = remaining.pop("thinking_level")
    if "thinking_budget" in remaining:
        thinking_kwargs["thinking_budget"] = remaining.pop("thinking_budget")
    if "include_thoughts" in remaining:
        thinking_kwargs["include_thoughts"] = bool(remaining.pop("include_thoughts"))

    config_kwargs: dict[str, Any] = {
        "thinking_config": types.ThinkingConfig(**thinking_kwargs),
    }
    passthrough = {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "candidate_count",
        "response_mime_type",
    }
    for key in list(remaining.keys()):
        if key in passthrough:
            config_kwargs[key] = remaining.pop(key)

    if remaining:
        logger.warning("Ignoring unsupported Gemini sampling_kwargs: %s", sorted(remaining.keys()))

    return types.GenerateContentConfig(**config_kwargs)


def _normalize_gemini_response(response: Any) -> GraderResponse:
    reasoning_chunks: list[str] = []
    output_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                reasoning_chunks.append(text)
            else:
                output_chunks.append(text)
    output_text = "\n".join(output_chunks).strip()
    if not output_text:
        # Fallback: SDK exposes `.text` that concatenates non-thought parts.
        output_text = (getattr(response, "text", None) or "").strip()
    reasoning_text = "\n\n".join(reasoning_chunks).strip()

    usage = getattr(response, "usage_metadata", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None)
        candidates_tokens = getattr(usage, "candidates_token_count", None)
        thought_tokens = getattr(usage, "thoughts_token_count", None)
        if candidates_tokens is not None or thought_tokens is not None:
            output_tokens = (candidates_tokens or 0) + (thought_tokens or 0)
    return GraderResponse(
        output_text=output_text,
        reasoning_text=reasoning_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _should_collect_metrics(collect_flag: bool | None) -> bool:
    if collect_flag is None:
        return True
    return collect_flag


def _merge_metrics(
    base_metrics: dict[str, float | int],
    rollout_metrics: dict[str, float | int],
) -> dict[str, float | int]:
    metrics = dict(rollout_metrics)
    metrics.update(base_metrics)
    return metrics


def _build_rollout_metrics(success: bool, failure_causes: list[str], num_retries: int = 0) -> dict[str, int]:
    metrics: dict[str, int] = {}
    if success:
        metrics["verifier/rollouts/success"] = 1
    else:
        metrics["verifier/rollouts/failure"] = 1
        failure_metric_recorded = False
        if failure_causes:
            unique_causes = set(failure_causes)
            if unique_causes == {"timeout"}:
                metrics["verifier/failures/timeout"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"rate_limit"}:
                metrics["verifier/failures/rate_limit"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"no_input"}:
                metrics["verifier/failures/no_input"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"no_score_tag"}:
                metrics["verifier/failures/no_score_tag"] = 1
                failure_metric_recorded = True
        if not failure_metric_recorded:
            metrics["verifier/failures/all_attempts_failed"] = 1

    if num_retries > 0:
        metrics["verifier/failures/num_retries"] = num_retries
    return metrics


def get_grader_client(provider: str):
    """
    Lazily initialize and cache a grader client for the given provider.

    - `openai` -> OpenAI-compatible client using OPENAI_API_KEY / OPENAI_BASE_URL.
    - `gemini` -> google-genai client using GEMINI_API_KEY.
    """
    if provider in _grader_clients:
        return _grader_clients[provider]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL environment variable")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable for Gemini grader")
        from google import genai  # local import to keep this optional
        client = genai.Client(api_key=api_key)
    else:
        raise RuntimeError(f"Unknown grader provider: {provider}")

    _grader_clients[provider] = client
    return client


def get_openai_client():
    """Back-compat shim; prefer get_grader_client('openai')."""
    return get_grader_client("openai")

def _classify_gemini_exception(exc: Exception) -> str:
    """Map a google-genai exception to our failure-cause buckets."""
    try:
        from google.genai import errors as genai_errors  # type: ignore
    except Exception:
        genai_errors = None  # type: ignore

    if genai_errors is not None and isinstance(exc, getattr(genai_errors, "APIError", tuple())):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status in (429,):
            return "rate_limit"
        if status in (408, 504):
            return "timeout"
    msg = str(exc).lower()
    if "rate limit" in msg or "quota" in msg or " 429" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return "timeout"
    return "other"


# =================================================
# Proof evaluator: dispatches to OpenAI-compatible or Gemini grader
# =================================================
async def verify_proof(
    problem: str,
    ref_solution: str,
    schema: str,
    generation: str,
    prompt_name: str | os.PathLike | None = None,
    model: str | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    client=None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: list[int] = [15, 30, 60, 90, 120],
    log_wandb_metrics: bool | None = None,
    collect_table_entry: bool | None = None,
    provider: str | None = None,
) -> ProofVerificationResult:
    """
    Evaluate a model-generated proof via an LLM grader.

    Providers supported:
      * `openai` (default): OpenAI-compatible Responses API.
      * `gemini`: google-genai SDK with GEMINI_API_KEY.

    `provider` can be passed explicitly; otherwise it is inferred from the model
    name (see `resolve_provider`).
    """

    collect_metrics = _should_collect_metrics(log_wandb_metrics)
    should_collect_table_entry = collect_metrics if collect_table_entry is None else collect_table_entry

    if len(generation.strip()) == 0:
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return ProofVerificationResult(
            score=0,
            metrics=_merge_metrics({}, rollout_metrics),
        )

    if not model:
        raise RuntimeError("verify_proof requires a grader model name; pass via cfg.llm_grader.name")

    resolved_provider = resolve_provider(model, provider)
    client = client or get_grader_client(resolved_provider)
    if not isinstance(schema, str):
        raise TypeError("verify_proof expects schema as Markdown string; convert via parse_schema() first.")

    prompt_template = load_evaluator_prompt(prompt_name)
    prompt_text = prompt_template.format(
        problem=problem,
        human_solution=ref_solution,
        marking_scheme=schema,
        solution=generation,
    )
    api_kwargs = dict(sampling_kwargs) if sampling_kwargs else {}

    loop = asyncio.get_event_loop()

    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model=model,
                input=prompt_text,
                **api_kwargs,
            ),
        )

    async def _call_gemini():
        gemini_model = _strip_google_prefix(model)
        gemini_config = _build_gemini_config(api_kwargs)
        return await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=gemini_model,
                contents=prompt_text,
                config=gemini_config,
            ),
        )

    async def _call_grader():
        if resolved_provider == "gemini":
            return await _call_gemini()
        return await _call_openai()

    def _normalize(raw) -> GraderResponse:
        if resolved_provider == "gemini":
            return _normalize_gemini_response(raw)
        return _normalize_openai_response(raw)

    attempt_failure_causes: list[str] = []
    num_retries = 0
    runtime_metrics: dict[str, float | int] = {}

    for attempt in range(1, max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            response = await asyncio.wait_for(_call_grader(), timeout=timeout_seconds)
            latency_seconds = time.perf_counter() - attempt_start
            normalized = _normalize(response)
            if collect_metrics:
                runtime_metrics = {"verifier/runtime/latency_per_request": latency_seconds}
                if normalized.output_tokens is not None:
                    runtime_metrics["verifier/runtime/output_tokens"] = normalized.output_tokens
                if normalized.input_tokens is not None:
                    runtime_metrics["verifier/runtime/input_tokens"] = normalized.input_tokens
            output_text = normalized.output_text
            match = re.search(r"<score>(\d+)</score>", output_text)
            if match:
                score = int(match.group(1))
                table_entry = None
                if should_collect_table_entry:
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": normalized.reasoning_text,
                        "output_text": output_text,
                        "score": score,
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=True,
                    failure_causes=attempt_failure_causes,
                    num_retries=num_retries,
                )
                return ProofVerificationResult(
                    score=score,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )
            else:
                table_entry = None
                if should_collect_table_entry:
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": normalized.reasoning_text,
                        "output_text": output_text,
                        "score": 0,
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=False,
                    failure_causes=["no_score_tag"],
                    num_retries=num_retries,
                )
                print(f"[verify_proof]: {_timestamp()} - No <score> tag found (attempt {attempt}) — returning 0")
                return ProofVerificationResult(
                    score=0,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )

        except openai.RateLimitError as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("rate_limit")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_proof]: {_timestamp()} - Rate limit hit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except (asyncio.TimeoutError, TimeoutException):
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("timeout")
            if attempt < max_retries:
                num_retries += 1
            print(
                f"[verify_proof]: {_timestamp()} - Timeout after {timeout_seconds}s (attempt {attempt}/{max_retries}), "
                f"retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            cause = _classify_gemini_exception(e) if resolved_provider == "gemini" else "other"
            attempt_failure_causes.append(cause)
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_proof]: {_timestamp()} - Error on attempt {attempt}/{max_retries} ({cause}): {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_proof]: {_timestamp()} - All {max_retries} attempts failed — returning score=0")
    rollout_metrics = _build_rollout_metrics(
        success=False,
        failure_causes=attempt_failure_causes,
        num_retries=num_retries,
    )
    return ProofVerificationResult(
        score=0,
        metrics=_merge_metrics(runtime_metrics, rollout_metrics),
    )

class MathProofEnvironment:
    def __init__(
        self,
        model_name: str | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
        use_wandb: bool | None = True,
        prompt_name: str | os.PathLike | None = None,
        provider: str | None = None,
    ):
        self.model_name = model_name
        self.sampling_kwargs = sampling_kwargs
        self.use_wandb = use_wandb
        if not prompt_name:
            raise ValueError("MathProofEnvironment requires llm_grader.prompt_name to be set")
        self.prompt_name = prompt_name
        self.provider = resolve_provider(model_name, provider)

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        process_pool = ProcessPoolExecutor(max_workers=4)

        @app.post("/verify_answer")
        async def verify(request: dict):
            """
            Evaluate a proof-based problem.
            Expected JSON:
            {
                "problem": "...",
                "ref_solution": "...",
                "schema": "...",
                "generation": "..."
            }
            """
            problem = request["problem"]
            ref_solution = request["ref_solution"]
            schema = parse_schema(request["schema"])
            generation = request["generation"]

            client = get_grader_client(self.provider)
            verification = await verify_proof(
                problem=problem,
                ref_solution=ref_solution,
                schema=schema,
                generation=generation,
                prompt_name=self.prompt_name,
                client=client,
                model=self.model_name,
                sampling_kwargs=self.sampling_kwargs,
                log_wandb_metrics=self.use_wandb,
                collect_table_entry=False,
                provider=self.provider,
            )
            return JSONResponse(content={"score": verification.score})

        @app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)

def main():
    parser = argparse.ArgumentParser(description="Run proof verifier locally for debugging.")
    parser.add_argument("--model", required=True, help="Fully qualified grader model name (e.g. openai/gpt-oss-20b or google/gemini-3.1-flash-lite-preview)")
    parser.add_argument(
        "--sampling-kwargs",
        default=None,
        help="JSON dict of sampling params forwarded to the grader (e.g. '{\"temperature\": 0.8}')",
    )
    parser.add_argument(
        "--prompt-name",
        default="v0",
        help="Name of evaluator prompt file located in prompts/evaluator_prompts (with or without .md suffix).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=[None, "auto", "openai", "gemini"],
        help="Grader provider. Default: infer from model name.",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Number of grader calls to run.")
    args = parser.parse_args()
    sampling_kwargs = json.loads(args.sampling_kwargs) if args.sampling_kwargs else None

    provider = resolve_provider(args.model, args.provider)
    print(f"[verify_proof-cli] Using provider={provider} model={args.model}")

    # Matches training-time grader inputs: problem + rubric, no reference solution.
    dataset = load_dataset("lm-provers/FineProofs-RL", split="train")
    data = dataset[1]
    problem = data["problem"]
    ref_solution = ""  # training pipeline feeds empty when dataset lacks `solution`
    schema = parse_schema(data.get("rubrics") or data.get("schema") or data.get("schema_0"))
    prediction = (
        "We consider the triangle described in the problem. "
        "After setting up coordinates and carrying out the relevant computation, "
        "the required quantity follows from the Pythagorean identity. "
        "Therefore the answer is \\boxed{0}."
    )
    for i in range(args.iterations):
        verification = asyncio.run(
            verify_proof(
                problem,
                ref_solution,
                schema,
                prediction,
                prompt_name=args.prompt_name,
                model=args.model,
                sampling_kwargs=sampling_kwargs,
                provider=provider,
                collect_table_entry=True,
                log_wandb_metrics=True,
            )
        )
        reasoning_preview = (verification.table_entry or {}).get("reasoning", "") or ""
        output_preview = (verification.table_entry or {}).get("output_text", "") or ""
        print(
            f"[{i}] score={verification.score} "
            f"latency={verification.metrics.get('verifier/runtime/latency_per_request', 0):.2f}s "
            f"in_tok={verification.metrics.get('verifier/runtime/input_tokens', '-')} "
            f"out_tok={verification.metrics.get('verifier/runtime/output_tokens', '-')}"
        )
        if reasoning_preview:
            print(f"    reasoning[:200]: {reasoning_preview[:200]!r}")
        if output_preview:
            print(f"    output[:200]: {output_preview[:200]!r}")

if __name__ == "__main__":
    main()
