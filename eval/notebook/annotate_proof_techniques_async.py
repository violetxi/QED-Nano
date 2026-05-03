#!/usr/bin/env python3
"""Async Gemini annotation for proof-technique extraction.

This script loads a ProofBench-style dataset, sends each `(problem, solution text)`
pair to Gemini using Google's OpenAI-compatible chat completions endpoint, parses the
JSON annotation, and writes an augmented JSONL file with the original row plus:

- `proof_technique_annotation`
- `proof_technique_annotation_raw`
- `proof_technique_annotation_error`
- `proof_technique_annotation_usage`
- `proof_technique_annotation_model`

Example:
    /hai/scratch/ziyxiang/QED-Nano/eval/.venv/bin/python \
      notebook/annotate_proof_techniques_async.py \
      --data-path violetxi/stage1_proof-qwen3-4b-grpo-proofbench-summary-graded \
      --output-path notebook/model_solution_annotations/stage1_proof-qwen3-4b-grpo-proofbench-summary-graded-techniques.jsonl \
      --model-config google/gemini-3-pro \
      --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_dataset
from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml


PROMPT = """You are an expert olympiad mathematics annotator. Your task is to analyze a model-generated proof solution and identify the distinct proof techniques actually used or clearly attempted.

You are NOT grading correctness. A solution may be wrong, incomplete, or incoherent and still contain recognizable proof techniques. Your goal is to characterize the proof strategy and mathematical methods attempted by the model.

A "proof technique" is any substantive method that shapes the proof. This includes both mathematical tools and higher-level proof strategies such as contradiction, casework, construction, induction, extremal principle, invariants, and descent.

Use ONLY the following controlled `proof_technique_tag` values.

Allowed `proof_technique_tag`s:

Geometry:
- coordinate_geometry
- synthetic_geometry
- angle_chasing
- cyclic_quadrilateral
- power_of_point
- radical_axis
- similarity
- homothety
- inversion
- complex_numbers
- barycentric_trilinear
- area_method

Number theory:
- modular_arithmetic
- parity
- divisibility
- prime_factorization
- p_adic_valuation
- gcd_coprime
- quadratic_residues

Algebra:
- algebraic_manipulation
- polynomial_factorization
- roots_vieta
- degree_argument
- recurrence
- substitution
- functional_equation

Combinatorics:
- double_counting
- inclusion_exclusion
- pigeonhole
- graph_modeling
- extremal_principle
- invariant_monovariant
- local_to_global
- coloring_argument

Inequality:
- bounding
- am_gm
- cauchy_schwarz
- convexity
- jensen
- smoothing

General proof techniques:
- contradiction
- casework
- construction
- uniqueness_argument
- existence_argument
- exhaustive_search
- induction
- infinite_descent

Annotation rules:
1. Read the full `submission_text`.
2. Identify all distinct proof techniques that are actually used or clearly attempted.
3. Do not annotate a tag just because a keyword appears. The technique must play a real role in the solution.
4. Prefer specific tags over broad tags whenever possible.
5. Broad tags such as `synthetic_geometry`, `algebraic_manipulation`, and `bounding` should be used only when no more specific tag captures the method.
6. Do not include both a broad tag and a more specific tag unless both genuinely contribute in different ways.
7. General proof techniques such as `contradiction`, `casework`, `construction`, `induction`, and `infinite_descent` are valid tags, but include them only when they materially shape the argument.
8. Do not infer techniques from the problem statement, official solution, or what a correct solution should have done. Only annotate what appears in `submission_text`.
9. If the solution is wrong or incomplete, still annotate recognizable attempted techniques.
10. Use `status = "attempted"` when a technique is clearly started but not successfully carried through.
11. Use at most 5 technique entries total.
12. `all_technique_tags` must be exactly the unique set of tags appearing in `proof_techniques`.

Return ONLY valid JSON with this schema:

{
  "proof_techniques": [
    {
      "tag": "<one allowed proof_technique_tag>",
      "confidence": "<high | medium | low>",
      "status": "<used | attempted>",
      "evidence": "<brief quote or paraphrase from submission_text>"
    }
  ],
  "all_technique_tags": [
    "<tag 1>",
    "<tag 2>"
  ],
  "solution_style_summary": "<1-2 sentence qualitative summary of the proof strategy>",
  "solution_coherence": "<coherent | partially_coherent | incoherent>",
  "notes": "<short note about ambiguity, failed attempts, or overly generic reasoning>"
}
"""


ALLOWED_TAGS = {
    "coordinate_geometry",
    "synthetic_geometry",
    "angle_chasing",
    "cyclic_quadrilateral",
    "power_of_point",
    "radical_axis",
    "similarity",
    "homothety",
    "inversion",
    "complex_numbers",
    "barycentric_trilinear",
    "area_method",
    "modular_arithmetic",
    "parity",
    "divisibility",
    "prime_factorization",
    "p_adic_valuation",
    "gcd_coprime",
    "quadratic_residues",
    "algebraic_manipulation",
    "polynomial_factorization",
    "roots_vieta",
    "degree_argument",
    "recurrence",
    "substitution",
    "functional_equation",
    "double_counting",
    "inclusion_exclusion",
    "pigeonhole",
    "graph_modeling",
    "extremal_principle",
    "invariant_monovariant",
    "local_to_global",
    "coloring_argument",
    "bounding",
    "am_gm",
    "cauchy_schwarz",
    "convexity",
    "jensen",
    "smoothing",
    "contradiction",
    "casework",
    "construction",
    "uniqueness_argument",
    "existence_argument",
    "exhaustive_search",
    "induction",
    "infinite_descent",
}

ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_STATUS = {"used", "attempted"}
ALLOWED_COHERENCE = {"coherent", "partially_coherent", "incoherent"}

GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_DATA_PATH = "violetxi/stage1_proof-qwen3-4b-grpo-proofbench-summary-graded"


def remove_self_evaluation(text: str) -> str:
    """Drop common self-evaluation sections that can pollute technique tagging."""
    pattern = r"(?ms)^#+\s*[*\s]*(?:(?:Self|Final)\s*[*\s]*){0,2}Evaluation\b.*$"
    return re.sub(pattern, "", text).strip()


def load_yaml_config(config_name: str) -> dict[str, Any]:
    config_path = Path("configs/models") / config_name
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sanitize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(model_config)
    for key in ["date", "human_readable_id", "prompt_margin", "model_revision"]:
        cleaned.pop(key, None)
    return cleaned


def ensure_writable_hf_cache_env() -> None:
    """Point HF cache env vars at writable locations if they are unset."""
    tmp_root = Path("/tmp/hf_cache_qed_nano")
    hf_home = Path(os.environ.get("HF_HOME", tmp_root.as_posix()))
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", (hf_home / "datasets").as_posix()))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", (hf_home / "hub").as_posix()))
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home.as_posix())
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_cache.as_posix())
    os.environ.setdefault("HF_HUB_CACHE", hub_cache.as_posix())


def repo_id_to_local_slug(repo_id: str) -> str:
    return repo_id.replace("/", "___")


def read_arrow_stream_file(path: Path) -> list[dict[str, Any]]:
    source = pa.memory_map(path.as_posix(), "r")
    reader = pa_ipc.RecordBatchStreamReader(source)
    table = reader.read_all()
    return table.to_pylist()


def read_parquet_file(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path.as_posix())
    return table.to_pylist()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in JSON file {path}")
        return data
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_candidate_data_files(root: Path, split: str) -> list[Path]:
    patterns = [
        f"**/*{split}*.arrow",
        f"**/*{split}*.parquet",
        f"**/*{split}*.jsonl",
        f"**/*{split}*.json",
        "**/*.arrow",
        "**/*.parquet",
        "**/*.jsonl",
        "**/*.json",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if path.name == "dataset_info.json":
                continue
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def load_records_from_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".arrow":
            rows.extend(read_arrow_stream_file(path))
        elif path.suffix == ".parquet":
            rows.extend(read_parquet_file(path))
        elif path.suffix in {".json", ".jsonl"}:
            rows.extend(load_json_records(path))
    return rows


def try_load_from_local_hf_cache(data_path: str, split: str) -> list[dict[str, Any]] | None:
    cache_roots = []
    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        cache_roots.append(Path(env_cache))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    repo_slug = repo_id_to_local_slug(data_path)

    for cache_root in cache_roots:
        dataset_root = cache_root / repo_slug
        if not dataset_root.exists():
            continue
        candidate_files = find_candidate_data_files(dataset_root, split)
        if candidate_files:
            rows = load_records_from_files(candidate_files)
            if rows:
                return rows
    return None


def try_snapshot_download_and_load(data_path: str, split: str) -> list[dict[str, Any]] | None:
    local_dir = Path(os.environ.get("HF_HUB_CACHE", "/tmp/hf_cache_qed_nano/hub")) / "dataset_snapshots" / data_path.replace("/", "--")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=data_path,
            repo_type="dataset",
            local_dir=local_dir.as_posix(),
            local_dir_use_symlinks=False,
            allow_patterns=["*.arrow", "*.parquet", "*.json", "*.jsonl", "data/**"],
        )
    )
    candidate_files = find_candidate_data_files(snapshot_path, split)
    if not candidate_files:
        return None
    rows = load_records_from_files(candidate_files)
    return rows or None


def load_records(data_path: str, split: str) -> list[dict[str, Any]]:
    ensure_writable_hf_cache_env()
    if os.path.exists(data_path):
        dataset = load_dataset("json", data_files=data_path)[split]
        records = [dict(row) for row in dataset]
    else:
        records: list[dict[str, Any]] | None = None
        try:
            dataset = load_dataset(data_path)[split]
            records = [dict(row) for row in dataset]
        except Exception:
            records = try_load_from_local_hf_cache(data_path, split)
            if records is None:
                records = try_snapshot_download_and_load(data_path, split)
            if records is None:
                raise
    for idx, row in enumerate(records):
        row["source_index"] = idx
    return records


def read_existing_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "source_index" in row:
                existing[int(row["source_index"])] = row
    return existing


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def build_user_message(problem: str, submission_text: str, solution_column: str) -> str:
    return (
        "Annotate the following olympiad-proof solution.\n\n"
        "problem:\n"
        f"{problem}\n\n"
        "submission_text_source_column:\n"
        f"{solution_column}\n\n"
        "submission_text:\n"
        f"{submission_text}"
    )


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    # Repair malformed backslash escapes that sometimes appear inside otherwise-valid JSON
    # when the model includes LaTeX-like content in string fields.
    stripped = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        trailer = stripped[idx + end :].strip()
        if not trailer or trailer.startswith("```"):
            return obj
    raise ValueError("Could not locate a valid JSON object in the model response.")


def normalize_technique_entry(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"Technique entry must be an object, got {type(entry).__name__}.")
    tag = str(entry.get("tag", "")).strip()
    confidence = str(entry.get("confidence", "")).strip().lower()
    status = str(entry.get("status", "")).strip().lower()
    evidence = str(entry.get("evidence", "")).strip()

    if tag not in ALLOWED_TAGS:
        raise ValueError(f"Unknown technique tag: {tag!r}")
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Unknown confidence value: {confidence!r}")
    if status not in ALLOWED_STATUS:
        raise ValueError(f"Unknown status value: {status!r}")
    return {
        "tag": tag,
        "confidence": confidence,
        "status": status,
        "evidence": evidence,
    }


def normalize_annotation(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Annotation must be a JSON object, got {type(obj).__name__}.")

    raw_techniques = obj.get("proof_techniques", [])
    if not isinstance(raw_techniques, list):
        raise ValueError("`proof_techniques` must be a list.")

    normalized_techniques: list[dict[str, str]] = []
    seen = set()
    for raw_entry in raw_techniques:
        entry = normalize_technique_entry(raw_entry)
        key = (entry["tag"], entry["status"], entry["evidence"])
        if key in seen:
            continue
        seen.add(key)
        normalized_techniques.append(entry)
        if len(normalized_techniques) == 5:
            break

    all_tags = []
    for entry in normalized_techniques:
        if entry["tag"] not in all_tags:
            all_tags.append(entry["tag"])

    coherence = str(obj.get("solution_coherence", "")).strip().lower()
    if coherence not in ALLOWED_COHERENCE:
        raise ValueError(f"Unknown solution_coherence value: {coherence!r}")

    return {
        "proof_techniques": normalized_techniques,
        "all_technique_tags": all_tags,
        "solution_style_summary": str(obj.get("solution_style_summary", "")).strip(),
        "solution_coherence": coherence,
        "notes": str(obj.get("notes", "")).strip(),
    }


def estimate_cost(usage: dict[str, Any], read_cost: float, write_cost: float) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    usd = (prompt_tokens * read_cost + completion_tokens * write_cost) / 1_000_000
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": usd,
    }


def summarize_costs(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    read_cost: float,
    write_cost: float,
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    rows_with_usage = 0
    error_rows = 0

    for row in rows:
        usage = row.get("proof_technique_annotation_usage") or {}
        if usage:
            rows_with_usage += 1
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)
        estimated_cost_usd += float(usage.get("estimated_cost_usd", 0.0) or 0.0)
        if row.get("proof_technique_annotation_error"):
            error_rows += 1

    return {
        "model": model_name,
        "rows": len(rows),
        "rows_with_usage": rows_with_usage,
        "error_rows": error_rows,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing": {
            "read_cost_per_1m_tokens": read_cost,
            "write_cost_per_1m_tokens": write_cost,
        },
    }


def has_successful_annotation(row: dict[str, Any]) -> bool:
    return row.get("proof_technique_annotation") is not None and not row.get("proof_technique_annotation_error")


async def post_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_key: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    extra_body: dict[str, Any] | None,
    max_retries: int,
    use_json_object_response_format: bool,
) -> tuple[str, dict[str, Any], str | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
    }
    if use_json_object_response_format:
        payload["response_format"] = {"type": "json_object"}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if extra_body:
        payload["extra_body"] = extra_body

    for attempt in range(max_retries + 1):
        try:
            async with session.post(GOOGLE_OPENAI_URL, headers=headers, json=payload) as response:
                raw_response = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {raw_response[:2000]}")
                data = json.loads(raw_response)
                message = data["choices"][0]["message"]
                content = message.get("content", "") or ""
                usage = data.get("usage", {}) or {}
                response_id = data.get("id")
                return content, usage, response_id
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as exc:
            if attempt >= max_retries:
                raise
            backoff = min(60.0, 2 ** attempt) + random.random()
            await asyncio.sleep(backoff)
            last_error = exc
    raise RuntimeError(f"Unreachable retry state: {last_error}")


async def annotate_row(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    row: dict[str, Any],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    extra_body: dict[str, Any] | None,
    max_retries: int,
    read_cost: float,
    write_cost: float,
    use_json_object_response_format: bool,
    problem_column: str,
    solution_column: str,
) -> dict[str, Any]:
    async with semaphore:
        problem = str(row.get(problem_column, ""))
        submission_text = remove_self_evaluation(str(row.get(solution_column, "")))
        prompt_text = build_user_message(problem, submission_text, solution_column)
        raw_text = ""
        response_id = None
        usage: dict[str, Any] = {}
        error = None
        annotation = None

        try:
            raw_text, usage, response_id = await post_chat_completion(
                session,
                api_key=api_key,
                model=model,
                prompt_text=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
                max_retries=max_retries,
                use_json_object_response_format=use_json_object_response_format,
            )
            parsed = extract_json_object(raw_text)
            annotation = normalize_annotation(parsed)
        except Exception as exc:
            error = str(exc)

        output = dict(row)
        output["proof_technique_annotation"] = annotation
        output["proof_technique_annotation_raw"] = raw_text
        output["proof_technique_annotation_error"] = error
        output["proof_technique_annotation_usage"] = estimate_cost(usage, read_cost, write_cost)
        output["proof_technique_annotation_model"] = model
        output["proof_technique_annotation_response_id"] = response_id
        output["proof_technique_annotation_prompt_version"] = "v2"
        output["proof_technique_annotation_source_column"] = solution_column
        return output


def should_skip_existing(row: dict[str, Any], *, solution_column: str) -> bool:
    if not has_successful_annotation(row):
        return False

    existing_source_column = row.get("proof_technique_annotation_source_column")
    if existing_source_column is None:
        return solution_column == "model_solution"
    return str(existing_source_column) == solution_column


def select_rows(
    rows: list[dict[str, Any]],
    *,
    score_min: float | None,
    score_max: float | None,
    source_index_min: int | None,
    source_index_max: int | None,
    problem_ids: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        score = row.get("score")
        if score_min is not None and score is not None and float(score) < score_min:
            continue
        if score_max is not None and score is not None and float(score) > score_max:
            continue
        if source_index_min is not None and int(row["source_index"]) < source_index_min:
            continue
        if source_index_max is not None and int(row["source_index"]) > source_index_max:
            continue
        if problem_ids is not None and str(row.get("problem_id")) not in problem_ids:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


async def run_async(args: argparse.Namespace) -> None:
    model_config = sanitize_model_config(load_yaml_config(args.model_config))
    if model_config.get("api") != "google":
        raise ValueError(f"Expected a Google model config, got api={model_config.get('api')!r}")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.api_key_env} is not set.")

    rows = load_records(args.data_path, args.split)
    problem_ids = set(args.problem_id) if args.problem_id else None
    selected = select_rows(
        rows,
        score_min=args.score_min,
        score_max=args.score_max,
        source_index_min=args.source_index_min,
        source_index_max=args.source_index_max,
        problem_ids=problem_ids,
        limit=args.limit,
    )
    if not selected:
        raise ValueError("No rows selected after applying filters.")

    output_path = Path(args.output_path)
    existing = {} if args.overwrite else read_existing_jsonl(output_path)
    completed_by_index: dict[int, dict[str, Any]] = {}
    pending = []
    for row in selected:
        existing_row = existing.get(int(row["source_index"]))
        if existing_row is not None and should_skip_existing(existing_row, solution_column=args.solution_column):
            completed_by_index[int(row["source_index"])] = existing_row
        else:
            pending.append(row)

    concurrency = args.concurrency or int(model_config.get("concurrent_requests", 16))
    model_name = args.model or str(model_config["model"])
    max_tokens = args.max_tokens or int(model_config.get("max_tokens", 4096))
    temperature = args.temperature
    if temperature is None:
        temperature = model_config.get("temperature")
    top_p = args.top_p
    if top_p is None:
        top_p = model_config.get("top_p")
    extra_body = model_config.get("extra_body")
    if not args.use_config_extra_body:
        extra_body = None
    read_cost = float(model_config.get("read_cost", 0))
    write_cost = float(model_config.get("write_cost", 0))
    summary_path = Path(args.summary_path) if args.summary_path else Path(str(output_path) + ".summary.json")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency, ssl=True)
    semaphore = asyncio.Semaphore(concurrency)

    print(
        f"Selected {len(selected)} rows, reusing {len(completed_by_index)} existing rows, "
        f"processing {len(pending)} new rows with model={model_name} concurrency={concurrency}."
    )

    results_by_index = dict(completed_by_index)
    completed_count = 0
    failed_count = 0
    started = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                annotate_row(
                    session,
                    semaphore,
                    row=row,
                    api_key=api_key,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body,
                    max_retries=args.max_retries,
                    read_cost=read_cost,
                    write_cost=write_cost,
                    use_json_object_response_format=not args.disable_json_object_response_format,
                    problem_column=args.problem_column,
                    solution_column=args.solution_column,
                )
            )
            for row in pending
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), disable=(len(tasks) == 0)):
            annotated = await future
            idx = int(annotated["source_index"])
            if has_successful_annotation(annotated):
                results_by_index[idx] = annotated
                completed_count += 1
            else:
                failed_count += 1
            if args.save_every > 0 and completed_count % args.save_every == 0:
                merged_rows = [results_by_index[int(row["source_index"])] for row in selected if int(row["source_index"]) in results_by_index]
                merged_rows.sort(key=lambda row: int(row["source_index"]))
                write_jsonl(output_path, merged_rows)
                write_json(
                    summary_path,
                    summarize_costs(
                        merged_rows,
                        model_name=model_name,
                        read_cost=read_cost,
                        write_cost=write_cost,
                    ),
                )

    merged_rows = [results_by_index[int(row["source_index"])] for row in selected if int(row["source_index"]) in results_by_index]
    merged_rows.sort(key=lambda row: int(row["source_index"]))
    write_jsonl(output_path, merged_rows)
    summary = summarize_costs(
        merged_rows,
        model_name=model_name,
        read_cost=read_cost,
        write_cost=write_cost,
    )
    write_json(summary_path, summary)

    elapsed = time.time() - started
    print(
        f"Wrote {len(merged_rows)} rows to {output_path}. "
        f"failed_rows_not_written={failed_count} elapsed_sec={elapsed:.1f}"
    )
    print(
        f"Estimated cost: ${summary['estimated_cost_usd']:.4f} "
        f"(prompt_tokens={summary['prompt_tokens']}, completion_tokens={summary['completion_tokens']}). "
        f"Summary saved to {summary_path}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async Gemini technique annotation for proof solutions.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="HF dataset name or local JSON/JSONL path.")
    parser.add_argument("--split", default="train", help="Dataset split to use.")
    parser.add_argument(
        "--output-path",
        default="notebook/model_solution_annotations/stage1_proof-qwen3-4b-grpo-proofbench-summary-graded-techniques.jsonl",
        help="Path to the augmented JSONL output.",
    )
    parser.add_argument("--model-config", default="google/gemini-3-pro", help="Model config under configs/models/.")
    parser.add_argument("--model", default=None, help="Optional direct model name override.")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--problem-column", default="problem", help="Problem statement column.")
    parser.add_argument(
        "--solution-column",
        "--model-solution-column",
        dest="solution_column",
        default="model_solution",
        help="Text column to annotate, e.g. `model_solution` or `reasoning_trace`.",
    )
    parser.add_argument("--score-min", type=float, default=None, help="Optional minimum `score` filter.")
    parser.add_argument("--score-max", type=float, default=None, help="Optional maximum `score` filter.")
    parser.add_argument("--source-index-min", type=int, default=None, help="Optional minimum source index filter.")
    parser.add_argument("--source-index-max", type=int, default=None, help="Optional maximum source index filter.")
    parser.add_argument("--problem-id", action="append", default=None, help="Restrict to one or more problem_id values.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of selected rows.")
    parser.add_argument("--concurrency", type=int, default=None, help="Number of concurrent requests.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum completion tokens per request.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature override.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p override.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries per request.")
    parser.add_argument("--save-every", type=int, default=25, help="Write partial output every N completed rows.")
    parser.add_argument("--summary-path", default=None, help="Optional path for run-level cost summary JSON.")
    parser.add_argument(
        "--use-config-extra-body",
        action="store_true",
        help="Forward `extra_body` from the model config. Disabled by default to avoid thought output.",
    )
    parser.add_argument(
        "--disable-json-object-response-format",
        action="store_true",
        help="Do not send `response_format={type: json_object}`.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore any existing output file and rerun all rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
