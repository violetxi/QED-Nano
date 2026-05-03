import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import snapshot_download


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

    rows: list[dict[str, Any]] = []
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
    cache_roots.append(Path("/hai/scratch/ziyxiang/.cache/huggingface/datasets"))
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

    path = Path(data_path).expanduser()
    if path.is_file():
        return load_records_from_files([path])
    if path.is_dir():
        candidate_files = find_candidate_data_files(path, split)
        if not candidate_files:
            raise ValueError(f"No dataset files found in directory: {path}")
        return load_records_from_files(candidate_files)

    records = try_load_from_local_hf_cache(data_path, split)
    if records is not None:
        return records

    try:
        dataset = load_dataset(data_path)[split]
        return [dict(row) for row in dataset]
    except Exception:
        records = try_snapshot_download_and_load(data_path, split)
        if records is None:
            raise
        return records


def infer_dataset_label(data_path: str) -> str:
    name = data_path.lower()
    if "imoproofbench" in name:
        return "IMOProofBench"
    if "proofbench" in name:
        return "ProofBench"
    return "UnknownDataset"


def infer_model_label(data_path: str) -> str:
    name = data_path.lower()
    if "dense-process" in name or "dense_process" in name:
        return "dense_process"
    if "dense-outcome" in name or "dense_outcome" in name:
        return "dense_outcome"
    if "self-distill" in name or "self_distill" in name:
        return "self_distill"
    if "instruct" in name:
        return "instruct"
    if "grpo" in name:
        return "grpo"
    if "sft" in name:
        return "sft"

    path = Path(data_path)
    if path.name:
        return path.stem
    return data_path.replace("/", "__")


def choose_problem_key(df: pd.DataFrame) -> str:
    for key in ("question_id", "problem_id", "problem_idx", "problem"):
        if key in df.columns and df[key].notna().any():
            return key
    raise ValueError("Could not find a problem grouping column. Tried question_id, problem_id, problem_idx, problem.")


def choose_score_column(df: pd.DataFrame) -> str:
    for key in ("score", "graded_score"):
        if key in df.columns:
            return key
    raise ValueError("Could not find a score column. Tried score and graded_score.")


def load_problem_success_rates(data_path: str, *, split: str, success_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(load_records(data_path, split))
    if df.empty:
        raise ValueError(f"No rows found for dataset input: {data_path}")

    problem_key = choose_problem_key(df)
    score_key = choose_score_column(df)
    score_values = pd.to_numeric(df[score_key], errors="coerce").fillna(0.0)
    success = score_values >= float(success_threshold)
    grouped = (
        pd.DataFrame({"problem_key": df[problem_key].astype(str), "success": success.astype(int)})
        .groupby("problem_key", sort=True)["success"]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    success_counts = grouped["sum"].to_numpy(dtype=int)
    attempt_counts = grouped["count"].to_numpy(dtype=int)
    return success_counts, attempt_counts


def bootstrap_passk_curve(
    success_counts: np.ndarray,
    attempt_counts: np.ndarray,
    *,
    max_k: int,
    num_bootstrap: int,
    ci_alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(success_counts) == 0:
        raise ValueError("No problems found after grouping.")

    ks = np.arange(1, max_k + 1, dtype=int)
    p_hat = success_counts.astype(float) / attempt_counts.astype(float)
    point = np.mean(1.0 - np.power(1.0 - p_hat[:, None], ks[None, :]), axis=0)

    boot_curves = np.empty((num_bootstrap, max_k), dtype=float)
    problem_count = len(success_counts)
    for b in range(num_bootstrap):
        sample_idx = rng.integers(0, problem_count, size=problem_count)
        sampled_n = attempt_counts[sample_idx]
        sampled_p = p_hat[sample_idx]
        sampled_success_counts = rng.binomial(sampled_n, sampled_p)
        sampled_p_hat = sampled_success_counts.astype(float) / sampled_n.astype(float)
        boot_curves[b] = np.mean(
            1.0 - np.power(1.0 - sampled_p_hat[:, None], ks[None, :]),
            axis=0,
        )

    lower_q = 100.0 * (ci_alpha / 2.0)
    upper_q = 100.0 * (1.0 - ci_alpha / 2.0)
    lower = np.percentile(boot_curves, lower_q, axis=0)
    upper = np.percentile(boot_curves, upper_q, axis=0)
    return point, lower, upper


def plot_bootstrap_passk_dataset(
    dataset_label: str,
    dataset_rows: list[dict[str, object]],
    *,
    output_dir: Path,
    success_threshold: float,
    max_k: int,
) -> Path:
    plt.figure(figsize=(9.5, 6.0))
    for row in sorted(dataset_rows, key=lambda item: str(item["model"])):
        ks = np.arange(1, max_k + 1)
        point = np.asarray(row["point"])
        lower = np.asarray(row["lower"])
        upper = np.asarray(row["upper"])
        plt.plot(ks, point, linewidth=2, label=str(row["model"]))
        plt.fill_between(ks, lower, upper, alpha=0.18)

    plt.xlim(1, max_k)
    plt.ylim(0.0, 1.0)
    tick_candidates = [1, 2, 4, 8, 16, 32, 64]
    plt.xticks([tick for tick in tick_candidates if tick <= max_k])
    plt.xlabel("k")
    plt.ylabel(f"Bootstrap pass@k (score >= {int(success_threshold)}/7)")
    plt.title(dataset_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bootstrap_passk__{dataset_label.lower()}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path
