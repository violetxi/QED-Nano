#!/usr/bin/env python3
"""
Upload all *-graded*.jsonl files from outputs/ to HuggingFace Hub as public datasets.

Each file becomes a separate dataset under the logged-in user's namespace.
E.g. outputs/stage2-qwen3-4b-instruct-imoproofbench-graded.jsonl
  -> violetxi/stage2-qwen3-4b-instruct-imoproofbench-graded

Usage:
    python scripts/upload_graded.py              # upload all graded files
    python scripts/upload_graded.py --dry-run    # just print what would be uploaded
"""

import argparse
import glob
import os
from huggingface_hub import HfApi, whoami
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Upload graded JSONL files to HuggingFace Hub.")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Directory containing the JSONL files.")
    parser.add_argument("--namespace", type=str, default=None,
                        help="HF namespace (user/org). Defaults to logged-in user.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without actually uploading.")
    args = parser.parse_args()

    api = HfApi()
    namespace = args.namespace or whoami()["name"]

    files = sorted(glob.glob(os.path.join(args.outputs_dir, "*-*proofbench-*graded*.jsonl")))
    if not files:
        logger.warning(f"No *-graded*.jsonl files found in {args.outputs_dir}")
        return

    logger.info(f"Found {len(files)} graded files to upload under '{namespace}'")

    for filepath in files:
        filename = os.path.basename(filepath)
        repo_name = filename.replace(".jsonl", "")
        repo_id = f"{namespace}/{repo_name}"

        if args.dry_run:
            logger.info(f"[DRY RUN] {filename} -> {repo_id}")
            continue

        logger.info(f"Uploading {filename} -> {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info(f"Done: https://huggingface.co/datasets/{repo_id}")

    logger.info(f"All {len(files)} files uploaded.")


if __name__ == "__main__":
    main()
