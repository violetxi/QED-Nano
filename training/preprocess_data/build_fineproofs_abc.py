"""Build FineProofs-ABC: problems that appear in datasets A, B, and C, joined.

- A = lm-provers/FineProofs-SFT       (reference proof for self-distillation)
- B = haoranli-ml/genvf-multi-policy-train-v1_final_bulle_list_final_filtered_threshold_0.6
      (multi-policy prefix + suffix variants for dense process rewards)
- C = lm-provers/FineProofs-RL        (rubrics for rubric-based RL)

Matching: problem-statement equality with aggressive normalization (NFKC + lowercase
+ strip non-alphanumeric), plus fuzzy matching (SequenceMatcher ratio >= 0.9) to catch
formatting differences like `*Author*` vs `[i]Author[/i]` or `$...$` vs `\\(...\\)`.

A is a subset of C by problem, so A ∩ B == A ∩ B ∩ C (650 problems).

Output record schema:
    problem        str
    source         str
    category       str
    competition    str
    proof          str          from A (reference solution)
    rubrics        str          from C (grading rubric)
    variants       list[str]    from B; each is prefix_steps + one filtered suffix
                                detailed_steps, joined with "\\n\\n"
    num_variants   int
    c_reward_mean  float
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

from datasets import Dataset, DatasetDict, load_dataset

DATASET_A = "lm-provers/FineProofs-SFT"
DATASET_B = "haoranli-ml/genvf-multi-policy-train-v1_final_bulle_list_final_filtered_threshold_0.6"
DATASET_C = "lm-provers/FineProofs-RL"
HUB_REPO = "violetxi/FineProofs-ABC"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineproofs_abc")

FUZZY_RATIO = 0.9
FUZZY_BUCKET = 20
FUZZY_LEN_TOL = 0.4


def norm_alnum(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z]+", "", s)


def join_steps(steps) -> str:
    return "\n\n".join(s for s in (steps or []) if s)


def main(push: bool = True, private: bool = True):
    a = load_dataset(DATASET_A, split="train")
    b = load_dataset(DATASET_B, split="train")
    c = load_dataset(DATASET_C, split="train")

    a_by_key = {norm_alnum(r["problem"]): r for r in a}
    c_by_key = {norm_alnum(r["problem"]): r for r in c}
    b_by_key = defaultdict(list)
    for r in b:
        b_by_key[norm_alnum(r["problem"])].append(r)

    a_keys = set(a_by_key)
    b_keys = set(b_by_key)
    exact = a_keys & b_keys

    b_bucket = defaultdict(list)
    for bk in b_keys:
        if len(bk) >= FUZZY_BUCKET:
            b_bucket[bk[:FUZZY_BUCKET]].append(bk)

    fuzzy_map = {}
    for ak in a_keys - exact:
        if len(ak) < FUZZY_BUCKET:
            continue
        best = (0.0, None)
        for bk in b_bucket.get(ak[:FUZZY_BUCKET], []):
            if abs(len(ak) - len(bk)) / max(len(ak), len(bk), 1) > FUZZY_LEN_TOL:
                continue
            ratio = SequenceMatcher(None, ak, bk).ratio()
            if ratio > best[0]:
                best = (ratio, bk)
        if best[0] >= FUZZY_RATIO:
            fuzzy_map[ak] = best[1]

    records = []
    for ak in sorted(exact | set(fuzzy_map)):
        bk = ak if ak in exact else fuzzy_map[ak]
        a_row = a_by_key[ak]
        c_row = c_by_key.get(ak)

        # Collapse the rubric/no-rubric duplication in B: one row per unique prefix.
        prefix_groups = {}
        for br in b_by_key[bk]:
            prefix_groups.setdefault(br["prefix"], br)

        variants = []
        for br in prefix_groups.values():
            prefix_text = join_steps(br.get("prefix_steps"))
            for fs in br.get("filtered_suffix") or []:
                suffix_text = join_steps(fs.get("detailed_steps"))
                if not suffix_text:
                    continue
                sep = "\n\n" if prefix_text and suffix_text else ""
                variants.append(prefix_text + sep + suffix_text)

        records.append(
            {
                "problem": a_row["problem"],
                "source": a_row.get("source"),
                "category": a_row.get("category"),
                "competition": a_row.get("competition"),
                "proof": a_row.get("proof"),
                "rubrics": c_row.get("rubrics") if c_row else None,
                "variants": variants,
                "num_variants": len(variants),
                "c_reward_mean": float(c_row["reward_mean"]) if c_row else None,
            }
        )

    print(f"records: {len(records)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, "train.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ds = Dataset.from_list(records)
    ds.to_parquet(os.path.join(OUT_DIR, "train.parquet"))

    if push:
        DatasetDict({"train": ds}).push_to_hub(HUB_REPO, private=private)
        print(f"pushed to https://huggingface.co/datasets/{HUB_REPO}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true", help="skip HF upload")
    parser.add_argument("--public", action="store_true", help="push as public repo")
    args = parser.parse_args()

    main(push=not args.no_push, private=not args.public)
