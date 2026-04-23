from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipelinerl"
    / "domains"
    / "math"
    / "process_reward_utils.py"
)
SPEC = spec_from_file_location("process_reward_utils", MODULE_PATH)
assert SPEC and SPEC.loader
process_reward_utils = module_from_spec(SPEC)
sys.modules[SPEC.name] = process_reward_utils
SPEC.loader.exec_module(process_reward_utils)

assign_chunk_values_to_output_tokens = process_reward_utils.assign_chunk_values_to_output_tokens
compute_chunk_rewards = process_reward_utils.compute_chunk_rewards
compute_chunk_advantages = process_reward_utils.compute_chunk_advantages
expand_output_token_values_to_labels = process_reward_utils.expand_output_token_values_to_labels
maybe_clip_chunk_advantages_for_length = process_reward_utils.maybe_clip_chunk_advantages_for_length
split_reward_chunks = process_reward_utils.split_reward_chunks


TOKEN_TEXT_BY_ID = {
    1: "al",
    2: "pha",
    3: "### ",
    4: "be",
    5: "ta",
    6: "gamma",
    7: "plain ",
    8: "trace ",
    9: "without ",
    10: "step ",
    11: "delimiters",
    12: "alpha##",
    13: "# ",
    14: "beta### ",
    15: "gamma",
}


def decode_token(token_id: int) -> str:
    return TOKEN_TEXT_BY_ID[token_id]


try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class SplitRewardChunksTest(unittest.TestCase):
    def test_trailing_delimiter_ownership(self) -> None:
        input_ids = [1, 2, 3, 4, 5, 3, 6]

        chunks = split_reward_chunks(input_ids, decode_token)

        self.assertEqual([chunk.token_span for chunk in chunks], [(0, 3), (3, 6), (6, 7)])

    def test_no_delimiter_returns_single_chunk(self) -> None:
        input_ids = [7, 8, 9, 10, 11]

        chunks = split_reward_chunks(input_ids, decode_token)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].token_span, (0, len(input_ids)))


class AssignChunkValuesToLabelsTest(unittest.TestCase):
    def test_maps_values_only_to_output_and_label_tokens(self) -> None:
        input_ids = [1, 2, 3, 4, 5, 3, 6]
        chunks = split_reward_chunks(input_ids, decode_token)
        prefix_scores = [3.0, 5.0, 5.0]
        chunk_values = compute_chunk_advantages(prefix_scores)
        chunk_token_spans = [chunk.token_span for chunk in chunks]

        output_values = assign_chunk_values_to_output_tokens(
            num_output_tokens=len(input_ids),
            chunk_token_spans=chunk_token_spans,
            chunk_values=chunk_values,
            normalize_by_token_count=True,
        )

        self.assertEqual(output_values[:3], [1.0, 1.0, 1.0])
        self.assertEqual(output_values[3:6], [7.0 / 3.0, 7.0 / 3.0, 7.0 / 3.0])
        self.assertEqual(output_values[6], 5.0)
        self.assertAlmostEqual(sum(output_values), sum(chunk_values))

        labels = [-100, -100, 101, 102, 103, 104, 105, 106, 107]
        values = expand_output_token_values_to_labels(labels=labels, output_token_values=output_values)

        self.assertEqual(values[:2], [0.0, 0.0])
        self.assertEqual(values[2:5], [1.0, 1.0, 1.0])
        self.assertEqual(values[5:8], [7.0 / 3.0, 7.0 / 3.0, 7.0 / 3.0])
        self.assertEqual(values[8], 5.0)
        self.assertAlmostEqual(sum(values), sum(chunk_values))

    def test_maps_raw_rewards_only_to_output_tokens(self) -> None:
        input_ids = [1, 2, 3, 4, 5, 3, 6]
        chunks = split_reward_chunks(input_ids, decode_token)
        chunk_values = compute_chunk_rewards([3.0, 5.0, 5.0])

        values = assign_chunk_values_to_output_tokens(
            num_output_tokens=len(input_ids),
            chunk_token_spans=[chunk.token_span for chunk in chunks],
            chunk_values=chunk_values,
            normalize_by_token_count=True,
        )

        self.assertEqual(values[:3], [1.0, 1.0, 1.0])
        self.assertEqual(values[3:6], [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
        self.assertEqual(values[6], 0.0)
        self.assertAlmostEqual(sum(values), sum(chunk_values))

    def test_disable_normalization_keeps_full_chunk_value_per_token(self) -> None:
        input_ids = [1, 2, 3, 4, 5, 3, 6]
        reward_chunks = split_reward_chunks(input_ids, decode_token)
        chunk_values = compute_chunk_advantages([3.0, 5.0, 5.0])

        values = assign_chunk_values_to_output_tokens(
            num_output_tokens=len(input_ids),
            chunk_token_spans=[chunk.token_span for chunk in reward_chunks],
            chunk_values=chunk_values,
            normalize_by_token_count=False,
        )

        self.assertEqual(values, [3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 5.0])

    def test_length_clipping_zeroes_advantages_but_not_rewards_on_token_spans(self) -> None:
        input_ids = [1, 2, 3, 4, 5, 3, 6]
        reward_chunks = split_reward_chunks(input_ids, decode_token)
        chunk_token_spans = [chunk.token_span for chunk in reward_chunks]
        prefix_scores = [5.0, 5.0, 1.0]
        chunk_rewards = compute_chunk_rewards(prefix_scores)
        raw_chunk_advantages = compute_chunk_advantages(prefix_scores)
        clipped_chunk_advantages, _, _ = maybe_clip_chunk_advantages_for_length(
            output_token_ids=input_ids,
            eos_token_id=99,
            chunk_advantages=raw_chunk_advantages,
            is_clip_length=True,
        )

        token_rewards = assign_chunk_values_to_output_tokens(
            num_output_tokens=len(input_ids),
            chunk_token_spans=chunk_token_spans,
            chunk_values=chunk_rewards,
            normalize_by_token_count=True,
        )
        token_advantages = assign_chunk_values_to_output_tokens(
            num_output_tokens=len(input_ids),
            chunk_token_spans=chunk_token_spans,
            chunk_values=clipped_chunk_advantages,
            normalize_by_token_count=True,
        )

        self.assertAlmostEqual(sum(token_rewards), sum(chunk_rewards))
        self.assertEqual(token_advantages, [0.0] * len(input_ids))

    def test_delimiter_completion_inside_token_assigns_whole_token_to_previous_chunk(self) -> None:
        input_ids = [12, 13, 14, 15]
        chunks = split_reward_chunks(input_ids, decode_token)

        self.assertEqual([chunk.token_span for chunk in chunks], [(0, 2), (2, 3), (3, 4)])


class ChunkAdvantageTest(unittest.TestCase):
    def test_chunk_rewards_are_prefix_deltas(self) -> None:
        self.assertEqual(compute_chunk_rewards([2.0, 4.0, 3.0]), [2.0, 2.0, -1.0])

    def test_empty_scores(self) -> None:
        self.assertEqual(compute_chunk_advantages([]), [])
        self.assertEqual(compute_chunk_rewards([]), [])

    def test_single_score(self) -> None:
        self.assertEqual(compute_chunk_advantages([4.0]), [4.0])

    def test_flat_scores_keep_credit(self) -> None:
        self.assertEqual(compute_chunk_advantages([5.0, 5.0, 5.0, 5.0]), [5.0, 5.0, 5.0, 5.0])

    def test_mixed_scores_follow_heuristic(self) -> None:
        self.assertEqual(compute_chunk_advantages([2.0, 4.0, 3.0]), [2.0, 5.0, 2.0])

    def test_negative_and_positive_scores(self) -> None:
        self.assertEqual(compute_chunk_advantages([-1.0, 2.0, 1.0]), [-1.0, 4.0, 0.0])

    def test_length_clipping_keeps_rewards_but_zeros_advantages_on_overflow(self) -> None:
        prefix_scores = [5.0, 5.0, 1.0]
        chunk_rewards = compute_chunk_rewards(prefix_scores)
        raw_chunk_advantages = compute_chunk_advantages(prefix_scores)

        clipped_advantages, is_overflow, is_length_clipped = maybe_clip_chunk_advantages_for_length(
            output_token_ids=[11, 12, 13],
            eos_token_id=99,
            chunk_advantages=raw_chunk_advantages,
            is_clip_length=True,
        )

        self.assertEqual(chunk_rewards, [5.0, 0.0, -4.0])
        self.assertEqual(raw_chunk_advantages, [5.0, 1.0, -3.0])
        self.assertEqual(clipped_advantages, [0.0, 0.0, 0.0])
        self.assertTrue(is_overflow)
        self.assertTrue(is_length_clipped)

    def test_length_clipping_disabled_preserves_advantages(self) -> None:
        raw_chunk_advantages = compute_chunk_advantages([5.0, 5.0, 1.0])

        clipped_advantages, is_overflow, is_length_clipped = maybe_clip_chunk_advantages_for_length(
            output_token_ids=[11, 12, 13],
            eos_token_id=99,
            chunk_advantages=raw_chunk_advantages,
            is_clip_length=False,
        )

        self.assertEqual(clipped_advantages, raw_chunk_advantages)
        self.assertTrue(is_overflow)
        self.assertFalse(is_length_clipped)

    def test_eos_present_preserves_advantages(self) -> None:
        raw_chunk_advantages = compute_chunk_advantages([5.0, 5.0, 1.0])

        clipped_advantages, is_overflow, is_length_clipped = maybe_clip_chunk_advantages_for_length(
            output_token_ids=[11, 99, 13],
            eos_token_id=99,
            chunk_advantages=raw_chunk_advantages,
            is_clip_length=True,
        )

        self.assertEqual(clipped_advantages, raw_chunk_advantages)
        self.assertFalse(is_overflow)
        self.assertFalse(is_length_clipped)


class QwenTokenizerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if AutoTokenizer is None:
            raise unittest.SkipTest("transformers is unavailable")

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B-Instruct-2507",
                local_files_only=True,
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Qwen tokenizer unavailable locally: {exc}")

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "ignore_data"
            / "proof_judge_design"
            / "qwen4b_50rows.jsonl"
        )
        if not fixture_path.exists():
            raise unittest.SkipTest(f"fixture file missing: {fixture_path}")

        with fixture_path.open() as f:
            row = json.loads(next(f))
        full_text = row["qwen_solution"]
        cutoff = full_text.find("\n\n####")
        cls.fixture_text = full_text[:cutoff] if cutoff != -1 else full_text
        if cls.fixture_text.count("###") < 2:
            raise unittest.SkipTest("fixture text does not contain enough ### delimiters")

    def test_real_qwen_solution_prefix_splits_cleanly(self) -> None:
        text = self.fixture_text
        tokenizer = self.tokenizer
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        def decode_one(token_id: int) -> str:
            return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        chunks = split_reward_chunks(input_ids, decode_one, delimiter="### ")
        decoded_chunks = [
            "".join(decode_one(token_id) for token_id in input_ids[start:end])
            for start, end in [chunk.token_span for chunk in chunks]
        ]
        reconstructed = "".join(decode_one(token_id) for token_id in input_ids)

        self.assertEqual("".join(decoded_chunks), reconstructed)
        self.assertEqual(len(chunks), text.count("### ") + 1)
        self.assertTrue(all("### " in chunk for chunk in decoded_chunks[:-1]))
        self.assertNotIn("### ", decoded_chunks[-1])


if __name__ == "__main__":
    unittest.main()
