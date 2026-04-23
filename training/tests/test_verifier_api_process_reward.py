from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipelinerl"
    / "domains"
    / "math"
    / "verifier_api.py"
)
SPEC = spec_from_file_location("verifier_api", MODULE_PATH)
assert SPEC and SPEC.loader
verifier_api = module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier_api
SPEC.loader.exec_module(verifier_api)

format_variants_block = verifier_api.format_variants_block
parse_process_judge_response = verifier_api.parse_process_judge_response


class ProcessJudgeFormattingTest(unittest.TestCase):
    def test_format_variants_block_numbers_variants(self) -> None:
        text = format_variants_block(["first path", "second path"])

        self.assertIn("--- Variant 1 ---\nfirst path", text)
        self.assertIn("--- Variant 2 ---\nsecond path", text)

    def test_parse_valid_process_judge_response(self) -> None:
        response = """Why: The prefix matches Variant 2 by introducing induction on m and stating the inductive hypothesis.\nAligned path: Variant 2\nScore: 4"""

        parsed = parse_process_judge_response(response)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.score, 4)
        self.assertEqual(parsed.aligned_path, "Variant 2")
        self.assertIn("induction on m", parsed.why)

    def test_parse_score_only_response(self) -> None:
        parsed = parse_process_judge_response("Score: 3")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.score, 3)
        self.assertIsNone(parsed.aligned_path)
        self.assertIsNone(parsed.why)

    def test_missing_score_returns_none(self) -> None:
        response = """Why: The prefix is mostly setup and does not yet identify a valid invariant.\nAligned path: None"""

        self.assertIsNone(parse_process_judge_response(response))


if __name__ == "__main__":
    unittest.main()
