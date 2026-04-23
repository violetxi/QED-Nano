import unittest

from pipelinerl.finetune.data import preprocess_fn
from pipelinerl.finetune.rl import RLConfig, populate_rl_data, prepare_rl_fields


class PrepareRlFieldsProcessRewardTest(unittest.TestCase):
    def test_prepare_rl_fields_preserves_precomputed_vectors(self) -> None:
        encoding = {
            "input_ids": [10, 11, 12, 13],
            "labels": [-100, -100, 12, 13],
            "attention_mask": [1, 1, 1, 1],
        }

        result = prepare_rl_fields(
            encoding=encoding,
            reward=5.0,
            old_logprobs=[-0.3, -0.2],
            ref_logprobs=[-0.4, -0.25],
            token_rewards=[0.0, 0.0, 1.5, -0.5],
            token_advantages=[0.0, 0.0, 4.0, 2.0],
        )

        self.assertEqual(result["rewards"], [0.0, 0.0, 1.5, -0.5])
        self.assertEqual(result["advantages"], [0.0, 0.0, 4.0, 2.0])
        self.assertEqual(result["old_logprobs"], [0, 0, -0.3, -0.2])
        self.assertEqual(result["ref_logprobs"], [0, 0, -0.4, -0.25])

    def test_preprocess_fn_treats_empty_process_vectors_as_absent(self) -> None:
        entry = {
            "input_ids": [10, 11, 12, 13],
            "labels": [-100, -100, 12, 13],
            "reward": 5.0,
            "logprobs": [-0.3, -0.2],
            "ref_logprobs": [-0.4, -0.25],
            "token_rewards": [],
            "token_advantages": [],
        }

        result = preprocess_fn(
            entry=entry,
            tokenizer=object(),
            seq_length=32,
            is_rl=True,
        )

        self.assertEqual(result["rewards"], [5.0, 5.0, 5.0, 5.0])
        self.assertEqual(result["advantages"], [0.0, 0.0, 0.0, 0.0])


class PopulateRlDataProcessRewardTest(unittest.TestCase):
    def test_populate_rl_data_preserves_precomputed_rewards_and_advantages(self) -> None:
        dataset = [
            {
                "group_id": "g0",
                "rollout_index": 0,
                "step_index": 0,
                "input_ids": [10, 11, 12, 13],
                "labels": [-100, -100, 12, 13],
                "rewards": [0.0, 0.0, 1.5, -0.5],
                "advantages": [0.0, 0.0, 4.0, 2.0],
                "group_tokens": [0.0, 0.0, 0.0, 0.0],
                "overflow": [0.0, 0.0, 0.0, 0.0],
            },
            {
                "group_id": "g0",
                "rollout_index": 1,
                "step_index": 0,
                "input_ids": [20, 21, 22],
                "labels": [-100, 21, 22],
                "rewards": [0.0, 2.0, 1.0],
                "advantages": [0.0, 6.0, 3.0],
                "group_tokens": [0.0, 0.0, 0.0],
                "overflow": [0.0, 0.0, 0.0],
            },
        ]

        config = RLConfig(precomputed_token_advantages=True)
        populated = populate_rl_data(dataset=dataset, eos_token_id=13, config=config)

        self.assertEqual(populated[0]["rewards"], [0.0, 0.0, 1.5, -0.5])
        self.assertEqual(populated[0]["advantages"], [0.0, 0.0, 4.0, 2.0])
        self.assertEqual(populated[1]["rewards"], [0.0, 2.0, 1.0])
        self.assertEqual(populated[1]["advantages"], [0.0, 6.0, 3.0])
        self.assertEqual(populated[0]["num_labels"], [2, 2, 2, 2])
        self.assertEqual(populated[1]["num_labels"], [2, 2, 2])
        self.assertEqual(populated[0]["group_tokens"], [3.5, 3.5, 3.5, 3.5])
        self.assertEqual(populated[1]["group_tokens"], [3.5, 3.5, 3.5])


if __name__ == "__main__":
    unittest.main()
