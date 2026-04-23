import unittest

from pipelinerl.domains.math.process_reward_logging import (
    AVG_LAST_STEP_REWARD_METRIC,
    AVG_NUM_STEPS_METRIC,
    CLIPPED_ADVANTAGE_FRAC_METRIC,
    NUM_STEPS_FREQ_GT_METRIC,
    STEP_ADVANTAGE_GT_METRIC,
    aggregate_exp_rl_metric_values,
    extract_exp_rl_metric_values,
)


class ProcessRewardLoggingTest(unittest.TestCase):
    def test_extracts_expected_metrics_for_short_trace(self):
        metrics = extract_exp_rl_metric_values(
            {
                "prefix_scores": [2.0, 4.0, 3.0],
                "chunk_rewards": [2.0, 2.0, -1.0],
                "chunk_advantages": [2.0, 4.0, 2.0],
                "is_length_clipped": False,
            }
        )

        self.assertEqual(metrics[AVG_NUM_STEPS_METRIC], [3.0])
        self.assertEqual(metrics[CLIPPED_ADVANTAGE_FRAC_METRIC], [0.0])
        self.assertEqual(metrics["exp_rl/num_steps_freq_3"], [1.0])
        self.assertEqual(metrics["exp_rl/num_steps_freq_2"], [0.0])
        self.assertEqual(metrics[NUM_STEPS_FREQ_GT_METRIC], [0.0])
        self.assertEqual(metrics[AVG_LAST_STEP_REWARD_METRIC], [-1.0])
        self.assertEqual(metrics["exp_rl/avg_advantage_step_1"], [2.0])
        self.assertEqual(metrics["exp_rl/avg_advantage_step_2"], [4.0])
        self.assertEqual(metrics["exp_rl/avg_advantage_step_3"], [2.0])
        self.assertNotIn(STEP_ADVANTAGE_GT_METRIC, metrics)

    def test_aggregates_step_bins_and_gt10_bucket(self):
        aggregated_values = {}
        for process_reward in (
            {
                "chunk_rewards": [1.0, 0.0, -2.0],
                "chunk_advantages": [1.0, 3.0, 1.0],
                "is_length_clipped": False,
            },
            {
                "chunk_rewards": [0.5] * 12,
                "chunk_advantages": [float(i) for i in range(1, 13)],
                "is_length_clipped": True,
            },
        ):
            for metric_name, metric_values in extract_exp_rl_metric_values(process_reward).items():
                aggregated_values.setdefault(metric_name, []).extend(metric_values)

        aggregated = aggregate_exp_rl_metric_values(aggregated_values)

        self.assertAlmostEqual(aggregated[AVG_NUM_STEPS_METRIC], 7.5)
        self.assertAlmostEqual(aggregated[CLIPPED_ADVANTAGE_FRAC_METRIC], 0.5)
        self.assertAlmostEqual(aggregated["exp_rl/num_steps_freq_3"], 0.5)
        self.assertAlmostEqual(aggregated[NUM_STEPS_FREQ_GT_METRIC], 0.5)
        self.assertAlmostEqual(aggregated[AVG_LAST_STEP_REWARD_METRIC], -0.75)
        self.assertAlmostEqual(aggregated["exp_rl/avg_advantage_step_1"], 1.0)
        self.assertAlmostEqual(aggregated["exp_rl/avg_advantage_step_3"], 2.0)
        self.assertAlmostEqual(aggregated["exp_rl/avg_advantage_step_10"], 10.0)
        self.assertAlmostEqual(aggregated[STEP_ADVANTAGE_GT_METRIC], 11.5)

    def test_empty_inputs_return_zeroed_aggregate(self):
        aggregated = aggregate_exp_rl_metric_values({})
        self.assertEqual(aggregated[AVG_NUM_STEPS_METRIC], 0.0)
        self.assertEqual(aggregated[CLIPPED_ADVANTAGE_FRAC_METRIC], 0.0)
        self.assertEqual(aggregated[AVG_LAST_STEP_REWARD_METRIC], 0.0)
        self.assertEqual(aggregated[NUM_STEPS_FREQ_GT_METRIC], 0.0)
        self.assertEqual(aggregated[STEP_ADVANTAGE_GT_METRIC], 0.0)


if __name__ == "__main__":
    unittest.main()
