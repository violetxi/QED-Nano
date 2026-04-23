from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


MAX_TRACKED_STEPS = 10

AVG_NUM_STEPS_METRIC = "exp_rl/avg_num_steps"
AVG_LAST_STEP_REWARD_METRIC = "exp_rl/avg_last_step_reward"
CLIPPED_ADVANTAGE_FRAC_METRIC = "exp_rl/clipped_advantage_frac"
NUM_STEPS_FREQ_METRICS = [f"exp_rl/num_steps_freq_{step}" for step in range(1, MAX_TRACKED_STEPS + 1)]
NUM_STEPS_FREQ_GT_METRIC = f"exp_rl/num_steps_freq_gt{MAX_TRACKED_STEPS}"
STEP_ADVANTAGE_METRICS = [
    f"exp_rl/avg_advantage_step_{step}" for step in range(1, MAX_TRACKED_STEPS + 1)
]
STEP_ADVANTAGE_GT_METRIC = f"exp_rl/avg_advantage_step_gt{MAX_TRACKED_STEPS}"

ALL_EXP_RL_METRICS = (
    [AVG_NUM_STEPS_METRIC]
    + [CLIPPED_ADVANTAGE_FRAC_METRIC]
    + NUM_STEPS_FREQ_METRICS
    + [NUM_STEPS_FREQ_GT_METRIC, AVG_LAST_STEP_REWARD_METRIC]
    + STEP_ADVANTAGE_METRICS
    + [STEP_ADVANTAGE_GT_METRIC]
)


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def extract_exp_rl_metric_values(process_reward: Mapping[str, Any] | None) -> dict[str, list[float]]:
    if not process_reward:
        return {}

    chunk_rewards = _as_float_list(process_reward.get("chunk_rewards") or [])
    chunk_advantages = _as_float_list(process_reward.get("chunk_advantages") or [])
    prefix_scores = _as_float_list(process_reward.get("prefix_scores") or [])

    num_steps = max(len(chunk_rewards), len(chunk_advantages), len(prefix_scores))
    if num_steps == 0:
        return {}

    metrics = defaultdict(list)
    metrics[AVG_NUM_STEPS_METRIC].append(float(num_steps))
    metrics[CLIPPED_ADVANTAGE_FRAC_METRIC].append(
        1.0 if bool(process_reward.get("is_length_clipped", False)) else 0.0
    )

    for step in range(1, MAX_TRACKED_STEPS + 1):
        metrics[f"exp_rl/num_steps_freq_{step}"].append(1.0 if num_steps == step else 0.0)
    metrics[NUM_STEPS_FREQ_GT_METRIC].append(1.0 if num_steps > MAX_TRACKED_STEPS else 0.0)

    if chunk_rewards:
        metrics[AVG_LAST_STEP_REWARD_METRIC].append(float(chunk_rewards[-1]))

    for step in range(1, MAX_TRACKED_STEPS + 1):
        if len(chunk_advantages) >= step:
            metrics[f"exp_rl/avg_advantage_step_{step}"].append(float(chunk_advantages[step - 1]))
    if len(chunk_advantages) > MAX_TRACKED_STEPS:
        metrics[STEP_ADVANTAGE_GT_METRIC].extend(
            float(value) for value in chunk_advantages[MAX_TRACKED_STEPS:]
        )

    return dict(metrics)


def aggregate_exp_rl_metric_values(metric_values: Mapping[str, list[float]]) -> dict[str, float]:
    aggregated = {metric_name: 0.0 for metric_name in ALL_EXP_RL_METRICS}
    for metric_name, values in metric_values.items():
        if not values:
            continue
        aggregated[metric_name] = float(sum(values) / len(values))
    return aggregated
