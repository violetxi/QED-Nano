from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


MASKED_TOKEN_ID = -100
DEFAULT_STEP_DELIMITER = "### "


@dataclass(frozen=True)
class RewardChunk:
    index: int
    token_span: tuple[int, int]


def validate_ordered_spans(length: int, spans: Sequence[tuple[int, int]]) -> None:
    for start, end in spans:
        if start < 0 or end > length:
            raise ValueError(f"Span {start}:{end} is out of bounds for length {length}")
        if start > end:
            raise ValueError(f"Span {start}:{end} is invalid")
    for (start1, end1), (start2, end2) in zip(spans, spans[1:]):
        if start2 < end1:
            raise ValueError(f"Spans {start1}:{end1} and {start2}:{end2} overlap")


def _find_delimiter_starts(
    input_ids: torch.Tensor,
    delimiter_token_id: int,
) -> list[int]:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")
    if not isinstance(delimiter_token_id, int):
        raise ValueError(f"delimiter_token_id must be an int, got {type(delimiter_token_id).__name__}")
    return torch.nonzero(input_ids == delimiter_token_id, as_tuple=False).flatten().tolist()


def split_reward_chunks(
    input_ids: torch.Tensor,
    delimiter_token_id: int,
) -> list[RewardChunk]:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")

    num_tokens = int(input_ids.numel())
    delimiter_starts = _find_delimiter_starts(
        input_ids=input_ids,
        delimiter_token_id=delimiter_token_id,
    )
    if not delimiter_starts:
        token_spans = [(0, num_tokens)]
    else:
        token_spans = [
            (start, next_start)
            for start, next_start in zip(delimiter_starts, delimiter_starts[1:])
        ]
        token_spans.append((delimiter_starts[-1], num_tokens))
    validate_ordered_spans(num_tokens, token_spans)

    return [
        RewardChunk(
            index=chunk_idx,
            token_span=token_span,
        )
        for chunk_idx, token_span in enumerate(token_spans)
    ]


def compute_chunk_advantages(prefix_scores: Sequence[float]) -> list[float]:
    """
    Convert prefix scores into heuristic per-chunk advantages:

    a_1 = s_1
    a_t = (s_t - s_{t-1}) + s_final for t > 1
    """

    if not prefix_scores:
        return []

    final_score = float(prefix_scores[-1])
    advantages = [float(prefix_scores[0])]
    for idx in range(1, len(prefix_scores)):
        advantages.append(float(prefix_scores[idx] - prefix_scores[idx - 1] + final_score))
    return advantages


def normalize_prefix_scores(
    prefix_scores: Sequence[float],
    min_score: float = 1.0,
    max_score: float = 5.0,
) -> list[float]:
    """
    Map process-judge scores onto [0, 1] before reward/advantage deltas.

    Gemini returns a 1-5 score. Synthetic failure scores use 0, which should stay
    at 0 instead of becoming negative under the 1-5 affine transform.
    """

    if max_score <= min_score:
        raise ValueError("max_score must be greater than min_score")

    score_range = max_score - min_score
    normalized_scores: list[float] = []
    for score in prefix_scores:
        score_value = float(score)
        if score_value <= 0.0:
            normalized_scores.append(0.0)
            continue
        clipped_score = min(max(score_value, min_score), max_score)
        normalized_scores.append((clipped_score - min_score) / score_range)
    return normalized_scores


def validate_unit_interval(values: Sequence[float], name: str) -> None:
    """Validate values that are expected to be probabilities or normalized scores."""

    for idx, value in enumerate(values):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name}[{idx}]={value} is outside [0, 1]")


def maybe_clip_chunk_advantages_for_length(
    output_token_ids: Sequence[int],
    eos_token_id: int | None,
    chunk_advantages: Sequence[float],
    is_clip_length: bool = False,
) -> tuple[list[float], bool, bool]:
    """
    Optionally zero process-reward advantages for overflowed generations.

    Overflow is defined as the generated completion token ids not containing the
    tokenizer EOS token. If EOS is unavailable, clipping is skipped.
    """

    clipped_advantages = [float(value) for value in chunk_advantages]
    if not isinstance(eos_token_id, int):
        return clipped_advantages, False, False

    is_overflow = bool(output_token_ids) and eos_token_id not in output_token_ids
    if not is_clip_length or not is_overflow:
        return clipped_advantages, is_overflow, False

    return [0.0] * len(clipped_advantages), is_overflow, True


def compute_chunk_rewards(prefix_scores: Sequence[float]) -> list[float]:
    """
    Use the normalized score for each prefix as that chunk's reward.
    """

    return [float(score) for score in prefix_scores]


def assign_chunk_values_to_output_tokens(
    num_output_tokens: int,
    chunk_token_spans: Sequence[tuple[int, int]],
    chunk_values: Sequence[float],
    normalize_by_token_count: bool = True,
) -> list[float]:
    """
    Project chunk-level values onto output tokens only.

    If normalize_by_token_count is True, every token in a chunk receives
    chunk_value / n_chunk_tokens so the total mass assigned to that chunk is
    equal to the original chunk value.
    """

    if len(chunk_token_spans) != len(chunk_values):
        raise ValueError(
            "chunk_token_spans and chunk_values must have the same length, got "
            f"{len(chunk_token_spans)} and {len(chunk_values)}"
        )
    validate_ordered_spans(num_output_tokens, chunk_token_spans)

    values = [0.0] * num_output_tokens
    for span_idx, ((start, end), chunk_value) in enumerate(zip(chunk_token_spans, chunk_values)):
        if normalize_by_token_count:
            token_count = end - start
            if token_count == 0:
                raise ValueError(f"Chunk token span {span_idx} has no output tokens")
            per_token_value = chunk_value / token_count
            for token_idx in range(start, end):
                values[token_idx] = per_token_value
            continue

        for token_idx in range(start, end):
            values[token_idx] = float(chunk_value)
    return values


def expand_output_token_values_to_labels(
    labels: Sequence[int],
    output_token_values: Sequence[float],
    masked_token_id: int = MASKED_TOKEN_ID,
) -> list[float]:
    """
    Expand output-token values onto the full label vector, leaving prompt tokens at zero.
    """

    num_labeled_tokens = sum(1 for label in labels if label != masked_token_id)
    if num_labeled_tokens != len(output_token_values):
        raise ValueError(
            f"labels contain {num_labeled_tokens} unmasked tokens, but got {len(output_token_values)} output-token values"
        )

    expanded = [0.0] * len(labels)
    output_idx = 0
    for label_idx, label in enumerate(labels):
        if label == masked_token_id:
            continue
        expanded[label_idx] = float(output_token_values[output_idx])
        output_idx += 1
    return expanded
