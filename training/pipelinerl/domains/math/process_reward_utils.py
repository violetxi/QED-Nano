from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


MASKED_TOKEN_ID = -100
DEFAULT_STEP_DELIMITER = "### "
TokenDecoder = Callable[[int], str]


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


def _find_delimiter_boundaries(
    input_ids: Sequence[int],
    decode_token: TokenDecoder,
    delimiter: str = DEFAULT_STEP_DELIMITER,
) -> list[int]:
    if not delimiter:
        raise ValueError("delimiter must be a non-empty string")

    boundaries: list[int] = []
    prefix_text = ""
    search_from = 0

    for token_idx, token_id in enumerate(input_ids):
        prefix_text += decode_token(token_id)
        while True:
            delimiter_pos = prefix_text.find(delimiter, search_from)
            if delimiter_pos == -1:
                break
            boundary = token_idx + 1
            if boundaries and boundary == boundaries[-1]:
                raise ValueError(
                    "Multiple delimiter completions inside the same token are not representable in token space"
                )
            boundaries.append(boundary)
            search_from = delimiter_pos + len(delimiter)
    return boundaries


def split_reward_chunks(
    input_ids: Sequence[int],
    decode_token: TokenDecoder,
    delimiter: str = DEFAULT_STEP_DELIMITER,
) -> list[RewardChunk]:
    token_spans: list[tuple[int, int]] = []
    start = 0
    boundaries = _find_delimiter_boundaries(
        input_ids=input_ids,
        decode_token=decode_token,
        delimiter=delimiter,
    )
    for boundary in boundaries:
        if boundary <= start:
            raise ValueError(
                f"Delimiter boundary {boundary} does not advance past prior chunk start {start}"
            )
        token_spans.append((start, boundary))
        start = boundary
    if start == len(input_ids) and input_ids:
        raise ValueError("Trace ends with a delimiter, leaving an empty final chunk")
    token_spans.append((start, len(input_ids)))
    validate_ordered_spans(len(input_ids), token_spans)

    return [
        RewardChunk(
            index=chunk_idx,
            token_span=token_span,
        )
        for chunk_idx, token_span in enumerate(token_spans)
    ]


def compute_chunk_advantages(prefix_scores: Sequence[float]) -> list[float]:
    """
    Convert prefix scores into the heuristic chunk advantages:

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
