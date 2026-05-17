"""Public sampling helpers shared by reanalyze datasets."""

from __future__ import annotations

import random


def sample_indexes(
    size: int,
    batch_size: int,
    rng: random.Random,
    *,
    recent_fraction: float = 0.0,
    recent_window: int = 0,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > size:
        raise ValueError("batch_size exceeds reanalyze target snapshot size")
    if recent_fraction <= 0.0:
        return rng.sample(range(size), batch_size)
    if recent_fraction > 1.0:
        raise ValueError("recent_fraction must be in [0, 1]")
    if recent_window <= 0:
        raise ValueError("recent_window must be positive when recency sampling is enabled")

    recent_count = min(recent_window, size)
    old_count = size - recent_count
    recent_take = min(round(batch_size * recent_fraction), recent_count, batch_size)
    old_take = min(batch_size - recent_take, old_count)
    recent_take = batch_size - old_take
    if recent_take > recent_count:
        raise ValueError("not enough rows to satisfy recency-biased sample")
    recent_start = size - recent_count
    indexes = [recent_start + index for index in rng.sample(range(recent_count), recent_take)]
    indexes.extend(rng.sample(range(old_count), old_take))
    rng.shuffle(indexes)
    return indexes


__all__ = ["sample_indexes"]
