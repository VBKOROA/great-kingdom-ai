from __future__ import annotations

import traceback

import pytest
from great_kingdom_ai.learner_prefetch import PrefetchIterator


def test_prefetch_iterator_strips_producer_traceback_from_errors() -> None:
    def producer() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError) as exc_info:
        list(PrefetchIterator(producer=producer, count=1))

    frames = traceback.extract_tb(exc_info.value.__traceback__)

    assert "producer" not in {frame.name for frame in frames}
