"""Small bounded prefetcher for learner CPU batch preparation."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class _PrefetchError:
    error: BaseException


class PrefetchIterator(Generic[T]):
    """Run a producer callable ahead of the consumer on a background thread."""

    def __init__(
        self,
        *,
        producer: Callable[[], T],
        count: int,
        max_prefetch: int = 1,
    ) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if max_prefetch <= 0:
            raise ValueError("max_prefetch must be positive")
        self._producer = producer
        self._count = count
        self._queue: queue.Queue[T | _PrefetchError | None] = queue.Queue(max_prefetch)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._closed = False

    def __iter__(self) -> Iterator[T]:
        self._thread.start()
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if isinstance(item, _PrefetchError):
                    raise item.error
                yield item
        finally:
            self.close()

    def close(self) -> None:
        self._closed = True

    def _run(self) -> None:
        try:
            for _ in range(self._count):
                if self._closed:
                    return
                self._queue.put(self._producer())
        except BaseException as exc:
            self._queue.put(_PrefetchError(exc))
        finally:
            self._queue.put(None)


__all__ = ["PrefetchIterator"]
