"""Small console progress helper shared by training entrypoints."""

from __future__ import annotations

import time


class PipelinePrinter:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started_at = time.monotonic()
        self.bar_width = 28

    def title(self, text: str) -> None:
        if self.enabled:
            print(f"\n== {text} ==", flush=True)

    def step(self, text: str) -> None:
        if self.enabled:
            print(f"  -> {text}", flush=True)

    def done(self, text: str) -> None:
        if self.enabled:
            print(f"  ok {text}", flush=True)

    def metric(self, key: str, value: object) -> None:
        if self.enabled:
            print(f"  {key:<18} {value}", flush=True)

    def progress(self, key: str, current: int, target: int, *, detail: str = "") -> None:
        if not self.enabled:
            return
        percent = 100.0 if target <= 0 else min(100.0, current / target * 100.0)
        filled = (
            self.bar_width
            if target <= 0
            else round(self.bar_width * min(current, target) / target)
        )
        bar = "#" * filled + "." * (self.bar_width - filled)
        suffix = f"  {detail}" if detail else ""
        print(
            f"  {key:<18} [{bar}] {current:>6}/{target:<6} {percent:>6.1f}%{suffix}",
            flush=True,
        )

    def elapsed(self) -> str:
        return _format_duration(time.monotonic() - self.started_at)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, second = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{second:02d}s"
    if minute:
        return f"{minute}m{second:02d}s"
    return f"{second}s"


__all__ = ["PipelinePrinter"]
