"""Periodic trajectory replay health monitor for async v2 runs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np

from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore

DEFAULT_REPLAY = Path("data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz")

Severity = Literal["warning", "critical"]


@dataclass(frozen=True)
class ReplayMonitorAlert:
    code: str
    severity: Severity
    message: str
    value: float | int | str | None = None
    threshold: float | int | str | None = None


@dataclass(frozen=True)
class ReplayMonitorConfig:
    replay_path: Path = DEFAULT_REPLAY
    interval_seconds: float = 300.0
    min_rows_for_root_check: int = 1
    min_root_available_fraction: float = 0.99
    max_illegal_mass_mean: float = 1e-6
    sharp_target_max_probability_mean: float = 0.98
    sharp_target_max_probability_p50: float = 0.9999
    min_target_prior_kl_mean: float = 1e-4
    max_prior_copy_mismatch_ratio: float = 0.01
    stale_checks: int = 3


@dataclass
class ReplayMonitorState:
    previous_rows: int | None = None
    unchanged_checks: int = 0


@dataclass(frozen=True)
class ReplayMonitorReport:
    checked_at: str
    replay_path: Path
    exists: bool
    rows: int | None
    episodes: int | None
    capacity: int | None
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    alerts: list[ReplayMonitorAlert] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(alert.severity == "critical" for alert in self.alerts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "replay_path": str(self.replay_path),
            "exists": self.exists,
            "rows": self.rows,
            "episodes": self.episodes,
            "capacity": self.capacity,
            "metrics": self.metrics,
            "alerts": [asdict(alert) for alert in self.alerts],
            "ok": self.ok,
        }


def check_replay_once(
    config: ReplayMonitorConfig,
    *,
    state: ReplayMonitorState | None = None,
) -> ReplayMonitorReport:
    if state is None:
        state = ReplayMonitorState()
    path = Path(config.replay_path)
    checked_at = _utc_now()
    if not path.exists():
        return ReplayMonitorReport(
            checked_at=checked_at,
            replay_path=path,
            exists=False,
            rows=None,
            episodes=None,
            capacity=None,
            alerts=[
                ReplayMonitorAlert(
                    code="replay_missing",
                    severity="warning",
                    message="trajectory replay file does not exist yet",
                )
            ],
        )

    try:
        replay = TrajectoryReplayStore.load(path)
    except Exception as exc:
        return ReplayMonitorReport(
            checked_at=checked_at,
            replay_path=path,
            exists=True,
            rows=None,
            episodes=None,
            capacity=None,
            alerts=[
                ReplayMonitorAlert(
                    code="replay_load_failed",
                    severity="critical",
                    message=f"failed to load trajectory replay: {exc}",
                )
            ],
        )

    alerts: list[ReplayMonitorAlert] = []
    metrics = _replay_metrics(replay)
    _update_stale_state(state, len(replay), alerts, config)
    _check_policy_metrics(metrics, alerts, config)
    _check_root_prior_metrics(replay, metrics, alerts, config)
    return ReplayMonitorReport(
        checked_at=checked_at,
        replay_path=path,
        exists=True,
        rows=len(replay),
        episodes=replay.episode_count,
        capacity=replay.capacity,
        metrics=metrics,
        alerts=alerts,
    )


def run_replay_monitor(
    config: ReplayMonitorConfig,
    *,
    once: bool = False,
    max_checks: int | None = None,
    json_output: bool = False,
) -> list[ReplayMonitorReport]:
    if config.interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    if max_checks is not None and max_checks <= 0:
        raise ValueError("max_checks must be positive")
    state = ReplayMonitorState()
    reports: list[ReplayMonitorReport] = []
    checks = 0
    while True:
        report = check_replay_once(config, state=state)
        reports.append(report)
        _print_report(report, json_output=json_output)
        checks += 1
        if once or (max_checks is not None and checks >= max_checks):
            return reports
        time.sleep(config.interval_seconds)


def _replay_metrics(replay: TrajectoryReplayStore) -> dict[str, float | int | str | None]:
    rows = len(replay)
    metrics: dict[str, float | int | str | None] = {
        "capacity_fraction": 0.0 if replay.capacity <= 0 else rows / replay.capacity,
    }
    if rows == 0:
        return metrics

    target = np.asarray(replay.policy_targets, dtype=np.float32)
    legal = np.asarray(replay.legal_masks, dtype=np.bool_)
    target_max = target.max(axis=1)
    illegal_mass = np.where(~legal, target, 0.0).sum(axis=1)
    metrics.update(
        {
            "target_entropy_mean": _mean(policy_entropy(target)),
            "target_entropy_p50": _percentile(policy_entropy(target), 50),
            "target_max_probability_mean": _mean(target_max),
            "target_max_probability_p50": _percentile(target_max, 50),
            "target_max_probability_p95": _percentile(target_max, 95),
            "target_support_p50": _percentile(np.count_nonzero(target > 1e-6, axis=1), 50),
            "illegal_mass_mean": _mean(illegal_mass),
            "illegal_mass_max": float(illegal_mass.max()),
        }
    )

    if replay.root_policy_logits is None:
        metrics.update(
            {
                "root_prior_available_rows": 0,
                "root_prior_available_fraction": 0.0,
                "root_prior_missing_rows": rows,
            }
        )
        return metrics

    root_logits = np.asarray(replay.root_policy_logits, dtype=np.float32)
    available = np.isfinite(root_logits).all(axis=1)
    available_rows = int(np.count_nonzero(available))
    metrics.update(
        {
            "root_prior_available_rows": available_rows,
            "root_prior_available_fraction": available_rows / rows,
            "root_prior_missing_rows": rows - available_rows,
        }
    )
    if available_rows == 0:
        return metrics

    prior = masked_softmax(root_logits[available], legal[available])
    target_available = target[available]
    prior_argmax = np.argmax(prior, axis=1)
    target_argmax = np.argmax(target_available, axis=1)
    mismatch = target_argmax != prior_argmax
    metrics.update(
        {
            "root_prior_entropy_mean": _mean(policy_entropy(prior)),
            "root_prior_max_probability_mean": _mean(prior.max(axis=1)),
            "target_prior_kl_mean": _mean(categorical_kl(target_available, prior)),
            "target_prior_kl_p50": _percentile(
                categorical_kl(target_available, prior),
                50,
            ),
            "target_prior_argmax_mismatch_ratio": float(np.mean(mismatch)),
        }
    )
    return metrics


def _update_stale_state(
    state: ReplayMonitorState,
    rows: int,
    alerts: list[ReplayMonitorAlert],
    config: ReplayMonitorConfig,
) -> None:
    if state.previous_rows == rows:
        state.unchanged_checks += 1
    else:
        state.previous_rows = rows
        state.unchanged_checks = 0
    if config.stale_checks > 0 and state.unchanged_checks >= config.stale_checks:
        alerts.append(
            ReplayMonitorAlert(
                code="replay_rows_not_growing",
                severity="warning",
                message="trajectory replay row count has not changed across checks",
                value=state.unchanged_checks,
                threshold=config.stale_checks,
            )
        )


def _check_policy_metrics(
    metrics: dict[str, float | int | str | None],
    alerts: list[ReplayMonitorAlert],
    config: ReplayMonitorConfig,
) -> None:
    illegal_mass_mean = _metric_float(metrics, "illegal_mass_mean")
    if illegal_mass_mean is not None and illegal_mass_mean > config.max_illegal_mass_mean:
        alerts.append(
            ReplayMonitorAlert(
                code="policy_target_illegal_mass",
                severity="critical",
                message="policy targets assign probability mass to illegal actions",
                value=illegal_mass_mean,
                threshold=config.max_illegal_mass_mean,
            )
        )
    target_max_mean = _metric_float(metrics, "target_max_probability_mean")
    if (
        target_max_mean is not None
        and target_max_mean >= config.sharp_target_max_probability_mean
    ):
        alerts.append(
            ReplayMonitorAlert(
                code="policy_target_too_sharp_mean",
                severity="warning",
                message="policy targets are very sharp on average",
                value=target_max_mean,
                threshold=config.sharp_target_max_probability_mean,
            )
        )
    target_max_p50 = _metric_float(metrics, "target_max_probability_p50")
    if target_max_p50 is not None and target_max_p50 >= config.sharp_target_max_probability_p50:
        alerts.append(
            ReplayMonitorAlert(
                code="policy_target_too_sharp_median",
                severity="warning",
                message="median policy target is nearly deterministic",
                value=target_max_p50,
                threshold=config.sharp_target_max_probability_p50,
            )
        )


def _check_root_prior_metrics(
    replay: TrajectoryReplayStore,
    metrics: dict[str, float | int | str | None],
    alerts: list[ReplayMonitorAlert],
    config: ReplayMonitorConfig,
) -> None:
    rows = len(replay)
    if rows < config.min_rows_for_root_check:
        return
    available_fraction = _metric_float(metrics, "root_prior_available_fraction")
    if replay.root_policy_logits is None:
        alerts.append(
            ReplayMonitorAlert(
                code="root_policy_logits_missing",
                severity="critical",
                message="trajectory replay does not include root_policy_logits",
                value=0,
                threshold=config.min_rows_for_root_check,
            )
        )
        return
    if (
        available_fraction is not None
        and available_fraction < config.min_root_available_fraction
    ):
        alerts.append(
            ReplayMonitorAlert(
                code="root_policy_logits_incomplete",
                severity="critical",
                message="root_policy_logits are missing or non-finite for some rows",
                value=available_fraction,
                threshold=config.min_root_available_fraction,
            )
        )
    kl_mean = _metric_float(metrics, "target_prior_kl_mean")
    mismatch_ratio = _metric_float(metrics, "target_prior_argmax_mismatch_ratio")
    if (
        kl_mean is not None
        and mismatch_ratio is not None
        and kl_mean <= config.min_target_prior_kl_mean
        and mismatch_ratio <= config.max_prior_copy_mismatch_ratio
    ):
        alerts.append(
            ReplayMonitorAlert(
                code="root_prior_target_copy_like",
                severity="warning",
                message="search target is very close to the root prior",
                value=kl_mean,
                threshold=config.min_target_prior_kl_mean,
            )
        )


def masked_softmax(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    masked = np.where(legal_masks, logits, -np.inf).astype(np.float32, copy=False)
    row_max = np.max(masked, axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("each row must have at least one legal root prior logit")
    exp = np.where(legal_masks, np.exp(masked - row_max), 0.0)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def categorical_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_clipped = np.clip(left, 1e-45, 1.0).astype(np.float32, copy=False)
    right_clipped = np.clip(right, 1e-45, 1.0).astype(np.float32, copy=False)
    return np.maximum(
        np.sum(left_clipped * (np.log(left_clipped) - np.log(right_clipped)), axis=1),
        0.0,
    ).astype(np.float32, copy=False)


def policy_entropy(policy: np.ndarray) -> np.ndarray:
    positive = policy > 0.0
    terms = np.zeros_like(policy, dtype=np.float32)
    terms[positive] = policy[positive] * np.log(np.clip(policy[positive], 1e-45, 1.0))
    return (-terms.sum(axis=1)).astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-replay-monitor-v2",
        description="Monitor async v2 trajectory replay health periodically.",
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-checks", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-rows-for-root-check", type=int, default=1)
    parser.add_argument("--min-root-available-fraction", type=float, default=0.99)
    parser.add_argument("--max-illegal-mass-mean", type=float, default=1e-6)
    parser.add_argument("--sharp-target-max-probability-mean", type=float, default=0.98)
    parser.add_argument("--sharp-target-max-probability-p50", type=float, default=0.9999)
    parser.add_argument("--min-target-prior-kl-mean", type=float, default=1e-4)
    parser.add_argument("--max-prior-copy-mismatch-ratio", type=float, default=0.01)
    parser.add_argument("--stale-checks", type=int, default=3)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = ReplayMonitorConfig(
        replay_path=args.replay,
        interval_seconds=args.interval_seconds,
        min_rows_for_root_check=args.min_rows_for_root_check,
        min_root_available_fraction=args.min_root_available_fraction,
        max_illegal_mass_mean=args.max_illegal_mass_mean,
        sharp_target_max_probability_mean=args.sharp_target_max_probability_mean,
        sharp_target_max_probability_p50=args.sharp_target_max_probability_p50,
        min_target_prior_kl_mean=args.min_target_prior_kl_mean,
        max_prior_copy_mismatch_ratio=args.max_prior_copy_mismatch_ratio,
        stale_checks=args.stale_checks,
    )
    try:
        run_replay_monitor(
            config,
            once=args.once,
            max_checks=args.max_checks,
            json_output=args.json,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(0)


def _print_report(report: ReplayMonitorReport, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.to_dict(), sort_keys=True), flush=True)
        return
    status = "OK" if report.ok and not report.alerts else "WARN"
    if any(alert.severity == "critical" for alert in report.alerts):
        status = "CRITICAL"
    print(
        f"[{report.checked_at}] replay={report.replay_path} status={status} "
        f"rows={report.rows} episodes={report.episodes}",
        flush=True,
    )
    for alert in report.alerts:
        suffix = ""
        if alert.value is not None or alert.threshold is not None:
            suffix = f" value={alert.value} threshold={alert.threshold}"
        print(f"  {alert.severity.upper()} {alert.code}: {alert.message}{suffix}", flush=True)


def _metric_float(metrics: dict[str, float | int | str | None], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _mean(values: np.ndarray) -> float:
    return float(np.asarray(values, dtype=np.float32).mean())


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


__all__ = [
    "ReplayMonitorAlert",
    "ReplayMonitorConfig",
    "ReplayMonitorReport",
    "ReplayMonitorState",
    "categorical_kl",
    "check_replay_once",
    "main",
    "masked_softmax",
    "policy_entropy",
    "run_replay_monitor",
]


if __name__ == "__main__":
    main()
