"""Rastreio de execucao dos estagios do pipeline: logs estruturados + metricas de saude."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from credit_guard.logging_utils import get_logger
from credit_guard.observability.publish import push_metrics

log = get_logger("pipeline")


@dataclass
class StageContext:
    stage: str
    rows: dict[str, int] = field(default_factory=dict)

    def add_rows(self, outcome: str, n: int) -> None:
        self.rows[outcome] = self.rows.get(outcome, 0) + int(n)


def _publish(ctx: StageContext, ok: bool, duration: float) -> None:
    samples = [
        ("credit_pipeline_stage_status", 1.0 if ok else 0.0, {"stage": ctx.stage}),
        ("credit_pipeline_stage_duration_seconds", duration, {"stage": ctx.stage}),
        ("credit_pipeline_stage_last_run_timestamp", time.time(), {"stage": ctx.stage}),
    ]
    samples += [
        ("credit_pipeline_rows", float(n), {"stage": ctx.stage, "outcome": outcome})
        for outcome, n in ctx.rows.items()
    ]
    push_metrics(f"pipeline_{ctx.stage}", samples)


@contextmanager
def track_stage(stage: str) -> Iterator[StageContext]:
    ctx = StageContext(stage)
    start = time.perf_counter()
    log.info("stage_started", extra={"stage": stage})
    try:
        yield ctx
    except Exception as exc:
        duration = time.perf_counter() - start
        log.error(
            "stage_failed",
            extra={
                "stage": stage,
                "duration_seconds": duration,
                "rows": ctx.rows,
                "error": repr(exc),
            },
        )
        _publish(ctx, ok=False, duration=duration)
        raise
    duration = time.perf_counter() - start
    log.info(
        "stage_succeeded", extra={"stage": stage, "duration_seconds": duration, "rows": ctx.rows}
    )
    _publish(ctx, ok=True, duration=duration)
