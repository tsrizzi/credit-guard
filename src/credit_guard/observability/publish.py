"""Publicacao best-effort de metricas: Prometheus Pushgateway (jobs batch) e Datadog opcional.

Nenhuma falha de destino externo pode derrubar o pipeline ou a API: em caso de erro,
registra um warning estruturado e segue.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from credit_guard.logging_utils import get_logger

log = get_logger("publish")

Sample = tuple[str, float, dict[str, str]]
SERVICE_TAG = "service:credit-guard"
_statsd = None


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_statsd():
    """Cliente DogStatsD quando DD_ENABLED=true; caso contrario None (Datadog desligado)."""
    global _statsd
    if not _truthy(os.environ.get("DD_ENABLED")):
        return None
    if _statsd is None:
        from datadog.dogstatsd import DogStatsd

        _statsd = DogStatsd(
            host=os.environ.get("DD_AGENT_HOST", "localhost"),
            port=int(os.environ.get("DD_DOGSTATSD_PORT", "8125")),
            constant_tags=[SERVICE_TAG],
        )
    return _statsd


def _as_samples(gauges: Mapping[str, tuple[float, dict]] | Iterable[Sample]) -> list[Sample]:
    if isinstance(gauges, Mapping):
        return [(name, value, labels) for name, (value, labels) in gauges.items()]
    return list(gauges)


def mirror_to_datadog(samples: list[Sample]) -> None:
    statsd = get_statsd()
    if statsd is None:
        return
    try:
        for name, value, labels in samples:
            statsd.gauge(name, value, tags=[f"{k}:{v}" for k, v in labels.items()])
    except Exception as exc:  # noqa: BLE001 - best effort
        log.warning("datadog_send_failed", extra={"error": str(exc)})


def push_metrics(
    job: str,
    gauges: Mapping[str, tuple[float, dict]] | Iterable[Sample],
    gateway: str | None = None,
    grouping_key: dict[str, str] | None = None,
) -> bool:
    samples = _as_samples(gauges)
    mirror_to_datadog(samples)
    gateway = gateway or os.environ.get("PUSHGATEWAY_URL")
    if not gateway:
        return False

    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    metrics: dict[str, Gauge] = {}
    for name, value, labels in samples:
        if name not in metrics:
            metrics[name] = Gauge(name, name, labelnames=sorted(labels), registry=registry)
        g = metrics[name]
        (g.labels(**labels) if labels else g).set(value)
    try:
        push_to_gateway(
            gateway, job=job, registry=registry, grouping_key=grouping_key or {}, timeout=3
        )
        return True
    except Exception as exc:  # noqa: BLE001 - best effort
        log.warning(
            "metrics_push_failed", extra={"job": job, "gateway": gateway, "error": str(exc)}
        )
        return False
