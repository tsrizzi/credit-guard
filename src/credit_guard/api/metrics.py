"""Metricas Prometheus da API (espelhadas no Datadog quando habilitado)."""

from __future__ import annotations

import time
from collections.abc import Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from credit_guard.observability.publish import get_statsd

REQUESTS = Counter("credit_requests_total", "Requisicoes recebidas", ["path", "status_code"])
LATENCY = Histogram(
    "credit_request_latency_seconds",
    "Latencia das requisicoes",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
CONTRACT_REJECTIONS = Counter(
    "credit_contract_rejections_total", "Violacoes do contrato, por regra", ["check"]
)
PREDICTIONS = Counter("credit_predictions_total", "Decisoes de credito", ["decision"])
SCORE = Histogram(
    "credit_score",
    "Distribuicao da probabilidade de inadimplencia",
    buckets=tuple(round(0.05 * i, 2) for i in range(1, 21)),
)
ROWS_SCORED = Counter("credit_rows_scored_total", "Linhas pontuadas")
ROWS_REJECTED = Counter(
    "credit_rows_rejected_total", "Linhas recebidas em lotes bloqueados pelo contrato"
)


def _dd(kind: str, name: str, value: float, tags: list[str] | None = None) -> None:
    statsd = get_statsd()
    if statsd is None:
        return
    try:
        getattr(statsd, kind)(name, value, tags=tags or [])
    except Exception:  # noqa: BLE001 - best effort
        pass


def record_rejections(by_check: dict[str, int], n_rows: int = 0) -> None:
    ROWS_REJECTED.inc(n_rows)
    _dd("increment", "credit.rows_rejected", n_rows)
    for check, n in by_check.items():
        CONTRACT_REJECTIONS.labels(check=check).inc(n)
        _dd("increment", "credit.contract_rejections", n, [f"check:{check}"])


def record_predictions(probas, decisions) -> None:
    for p, d in zip(probas, decisions, strict=True):
        SCORE.observe(float(p))
        PREDICTIONS.labels(decision=d).inc()
        _dd("histogram", "credit.score", float(p))
        _dd("increment", "credit.predictions", 1, [f"decision:{d}"])
    ROWS_SCORED.inc(len(decisions))


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(path=path).observe(elapsed)
        REQUESTS.labels(path=path, status_code=str(status)).inc()
        _dd("histogram", "credit.request_latency", elapsed, [f"path:{path}"])
        _dd("increment", "credit.requests", 1, [f"path:{path}", f"status_code:{status}"])


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
