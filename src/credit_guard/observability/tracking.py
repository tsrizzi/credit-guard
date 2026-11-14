"""Chamadas ao MLflow em modo best-effort: rastreamento indisponivel nao derruba o pipeline."""

from __future__ import annotations

from collections.abc import Callable

from credit_guard.logging_utils import get_logger

log = get_logger("tracking")


def mlflow_best_effort(action: Callable[[], object], what: str) -> bool:
    try:
        action()
        return True
    except Exception as exc:  # noqa: BLE001 - MlflowException, erros de rede, banco travado...
        log.warning("mlflow_unavailable", extra={"action": what, "error": repr(exc)})
        return False
