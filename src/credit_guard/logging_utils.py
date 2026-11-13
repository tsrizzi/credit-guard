"""Logger estruturado: uma linha JSON por evento, pronta para agregacao/busca."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _CurrentStderrHandler(logging.StreamHandler):
    """Escreve sempre no sys.stderr *atual* (sobrevive a redirecionamentos, ex.: pytest)."""

    _credit_guard = True

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"credit_guard.{name}")
    if not any(isinstance(h, _CurrentStderrHandler) for h in logger.handlers):
        handler = _CurrentStderrHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
