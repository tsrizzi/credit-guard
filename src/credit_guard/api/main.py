"""API de scoring de credito: contrato -> pontuacao -> log de predicao -> metricas."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_guard.api.metrics import (
    metrics_endpoint,
    metrics_middleware,
    record_predictions,
    record_rejections,
)
from credit_guard.api.schemas import PredictRequest, PredictResponse
from credit_guard.contracts.validation import validate_batch
from credit_guard.logging_utils import get_logger
from credit_guard.model.scoring import load_model, score_frame

log = get_logger("api")


def _prediction_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))


def _append_prediction_log(scored: pd.DataFrame, batch_id: str | None, version: str) -> None:
    path = _prediction_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    with open(path, "a", encoding="utf-8") as fh:
        for row in scored.itertuples(index=False):
            fh.write(
                json.dumps(
                    {
                        "ts": ts,
                        "batch_id": batch_id,
                        "client_id": row.client_id,
                        "probability_default": float(row.probability_default),
                        "decision": row.decision,
                        "model_version": version,
                    }
                )
                + "\n"
            )


def create_app(models_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Credit Guard - API de scoring",
        description="Scoring de credito protegido por contrato de dados (Pandera).",
        version="1.0.0",
    )
    try:
        model, meta = load_model(models_dir)
        app.state.model, app.state.meta = model, meta
    except Exception as exc:  # noqa: BLE001 - API sobe degradada e reporta no /health
        app.state.model, app.state.meta = None, {"model_version": None}
        log.error("model_load_failed", extra={"error": repr(exc)})

    app.middleware("http")(metrics_middleware)

    @app.exception_handler(RequestValidationError)
    async def _shape_error(request: Request, exc: RequestValidationError):
        body = exc.body if isinstance(exc.body, dict) else {}
        records = body.get("records")
        record_rejections({"schema_shape": 1}, len(records) if isinstance(records, list) else 0)
        log.warning("request_shape_rejected", extra={"errors": len(exc.errors())})
        errors = json.loads(json.dumps(exc.errors(), default=str))
        return JSONResponse(
            status_code=422, content={"detail": {"error": "schema_shape", "errors": errors}}
        )

    @app.get("/health")
    def health():
        if app.state.model is None:
            return JSONResponse(
                status_code=503, content={"status": "degraded", "model_version": None}
            )
        return {"status": "ok", "model_version": app.state.meta["model_version"]}

    @app.get("/metrics")
    def metrics():
        return metrics_endpoint()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if app.state.model is None:
            return JSONResponse(status_code=503, content={"detail": "modelo indisponivel"})
        df = pd.DataFrame([r.model_dump() for r in req.records])
        result = validate_batch(df)
        if not result.passed:
            record_rejections(result.violations_by_check, result.n_rows)
            payload = result.to_dict()
            log.warning(
                "contract_rejected",
                extra={
                    "batch_id": req.batch_id,
                    "rows": result.n_rows,
                    "invalid_rows": result.n_invalid_rows,
                    "violations_by_check": result.violations_by_check,
                },
            )
            return JSONResponse(
                status_code=422,
                content={
                    "detail": {
                        "error": "contract_violation",
                        "violations": payload["failure_cases"],
                        "violations_by_check": payload["violations_by_check"],
                    }
                },
            )
        meta = app.state.meta
        scored = score_frame(app.state.model, meta, result.valid)
        record_predictions(scored["probability_default"], scored["decision"])
        try:
            _append_prediction_log(scored, req.batch_id, meta["model_version"])
        except OSError as exc:  # o score ja foi calculado: falha de log nao vira 500
            log.warning("prediction_log_failed", extra={"error": repr(exc)})
        log.info(
            "prediction_served",
            extra={
                "batch_id": req.batch_id,
                "rows": len(scored),
                "approved": int((scored["decision"] == "approve").sum()),
                "model_version": meta["model_version"],
            },
        )
        return {"model_version": meta["model_version"], "predictions": scored.to_dict("records")}

    return app


app = create_app()
