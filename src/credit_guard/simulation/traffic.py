"""Simulador de trafego de producao (demo ao vivo).

Envia os meses de producao, em sequencia, para a API de scoring. No mes `--dirty-month`
envia tambem o lote sujo (o contrato bloqueia -> pico de rejeicoes no Grafana). Ao fim de
cada mes publica no Pushgateway as metricas de drift/modelo/fairness daquele mes (lidas do
resultado do estagio `monitor`) e o estado atual dos alertas.

Uso:
    python -m credit_guard.simulation.traffic --api-url http://localhost:8000 --pause-seconds 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator

import pandas as pd
import requests

from credit_guard.config import DATA_DIR, ID_COLUMN, INPUT_COLUMNS, REPORTS_DIR, load_params
from credit_guard.logging_utils import get_logger
from credit_guard.monitoring.alerts import RUNBOOKS
from credit_guard.observability.publish import Sample, push_metrics

log = get_logger("traffic")
SEND_COLUMNS = [ID_COLUMN, *INPUT_COLUMNS]
MODEL_ALERTS = [
    "FeatureDriftCritical",
    "DriftedShareHigh",
    "ModelAucDrop",
    "ApprovalRateShift",
    "FairnessDisparateImpact",
]


def _json_value(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[list[dict]]:
    cols = [c for c in SEND_COLUMNS if c in df.columns]
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size][cols]
        yield [{k: _json_value(v) for k, v in rec.items()} for rec in chunk.to_dict("records")]


def month_gauges(summary: dict) -> list[Sample]:
    m = f"{summary['month']:02d}" if isinstance(summary["month"], int) else str(summary["month"])
    samples: list[Sample] = [
        ("credit_drift_psi", float(r["psi"]), {"feature": r["feature"], "month": m})
        for r in summary.get("drift_table", [])
    ]
    for name, key in (
        ("credit_drift_share", "share_drifted"),
        ("credit_model_auc", "auc"),
        ("credit_approval_rate", "approval_rate"),
    ):
        if summary.get(key) is not None:
            samples.append((name, float(summary[key]), {"month": m}))
    for attr, key in (("CODE_GENDER", "di_gender"), ("age_band", "di_age")):
        if summary.get(key) is not None:
            samples.append(
                ("credit_fairness_di", float(summary[key]), {"attribute": attr, "month": m})
            )
    return samples


def alert_state_samples(summary: dict) -> list[Sample]:
    active = {a["name"]: a.get("severity", "warning") for a in summary.get("alerts", [])}
    return [
        (
            "credit_alerts_active",
            1.0 if name in active else 0.0,
            {"alert": name, "runbook": RUNBOOKS[name]},
        )
        for name in MODEL_ALERTS
    ]


def _post(session: requests.Session, url: str, records: list[dict], batch_id: str) -> int:
    try:
        resp = session.post(
            f"{url}/predict", json={"records": records, "batch_id": batch_id}, timeout=30
        )
        return resp.status_code
    except requests.RequestException as exc:
        log.error("api_unreachable", extra={"error": str(exc)})
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulador de trafego de producao.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--pushgateway", default="http://localhost:9091")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--pause-seconds", type=float, default=20)
    parser.add_argument("--dirty-month", type=int, default=3)
    args = parser.parse_args(argv)

    n_months = load_params()["simulation"]["n_months"]
    session = requests.Session()
    ref = json.loads(
        (REPORTS_DIR / "monitoring" / "reference_summary.json").read_text(encoding="utf-8")
    )
    push_metrics(
        "credit_reference",
        [
            ("credit_model_auc", ref["auc"], {"month": "ref"}),
            ("credit_approval_rate", ref["approval_rate"], {"month": "ref"}),
        ],
        gateway=args.pushgateway,
    )
    dirty = pd.read_parquet(DATA_DIR / "dirty" / "dirty_batch.parquet")

    for m in range(1, n_months + 1):
        month_df = pd.read_parquet(DATA_DIR / "production" / f"month_{m:02d}.parquet")
        status: dict[int, int] = {}
        for batch in iter_batches(month_df, args.batch_size):
            code = _post(session, args.api_url, batch, f"month_{m:02d}")
            status[code] = status.get(code, 0) + 1
        if m == args.dirty_month:
            for batch in iter_batches(dirty, 50):
                code = _post(session, args.api_url, batch, "dirty_batch")
                status[code] = status.get(code, 0) + 1

        summary = json.loads(
            (REPORTS_DIR / "monitoring" / f"month_{m:02d}" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        push_metrics(
            "credit_monitoring", month_gauges(summary), args.pushgateway, {"month": f"{m:02d}"}
        )
        push_metrics("credit_alerts", alert_state_samples(summary), args.pushgateway)
        push_metrics(
            "credit_monitoring_progress",
            [("credit_monitoring_current_month", m, {})],
            args.pushgateway,
        )
        log.info(
            "month_replayed",
            extra={
                "month": m,
                "batches_by_status": status,
                "alerts": [a["name"] for a in summary["alerts"]],
            },
        )
        if m < n_months:
            time.sleep(args.pause_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
