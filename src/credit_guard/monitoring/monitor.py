"""Estagio `monitor`: monitoramento mes a mes da producao contra a referencia.

Para cada mes: contrato -> pontuacao -> drift (PSI/KS/JS proprios + Evidently) ->
performance (rotulos atrasados) -> fairness -> alertas -> run no MLflow.
Ao final: relatorio Evidently completo (referencia vs. ultimo mes) e atribuicao causal.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

from credit_guard.config import (
    CATEGORICAL_INPUTS,
    DATA_DIR,
    INPUT_COLUMNS,
    REPORTS_DIR,
    ROOT,
    TARGET_COLUMN,
    load_params,
)
from credit_guard.contracts.validation import validate_batch
from credit_guard.logging_utils import get_logger
from credit_guard.model.features import NUMERIC_FEATURES, engineer_features
from credit_guard.model.metrics import evaluate_scores
from credit_guard.model.scoring import load_model, score_frame
from credit_guard.monitoring.alerts import evaluate_alerts
from credit_guard.monitoring.causal import drift_attribution
from credit_guard.monitoring.evidently_reports import build_drift_report
from credit_guard.monitoring.fairness import disparate_impact, fairness_table
from credit_guard.monitoring.stats import feature_drift_table
from credit_guard.observability.pipeline import StageContext, track_stage
from credit_guard.observability.tracking import mlflow_best_effort

log = get_logger("monitor")

# Nos de choque do grafo causal: a intervencao desfaz o choque inteiro de uma vez,
# preservando a coerencia entre variaveis que se movem juntas (ex.: credito e preco do bem).
CAUSAL_SHOCKS = {
    "inflacao (credito, parcela, preco do bem)": ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
    "erosao da renda real": ["AMT_INCOME_TOTAL"],
    "desemprego (tipo de renda + emprego)": ["NAME_INCOME_TYPE", "DAYS_EMPLOYED"],
    "piora de bureau (EXT_SOURCE_2/3)": ["EXT_SOURCE_2", "EXT_SOURCE_3"],
}
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def _clean(obj):
    """Converte para JSON estrito (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _performance(df: pd.DataFrame, scored: pd.DataFrame, threshold: float) -> dict:
    if TARGET_COLUMN not in df.columns or df[TARGET_COLUMN].nunique() < 2:
        approved = scored["decision"] == "approve"
        return {
            "auc": None,
            "ks": None,
            "approval_rate": float(approved.mean()),
            "default_rate_approved": None,
        }
    m = evaluate_scores(df[TARGET_COLUMN], scored["probability_default"], threshold)
    return {
        "auc": m["roc_auc"],
        "ks": m["ks"],
        "approval_rate": m["approval_rate"],
        "default_rate_approved": m["default_rate_approved"],
        "default_rate": m["default_rate"],
    }


def build_reference_summary(
    reference: pd.DataFrame, protected: pd.DataFrame, model, meta: dict
) -> dict:
    scored = score_frame(model, meta, reference)
    perf = _performance(reference, scored, meta["threshold"])
    fair = fairness_table(
        scored.assign(**{TARGET_COLUMN: reference[TARGET_COLUMN].to_numpy()}), protected
    )
    return {
        "auc": perf["auc"],
        "ks": perf["ks"],
        "approval_rate": perf["approval_rate"],
        "default_rate": perf.get("default_rate"),
        "top_features": meta.get("top_features", []),
        "di_gender": disparate_impact(fair, "CODE_GENDER"),
        "di_age": disparate_impact(fair, "age_band"),
        "fairness": fair.to_dict("records"),
    }


def monitor_month(
    month: int,
    current: pd.DataFrame,
    reference: pd.DataFrame,
    protected: pd.DataFrame,
    model,
    meta: dict,
    mon: dict,
    out_dir: Path,
    ref_summary: dict,
) -> dict:
    month_dir = Path(out_dir) / f"month_{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_batch(current)
    scored = score_frame(model, meta, current)

    ref_eng, cur_eng = engineer_features(reference), engineer_features(current)
    drift = feature_drift_table(
        ref_eng,
        cur_eng,
        NUMERIC_FEATURES,
        CATEGORICAL_INPUTS,
        bins=mon["psi_bins"],
        alpha=mon["ks_alpha"],
    )
    drift.to_csv(month_dir / "drift_table.csv", index=False)
    evidently = build_drift_report(
        ref_eng, cur_eng, NUMERIC_FEATURES, CATEGORICAL_INPUTS, month_dir / "drift_report.html"
    )

    perf = _performance(current, scored, meta["threshold"])
    with_target = (
        scored.assign(**{TARGET_COLUMN: current[TARGET_COLUMN].to_numpy()})
        if TARGET_COLUMN in current
        else scored
    )
    fair = fairness_table(with_target, protected)
    fair.to_csv(month_dir / "fairness.csv", index=False)

    summary = {
        "month": month,
        "n_rows": len(current),
        "contract_passed": contract.passed,
        "rejection_rate": contract.n_invalid_rows / max(contract.n_rows, 1),
        "drift_table": drift.to_dict("records"),
        "share_drifted": float(drift["drift"].mean()),
        "drifted_features": drift.loc[drift["drift"], "feature"].tolist(),
        "evidently": {k: evidently[k] for k in ("drifted_features", "share_drifted", "n_features")},
        "auc": perf["auc"],
        "ks": perf["ks"],
        "approval_rate": perf["approval_rate"],
        "default_rate_approved": perf["default_rate_approved"],
        "default_rate": perf.get("default_rate"),
        "mean_score": float(scored["probability_default"].mean()),
        "fairness": fair.to_dict("records"),
        "di_gender": disparate_impact(fair, "CODE_GENDER"),
        "di_age": disparate_impact(fair, "age_band"),
    }
    summary["alerts"] = [a.to_dict() for a in evaluate_alerts(summary, ref_summary, mon)]
    summary = _clean(summary)
    (month_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _log_month_to_mlflow(mlflow, summary: dict, meta: dict, month_dir: Path) -> None:
    with mlflow.start_run(run_name=f"month_{summary['month']:02d}"):
        mlflow.set_tags(
            {"model_version": meta["model_version"], "month": f"{summary['month']:02d}"}
        )
        metrics = {f"psi__{r['feature']}": r["psi"] for r in summary["drift_table"]}
        for key in (
            "share_drifted",
            "auc",
            "ks",
            "approval_rate",
            "di_gender",
            "di_age",
            "mean_score",
            "rejection_rate",
        ):
            if summary.get(key) is not None:
                metrics[key] = summary[key]
        metrics["n_alerts"] = len(summary["alerts"])
        mlflow.log_metrics(metrics, step=summary["month"])
        mlflow.log_artifacts(str(month_dir), artifact_path="monitoring")


def _run(ctx: StageContext) -> None:
    import mlflow

    params = load_params()
    mon, sim = params["monitoring"], params["simulation"]
    model, meta = load_model()
    reference = pd.read_parquet(DATA_DIR / "reference" / "test.parquet")
    protected = pd.concat(
        [
            pd.read_parquet(DATA_DIR / "protected" / f)
            for f in ("reference.parquet", "holdout.parquet")
        ]
    )
    out_dir = REPORTS_DIR / "monitoring"
    ref_summary = _clean(build_reference_summary(reference, protected, model, meta))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_summary.json").write_text(
        json.dumps(ref_summary, indent=2), encoding="utf-8"
    )

    def _setup():
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("credit-monitoring")

    tracking_ok = mlflow_best_effort(_setup, "set_experiment")
    rows, last = [], None
    for m in range(1, sim["n_months"] + 1):
        current = pd.read_parquet(DATA_DIR / "production" / f"month_{m:02d}.parquet")
        summary = monitor_month(
            m, current, reference, protected, model, meta, mon, out_dir, ref_summary
        )
        if tracking_ok:
            mlflow_best_effort(
                lambda: _log_month_to_mlflow(mlflow, summary, meta, out_dir / f"month_{m:02d}"),
                f"log_month_{m:02d}",
            )
        ctx.add_rows("monitored", summary["n_rows"])
        top_psi = {r["feature"]: r["psi"] for r in summary["drift_table"]}
        rows.append(
            {
                "month": m,
                "auc": summary["auc"],
                "ks": summary["ks"],
                "approval_rate": summary["approval_rate"],
                "default_rate": summary["default_rate"],
                "mean_score": summary["mean_score"],
                "share_drifted": summary["share_drifted"],
                "n_drifted": len(summary["drifted_features"]),
                "drifted_features": ";".join(summary["drifted_features"]),
                "di_gender": summary["di_gender"],
                "di_age": summary["di_age"],
                "alerts": ";".join(a["name"] for a in summary["alerts"]),
                **{f"psi__{k}": v for k, v in top_psi.items()},
            }
        )
        log.info(
            "month_monitored",
            extra={"month": m, "alerts": rows[-1]["alerts"], "auc": summary["auc"]},
        )
        last = current
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    # relatorio Evidently completo (drift + qualidade + performance) referencia vs. ultimo mes
    ref_full = engineer_features(reference).assign(
        probability_default=score_frame(model, meta, reference)["probability_default"].to_numpy()
    )
    cur_full = engineer_features(last).assign(
        probability_default=score_frame(model, meta, last)["probability_default"].to_numpy()
    )
    build_drift_report(
        ref_full,
        cur_full,
        NUMERIC_FEATURES,
        CATEGORICAL_INPUTS,
        REPORTS_DIR / "drift" / f"reference_vs_month_{sim['n_months']:02d}.html",
        target=TARGET_COLUMN,
        prediction="probability_default",
    )

    causal_dir = REPORTS_DIR / "causal"
    causal_dir.mkdir(parents=True, exist_ok=True)
    month_tag = f"month_{sim['n_months']:02d}"
    for name, spec in (
        (f"attribution_{month_tag}", INPUT_COLUMNS),
        (f"attribution_shocks_{month_tag}", CAUSAL_SHOCKS),
    ):
        causal = drift_attribution(model, meta, reference, last, spec, seed=params["seed"])
        causal.to_csv(causal_dir / f"{name}.csv", index=False)
        (causal_dir / f"{name}.md").write_text(
            causal.round(4).to_markdown(index=False), encoding="utf-8"
        )


def main() -> None:
    with track_stage("monitor") as ctx:
        _run(ctx)


if __name__ == "__main__":
    main()
