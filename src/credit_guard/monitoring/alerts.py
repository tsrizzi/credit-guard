"""Regras de alerta de degradacao. Os nomes sao os mesmos nas regras do Prometheus e no README."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

RUNBOOKS = {
    "FeatureDriftCritical": "README.md#runbook-feature-drift-critical",
    "DriftedShareHigh": "README.md#runbook-drifted-share-high",
    "ModelAucDrop": "README.md#runbook-model-auc-drop",
    "ApprovalRateShift": "README.md#runbook-approval-rate-shift",
    "ContractRejectionHigh": "README.md#runbook-contract-rejection-high",
    "FairnessDisparateImpact": "README.md#runbook-fairness-disparate-impact",
    "ApiLatencyHigh": "README.md#runbook-api-latency-high",
    "PipelineStageFailed": "README.md#runbook-pipeline-stage-failed",
}
PSI_CRITICAL = 0.20
DI_MIN = 0.80


@dataclass
class Alert:
    name: str
    severity: str
    value: float
    threshold: float
    message: str
    runbook: str

    def to_dict(self) -> dict:
        return asdict(self)


def _alert(name, severity, value, threshold, message) -> Alert:
    return Alert(name, severity, float(value), float(threshold), message, RUNBOOKS[name])


def evaluate_alerts(month_summary: dict, reference_summary: dict, mon: dict) -> list[Alert]:
    s, ref = month_summary, reference_summary
    alerts: list[Alert] = []

    top = set(ref.get("top_features", []))
    critical = [
        r for r in s.get("drift_table", []) if r["feature"] in top and r["psi"] >= PSI_CRITICAL
    ]
    if critical:
        worst = max(critical, key=lambda r: r["psi"])
        names = ", ".join(r["feature"] for r in critical)
        alerts.append(
            _alert(
                "FeatureDriftCritical",
                "critical",
                worst["psi"],
                PSI_CRITICAL,
                f"PSI >= 0,20 em feature top: {names}",
            )
        )

    if s.get("share_drifted", 0) >= mon["drifted_share_alert"]:
        alerts.append(
            _alert(
                "DriftedShareHigh",
                "warning",
                s["share_drifted"],
                mon["drifted_share_alert"],
                "fracao alta de features em drift",
            )
        )

    auc = s.get("auc")
    if auc is not None and not math.isnan(auc) and ref["auc"] - auc >= mon["auc_drop_alert"]:
        alerts.append(
            _alert(
                "ModelAucDrop",
                "critical",
                ref["auc"] - auc,
                mon["auc_drop_alert"],
                f"AUC caiu de {ref['auc']:.3f} para {auc:.3f}",
            )
        )

    shift = abs(s.get("approval_rate", ref["approval_rate"]) - ref["approval_rate"])
    if shift >= mon["approval_shift_alert"]:
        alerts.append(
            _alert(
                "ApprovalRateShift",
                "warning",
                shift,
                mon["approval_shift_alert"],
                "taxa de aprovacao deslocou da referencia",
            )
        )

    if s.get("rejection_rate", 0) >= mon["rejection_rate_alert"]:
        alerts.append(
            _alert(
                "ContractRejectionHigh",
                "critical",
                s["rejection_rate"],
                mon["rejection_rate_alert"],
                "contrato rejeitando dados acima do normal",
            )
        )

    # Fairness: dispara quando o DI esta abaixo da regra dos 80% E piorou em relacao a referencia.
    # Um vies ja presente na referencia e tratado na governanca (README), nao realertado todo mes.
    drop_limit = mon.get("fairness_di_drop_alert", 0.05)
    breaches = []
    for key, label in (("di_gender", "genero"), ("di_age", "faixa etaria")):
        value, ref_value = s.get(key), ref.get(key)
        if value is None or math.isnan(value) or value >= DI_MIN:
            continue
        if ref_value is None or math.isnan(ref_value) or ref_value - value >= drop_limit:
            breaches.append((value, label))
    if breaches:
        value, _ = min(breaches)
        labels = ", ".join(lbl for _, lbl in breaches)
        alerts.append(
            _alert(
                "FairnessDisparateImpact",
                "warning",
                value,
                DI_MIN,
                f"disparate impact < 0,8 e piorando vs. referencia ({labels})",
            )
        )
    return alerts
