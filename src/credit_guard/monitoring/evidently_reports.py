"""Relatorios do Evidently AI (0.7.x): data drift + qualidade + (opcional) performance."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

from credit_guard.config import load_params

PSI_THRESHOLD = 0.2  # mesma faixa "significativo" de stats.psi_band
_VALUE_DRIFT = re.compile(r"ValueDrift\(column=(?P<col>.+?),method=(?P<method>.+?),threshold=")


def _drift_flag(method: str, value: float, threshold: float) -> bool:
    # metodos de p-valor (K-S, qui-quadrado, Z) -> drift se p < limiar; distancias -> drift se >=
    return value < threshold if "p_value" in method else value >= threshold


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    out_html: Path,
    target: str | None = None,
    prediction: str | None = None,
) -> dict:
    columns = numeric + categorical
    # "dataset drift" com o mesmo limiar do alerta DriftedShareHigh (nao o padrao de 50%)
    drift_share = load_params()["monitoring"]["drifted_share_alert"]
    classification = None
    presets = [
        DataDriftPreset(
            columns=columns, method="psi", threshold=PSI_THRESHOLD, drift_share=drift_share
        ),
        DataSummaryPreset(columns=columns),
    ]
    keep = list(columns)
    if target and prediction:
        classification = [
            BinaryClassification(target=target, prediction_probas=prediction, pos_label=1)
        ]
        presets.append(ClassificationPreset())
        keep += [target, prediction]

    definition = DataDefinition(
        numerical_columns=numeric + ([prediction] if classification else []),
        categorical_columns=categorical + ([target] if classification else []),
        classification=classification,
    )
    snapshot = Report(presets).run(
        current_data=Dataset.from_pandas(current[keep], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[keep], data_definition=definition),
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    per_feature = {}
    for metric in snapshot.dict()["metrics"]:
        match = _VALUE_DRIFT.match(metric["metric_name"])
        if not match or match["col"] not in columns:
            continue
        value, threshold = float(metric["value"]), float(metric["config"]["threshold"])
        per_feature[match["col"]] = {
            "method": match["method"],
            "value": value,
            "threshold": threshold,
            "drift": _drift_flag(match["method"], value, threshold),
        }
    drifted = [c for c, v in per_feature.items() if v["drift"]]
    share = len(drifted) / max(len(per_feature), 1)
    return {
        "drifted_features": drifted,
        "share_drifted": share,
        "dataset_drift": share >= drift_share,
        "drift_share_threshold": drift_share,
        "n_features": len(per_feature),
        "per_feature": per_feature,
    }
