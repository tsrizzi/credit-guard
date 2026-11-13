import json

import numpy as np
import pytest

from credit_guard.config import load_params
from credit_guard.data.prepare import separate_protected
from credit_guard.model.features import build_pipeline
from credit_guard.monitoring.monitor import build_reference_summary, monitor_month
from credit_guard.simulation.drift import apply_data_drift
from tests.conftest import make_raw_like

PARAMS = load_params()
SUMMARY_KEYS = {
    "month",
    "n_rows",
    "contract_passed",
    "rejection_rate",
    "drift_table",
    "share_drifted",
    "drifted_features",
    "evidently",
    "auc",
    "ks",
    "approval_rate",
    "default_rate_approved",
    "fairness",
    "di_gender",
    "di_age",
    "alerts",
}


@pytest.fixture(scope="module")
def setup():
    raw = make_raw_like(n=2400, seed=21)
    feats, protected = separate_protected(raw)
    reference, current = feats.iloc[:1200], feats.iloc[1200:]
    model_params = {**PARAMS["model"], "n_estimators": 10}
    model_params.pop("target_approval_rate")
    model = build_pipeline(model_params, seed=42).fit(reference, reference["TARGET"])
    meta = {"threshold": 0.15, "model_version": "test", "top_features": ["EXT_SOURCE_2"]}
    ref_summary = build_reference_summary(reference, protected, model, meta)
    return reference, current, protected, model, meta, ref_summary


def test_monitor_month_outputs(setup, tmp_path):
    reference, current, protected, model, meta, ref_summary = setup
    summary = monitor_month(
        1, current, reference, protected, model, meta, PARAMS["monitoring"], tmp_path, ref_summary
    )
    assert SUMMARY_KEYS <= set(summary)
    assert summary["contract_passed"] is True
    month_dir = tmp_path / "month_01"
    for name in ("drift_report.html", "drift_table.csv", "fairness.csv", "summary.json"):
        assert (month_dir / name).exists(), name
    json.loads((month_dir / "summary.json").read_text(encoding="utf-8"))


def test_monitor_month_detects_drift(setup, tmp_path):
    reference, current, protected, model, meta, ref_summary = setup
    drifted = apply_data_drift(current, 6, PARAMS["simulation"], np.random.default_rng(0))
    summary = monitor_month(
        6, drifted, reference, protected, model, meta, PARAMS["monitoring"], tmp_path, ref_summary
    )
    assert summary["share_drifted"] > 0
    assert {"AMT_INCOME_TOTAL", "credit_income_ratio"} & set(summary["drifted_features"])
