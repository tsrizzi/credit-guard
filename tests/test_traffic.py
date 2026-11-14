import math

from credit_guard.simulation.traffic import alert_state_samples, iter_batches, month_gauges
from tests.conftest import make_raw_like

SUMMARY = {
    "month": 3,
    "drift_table": [
        {"feature": "EXT_SOURCE_2", "psi": 0.31},
        {"feature": "AMT_CREDIT", "psi": 0.04},
    ],
    "share_drifted": 0.25,
    "auc": 0.74,
    "approval_rate": 0.72,
    "di_gender": 0.9,
    "di_age": 0.6,
    "alerts": [{"name": "FeatureDriftCritical", "severity": "critical"}],
}


def test_iter_batches_json_safe():
    df = make_raw_like(n=450, seed=41)
    df.loc[0, "EXT_SOURCE_1"] = float("nan")
    batches = list(iter_batches(df, batch_size=200))
    assert len(batches) == math.ceil(450 / 200)
    first = batches[0][0]
    assert first["EXT_SOURCE_1"] is None
    for batch in batches:
        for rec in batch:
            assert "CODE_GENDER" not in rec and "TARGET" not in rec


def test_month_gauges_names():
    samples = month_gauges(SUMMARY)
    names = {name for name, _, _ in samples}
    assert {
        "credit_drift_psi",
        "credit_drift_share",
        "credit_model_auc",
        "credit_approval_rate",
    } <= names
    psi = [s for s in samples if s[0] == "credit_drift_psi"]
    assert {"feature": "EXT_SOURCE_2", "month": "03"} in [labels for _, _, labels in psi]
    assert ("credit_fairness_di", 0.6, {"attribute": "age_band", "month": "03"}) in samples


def test_alert_state_marks_active_and_inactive():
    samples = alert_state_samples(SUMMARY)
    by_alert = {labels["alert"]: value for _, value, labels in samples}
    assert by_alert["FeatureDriftCritical"] == 1.0
    assert by_alert["ModelAucDrop"] == 0.0
