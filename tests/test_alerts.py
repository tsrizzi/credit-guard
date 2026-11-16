from credit_guard.config import load_params
from credit_guard.monitoring.alerts import RUNBOOKS, evaluate_alerts

MON = load_params()["monitoring"]
REF = {"auc": 0.76, "approval_rate": 0.80, "top_features": ["EXT_SOURCE_2", "EXT_SOURCE_3"]}


def _summary(**overrides):
    base = {
        "drift_table": [
            {"feature": "EXT_SOURCE_2", "psi": 0.02},
            {"feature": "AMT_CREDIT", "psi": 0.05},
        ],
        "share_drifted": 0.0,
        "auc": 0.76,
        "approval_rate": 0.80,
        "rejection_rate": 0.0,
        "di_gender": 0.95,
        "di_age": 0.90,
    }
    return {**base, **overrides}


def test_alerts_fire_and_silence():
    stable = evaluate_alerts(_summary(), REF, MON)
    assert stable == []

    drifted = _summary(
        drift_table=[
            {"feature": "EXT_SOURCE_2", "psi": 0.25},
            {"feature": "AMT_CREDIT", "psi": 0.05},
        ],
        auc=0.71,
    )
    alerts = evaluate_alerts(drifted, REF, MON)
    assert {a.name for a in alerts} == {"FeatureDriftCritical", "ModelAucDrop"}
    assert all(a.runbook for a in alerts)


def test_non_top_feature_drift_is_not_critical():
    s = _summary(drift_table=[{"feature": "AMT_CREDIT", "psi": 0.40}])
    assert "FeatureDriftCritical" not in {a.name for a in evaluate_alerts(s, REF, MON)}


def test_auc_alert_skipped_without_labels():
    assert evaluate_alerts(_summary(auc=None), REF, MON) == []


def test_other_rules():
    s = _summary(share_drifted=0.5, approval_rate=0.70, rejection_rate=0.2, di_gender=0.7)
    names = {a.name for a in evaluate_alerts(s, REF, MON)}
    assert names == {
        "DriftedShareHigh",
        "ApprovalRateShift",
        "ContractRejectionHigh",
        "FairnessDisparateImpact",
    }


def test_every_rule_has_runbook_anchor():
    for name, link in RUNBOOKS.items():
        assert link.startswith(
            "https://github.com/tsrizzi/credit-guard/blob/main/README.md#runbook-"
        ), name


def test_fairness_alert_only_when_drift_worsens_known_bias():
    ref = {**REF, "di_gender": 0.91, "di_age": 0.65}
    same_bias = _summary(di_gender=0.90, di_age=0.64)
    assert evaluate_alerts(same_bias, ref, MON) == []
    worse = _summary(di_gender=0.90, di_age=0.55)
    alerts = evaluate_alerts(worse, ref, MON)
    assert [a.name for a in alerts] == ["FairnessDisparateImpact"]
