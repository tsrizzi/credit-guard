import pandas as pd
import pytest

from credit_guard.monitoring.fairness import age_band, disparate_impact, fairness_table


def test_age_band_labels():
    days = pd.Series([-22, -30, -40, -50, -60]) * 365.25
    assert age_band(days).tolist() == ["<25", "25-34", "35-44", "45-54", "55+"]


def test_fairness_table_and_di():
    n = 100
    ids = [f"id{i}" for i in range(2 * n)]
    decision = (["approve"] * 80 + ["reject"] * 20) + (["approve"] * 60 + ["reject"] * 40)
    scored = pd.DataFrame({"client_id": ids, "decision": decision, "TARGET": [0] * (2 * n)})
    protected = pd.DataFrame(
        {
            "client_id": ids,
            "CODE_GENDER": ["F"] * n + ["M"] * n,
            "DAYS_BIRTH": [-40 * 365.25] * (2 * n),
        }
    )
    table = fairness_table(scored, protected)
    assert set(table.columns) >= {"attribute", "group", "n", "approval_rate", "default_rate", "tpr"}
    gender = table[table["attribute"] == "CODE_GENDER"].set_index("group")
    assert gender.loc["F", "approval_rate"] == pytest.approx(0.8)
    assert gender.loc["M", "approval_rate"] == pytest.approx(0.6)
    assert disparate_impact(table, "CODE_GENDER") == pytest.approx(0.75)
