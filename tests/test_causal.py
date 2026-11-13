import numpy as np
import pandas as pd

from credit_guard.monitoring.causal import drift_attribution


class SigmoidOfA:
    def predict_proba(self, df):
        p = 1 / (1 + np.exp(-df["a"].to_numpy(float)))
        return np.column_stack([1 - p, p])


def test_drift_attribution_identifies_driver():
    rng = np.random.default_rng(3)
    n = 3000
    ref = pd.DataFrame(
        {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.choice(["x", "y"], n)}
    )
    cur = pd.DataFrame(
        {"a": rng.normal(1.5, 1, n), "b": rng.normal(0.8, 1, n), "c": rng.choice(["x", "y"], n)}
    )
    table = drift_attribution(SigmoidOfA(), {"threshold": 0.5}, ref, cur, ["a", "b", "c"])
    assert list(table.columns) == [
        "feature",
        "mean_score_drifted",
        "mean_score_restored",
        "delta_score",
        "delta_approval",
        "share_of_score_shift",
    ]
    top = table.iloc[0]
    assert top["feature"] == "a"
    assert top["share_of_score_shift"] > 0.8
    assert top["delta_approval"] > 0
    assert abs(table.set_index("feature").loc["b", "share_of_score_shift"]) < 0.05
