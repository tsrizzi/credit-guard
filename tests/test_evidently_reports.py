import numpy as np
import pandas as pd

from credit_guard.monitoring.evidently_reports import build_drift_report


def _frames(n=2000):
    rng = np.random.default_rng(11)
    ref = pd.DataFrame(
        {"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n), "c": rng.choice(["A", "B"], n)}
    )
    cur = pd.DataFrame(
        {"x": rng.normal(1.5, 1, n), "y": rng.normal(0, 1, n), "c": rng.choice(["A", "B"], n)}
    )
    return ref, cur


def test_drift_report_detects_shift(tmp_path):
    ref, cur = _frames()
    out = tmp_path / "drift.html"
    summary = build_drift_report(ref, cur, ["x", "y"], ["c"], out)
    assert out.exists() and out.stat().st_size > 10_000
    assert "x" in summary["drifted_features"]
    assert "y" not in summary["drifted_features"]
    assert summary["n_features"] == 3
    assert 0 < summary["share_drifted"] < 1


def test_drift_report_with_classification(tmp_path):
    ref, cur = _frames()
    rng = np.random.default_rng(12)
    for df in (ref, cur):
        df["TARGET"] = rng.integers(0, 2, len(df))
        df["probability_default"] = np.clip(df["TARGET"] * 0.4 + rng.uniform(0, 0.6, len(df)), 0, 1)
    out = tmp_path / "full.html"
    summary = build_drift_report(
        ref, cur, ["x", "y"], ["c"], out, target="TARGET", prediction="probability_default"
    )
    assert out.exists()
    assert "x" in summary["drifted_features"]
