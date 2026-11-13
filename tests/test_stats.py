import math

import numpy as np
import pandas as pd

from credit_guard.monitoring.stats import (
    chi2_pvalue,
    feature_drift_table,
    js_distance,
    ks_test,
    psi,
    psi_band,
)

RNG = np.random.default_rng(7)
REF = pd.Series(RNG.normal(0, 1, 5000))
SHIFTED = pd.Series(RNG.normal(1, 1, 5000))
SAME = pd.Series(RNG.normal(0, 1, 5000))


def test_psi_identical_is_zero():
    assert psi(REF, REF) < 0.01


def test_psi_shifted_is_significant():
    value = psi(REF, SHIFTED)
    assert value >= 0.20
    assert psi_band(value) == "significant"


def test_psi_band_edges():
    assert psi_band(0.099) == "stable"
    assert psi_band(0.10) == "moderate"
    assert psi_band(0.20) == "significant"


def test_psi_finite_with_missing_category():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 500 + ["C"] * 500)
    value = psi(ref, cur)
    assert math.isfinite(value) and value > 0


def test_psi_finite_with_empty_bins():
    value = psi(REF, pd.Series(np.full(1000, 0.3)))
    assert math.isfinite(value) and value > 0


def test_psi_handles_nan_bucket():
    cur = REF.copy()
    cur.iloc[:2500] = np.nan
    assert psi(REF, cur) >= 0.20


def test_ks_detects_shift():
    _, p_shift = ks_test(REF, SHIFTED)
    _, p_same = ks_test(REF, SAME)
    assert p_shift < 0.05
    assert p_same > 0.05


def test_js_bounds():
    assert 0 <= js_distance(REF, SHIFTED) <= 1
    assert js_distance(REF, REF) < 0.05


def test_chi2_detects_category_shift():
    ref = pd.Series(["A"] * 700 + ["B"] * 300)
    cur = pd.Series(["A"] * 300 + ["B"] * 700)
    assert chi2_pvalue(ref, cur) < 0.05
    assert chi2_pvalue(ref, ref) > 0.05


def test_feature_drift_table_flags_only_shifted():
    ref = pd.DataFrame({"x": REF, "y": REF, "c": ["A", "B"] * 2500})
    cur = pd.DataFrame({"x": SHIFTED, "y": SAME, "c": ["A", "B"] * 2500})
    table = feature_drift_table(ref, cur, ["x", "y"], ["c"], bins=10, alpha=0.05)
    assert list(table.columns) == [
        "feature",
        "kind",
        "psi",
        "psi_band",
        "ks_stat",
        "ks_pvalue",
        "js",
        "chi2_pvalue",
        "drift",
    ]
    flagged = set(table.loc[table["drift"], "feature"])
    assert flagged == {"x"}
    assert table.iloc[0]["feature"] == "x"
