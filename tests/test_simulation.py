import numpy as np
import pandas as pd

from credit_guard.config import load_params
from credit_guard.contracts.validation import validate_batch
from credit_guard.simulation.dirty import DEFECT_CHECKS, make_dirty_batch
from credit_guard.simulation.drift import apply_concept_drift, apply_data_drift
from tests.conftest import make_raw_like

SIM = load_params()["simulation"]


def _features(df):
    return df.drop(columns=["CODE_GENDER"])


def test_data_drift_monotonic():
    df = _features(make_raw_like(n=3000, seed=3))
    months = [apply_data_drift(df, m, SIM, np.random.default_rng(m)) for m in range(1, 7)]
    income = [m["AMT_INCOME_TOTAL"].mean() for m in months]
    credit = [m["AMT_CREDIT"].mean() for m in months]
    unemployed = [(m["NAME_INCOME_TYPE"] == "Unemployed").mean() for m in months]
    assert all(a > b for a, b in zip(income, income[1:], strict=False))
    assert all(a < b for a, b in zip(credit, credit[1:], strict=False))
    assert all(a < b for a, b in zip(unemployed, unemployed[1:], strict=False))
    assert income[0] < df["AMT_INCOME_TOTAL"].mean()


def test_data_drift_does_not_mutate_input(raw_like_df):
    df = _features(raw_like_df)
    before = df.copy()
    apply_data_drift(df, 6, SIM, np.random.default_rng(0))
    pd.testing.assert_frame_equal(df, before)


def test_drifted_month_passes_contract(raw_like_df):
    drifted = apply_data_drift(_features(raw_like_df), 6, SIM, np.random.default_rng(0))
    assert validate_batch(drifted).passed is True


def test_concept_drift_only_segment_and_after_start():
    df = _features(make_raw_like(n=3000, seed=4))
    month3 = apply_concept_drift(df, 3, SIM, np.random.default_rng(0))
    assert month3["TARGET"].equals(df["TARGET"])

    month6 = apply_concept_drift(df, 6, SIM, np.random.default_rng(0))
    changed = month6["TARGET"] != df["TARGET"]
    assert changed.any()
    assert (df.loc[changed, "NAME_INCOME_TYPE"] == SIM["concept_segment_income_type"]).all()
    assert (df.loc[changed, "TARGET"] == 0).all() and (month6.loc[changed, "TARGET"] == 1).all()
    pd.testing.assert_frame_equal(month6.drop(columns="TARGET"), df.drop(columns="TARGET"))


def test_dirty_batch_fails_with_each_defect():
    clean = _features(make_raw_like(n=600, seed=5))
    quarantined = _features(make_raw_like(n=10, seed=6))
    quarantined["client_id"] = [f"q{i}" for i in range(10)]
    quarantined.loc[quarantined.index[:3], "AMT_INCOME_TOTAL"] = 117_000_000
    batch, manifest = make_dirty_batch(clean, quarantined, 140, np.random.default_rng(0))
    assert len(batch) == 140
    assert manifest["real_quarantined"] == 3
    result = validate_batch(batch)
    assert result.passed is False
    checks = set(result.violations["check"])
    for defect, count in manifest.items():
        if defect != "real_quarantined" and count:
            assert DEFECT_CHECKS[defect] in checks, defect


def test_income_psi_grows_gradually_with_tied_values():
    """Rendas do Home Credit sao muito repetidas (multiplos de 4.500); o drift deve crescer aos poucos."""
    from credit_guard.monitoring.stats import psi

    df = _features(make_raw_like(n=5000, seed=8))
    tied = [67_500, 90_000, 112_500, 135_000, 157_500, 180_000, 225_000, 270_000, 315_000]
    df["AMT_INCOME_TOTAL"] = np.random.default_rng(9).choice(tied, size=len(df)).astype(float)
    values = [
        psi(
            df["AMT_INCOME_TOTAL"],
            apply_data_drift(df, m, SIM, np.random.default_rng(m))["AMT_INCOME_TOTAL"],
        )
        for m in range(1, 7)
    ]
    assert values[0] < 0.10, values
    assert all(a <= b for a, b in zip(values, values[1:], strict=False)), values
