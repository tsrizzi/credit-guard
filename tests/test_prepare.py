import numpy as np
import pandas as pd

from credit_guard.config import load_params
from credit_guard.contracts.validation import validate_batch
from credit_guard.data.prepare import (
    pseudonymize_ids,
    quarantine_invalid,
    separate_protected,
    split_frames,
)
from tests.conftest import make_raw_like


def test_pseudonymize_deterministic_and_salted():
    ids = pd.Series([100002, 100003])
    a = pseudonymize_ids(ids, "salt-a")
    assert a.tolist() == pseudonymize_ids(ids, "salt-a").tolist()
    assert a.tolist() != pseudonymize_ids(ids, "salt-b").tolist()
    assert all(len(x) == 16 for x in a)
    assert all("100002" not in x for x in a)


def test_quarantine_removes_only_invalid(raw_like_df):
    df = raw_like_df.copy()
    df.loc[0, "AMT_INCOME_TOTAL"] = np.nan
    df.loc[1, "CODE_GENDER"] = "XNA"
    df.loc[2, "NAME_FAMILY_STATUS"] = "Unknown"
    clean, quarantined, result = quarantine_invalid(df)
    assert len(quarantined) == 3
    assert len(clean) + 3 == len(df)
    assert validate_batch(clean, include_protected=True).passed is True
    assert result.passed is False


def test_split_fractions_and_disjoint():
    df = make_raw_like(n=2000, seed=1)
    parts = split_frames(df, load_params())
    n = len(df)
    assert abs(len(parts["train"]) - 0.60 * n) <= 1
    assert abs(len(parts["test"]) - 0.15 * n) <= 1
    assert abs(len(parts["holdout"]) - 0.25 * n) <= 1
    ids = [set(p["client_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    overall = df["TARGET"].mean()
    for p in parts.values():
        assert abs(p["TARGET"].mean() - overall) < 0.02


def test_separate_protected_removes_gender(raw_like_df):
    features, protected = separate_protected(raw_like_df)
    assert "CODE_GENDER" not in features.columns
    assert list(protected.columns) == ["client_id", "CODE_GENDER", "DAYS_BIRTH"]
    assert "DAYS_BIRTH" in features.columns  # idade e feature do modelo
