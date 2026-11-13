import numpy as np

from credit_guard.config import load_params
from credit_guard.model.features import MODEL_FEATURES, build_pipeline, engineer_features
from credit_guard.model.train import fit_threshold


def test_engineer_features_sentinel(raw_like_df):
    before = raw_like_df.copy()
    out = engineer_features(raw_like_df)
    pensioner = out[raw_like_df["NAME_INCOME_TYPE"] == "Pensioner"].iloc[0]
    worker = out[raw_like_df["NAME_INCOME_TYPE"] == "Working"].iloc[0]
    assert bool(pensioner["is_not_employed"]) is True
    assert np.isnan(pensioner["years_employed"])
    assert worker["years_employed"] > 0
    assert "DAYS_BIRTH" not in out.columns and "DAYS_EMPLOYED" not in out.columns
    assert raw_like_df.equals(before)


def test_model_features_exclude_protected():
    for col in ("CODE_GENDER", "TARGET", "client_id"):
        assert col not in MODEL_FEATURES


def test_pipeline_scores_unseen_category(raw_like_df):
    params = {**load_params()["model"], "n_estimators": 10}
    params.pop("target_approval_rate")
    pipe = build_pipeline(params, seed=42)
    pipe.fit(raw_like_df, raw_like_df["TARGET"])
    new = raw_like_df.head(20).copy()
    new["NAME_INCOME_TYPE"] = "Unemployed"
    proba = pipe.predict_proba(new)[:, 1]
    assert proba.shape == (20,)
    assert not np.isnan(proba).any()


def test_fit_threshold_hits_rate():
    proba = np.random.default_rng(0).uniform(size=5000)
    thr = fit_threshold(proba, 0.80)
    assert abs((proba < thr).mean() - 0.80) < 0.01
