"""Engenharia de features e pipeline do modelo (o mesmo codigo no treino, no batch e na API)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from credit_guard.config import CATEGORICAL_INPUTS, EMPLOYMENT_SENTINEL, NUMERIC_INPUTS

ENGINEERED_NUMERIC = [
    "age_years",
    "years_employed",
    "is_not_employed",
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_term",
    "goods_credit_ratio",
]
NUMERIC_FEATURES = [c for c in NUMERIC_INPUTS if c not in ("DAYS_BIRTH", "DAYS_EMPLOYED")] + (
    ENGINEERED_NUMERIC
)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_INPUTS


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    not_employed = out["DAYS_EMPLOYED"] == EMPLOYMENT_SENTINEL
    out["age_years"] = -out["DAYS_BIRTH"] / 365.25
    out["is_not_employed"] = not_employed.astype(float)
    out["years_employed"] = (-out["DAYS_EMPLOYED"] / 365.25).where(~not_employed)
    out["credit_income_ratio"] = _safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_income_ratio"] = _safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["credit_term"] = _safe_div(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    out["goods_credit_ratio"] = _safe_div(out["AMT_GOODS_PRICE"], out["AMT_CREDIT"])
    return out.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def build_pipeline(model_params: dict, seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_INPUTS,
            ),
        ]
    )
    clf = XGBClassifier(eval_metric="auc", n_jobs=-1, random_state=seed, **model_params)
    return Pipeline(
        [("features", FunctionTransformer(engineer_features)), ("pre", pre), ("clf", clf)]
    )


def base_feature_importance(pipe: Pipeline) -> pd.Series:
    """Importancia agregada por feature de entrada (soma das colunas one-hot)."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    base = []
    for n in names:
        kind, col = n.split("__", 1)
        if kind == "cat":
            col = next(c for c in CATEGORICAL_INPUTS if col.startswith(c))
        base.append(col)
    return pd.Series(imp, index=base).groupby(level=0).sum().sort_values(ascending=False)
