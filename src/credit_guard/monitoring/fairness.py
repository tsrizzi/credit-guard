"""Auditoria de fairness offline (os atributos protegidos nunca passam pela API nem pelo modelo).

- approval_rate por grupo e disparate impact (regra dos 80%: min/max >= 0,8);
- tpr = taxa de aprovacao entre bons pagadores (TARGET == 0): igualdade de oportunidade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_guard.config import ID_COLUMN, TARGET_COLUMN

AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]
AGE_EDGES = [-np.inf, 25, 35, 45, 55, np.inf]
AUDITED = ["CODE_GENDER", "age_band"]


def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25
    return pd.cut(years, AGE_EDGES, labels=AGE_LABELS, right=False).astype(str)


def fairness_table(scored: pd.DataFrame, protected: pd.DataFrame) -> pd.DataFrame:
    df = scored.merge(protected, on=ID_COLUMN, how="inner")
    df["age_band"] = age_band(df["DAYS_BIRTH"])
    df["approved"] = df["decision"] == "approve"
    has_target = TARGET_COLUMN in df.columns
    rows = []
    for attr in AUDITED:
        for group, g in df.groupby(attr, observed=True):
            good = g[g[TARGET_COLUMN] == 0] if has_target else g.iloc[0:0]
            rows.append(
                {
                    "attribute": attr,
                    "group": str(group),
                    "n": len(g),
                    "approval_rate": float(g["approved"].mean()),
                    "default_rate": float(g[TARGET_COLUMN].mean()) if has_target else np.nan,
                    "tpr": float(good["approved"].mean()) if len(good) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def disparate_impact(table: pd.DataFrame, attribute: str) -> float:
    rates = table.loc[table["attribute"] == attribute, "approval_rate"]
    return float(rates.min() / rates.max()) if len(rates) and rates.max() > 0 else float("nan")
