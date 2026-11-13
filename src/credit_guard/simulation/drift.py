"""Simulacao de degradacao economica gradual ao longo dos meses de producao.

Data drift (covariate shift), com historia economica:
- inflacao: valores de credito, parcela e bem financiado sobem `r*m` (precos maiores);
- erosao da renda real: renda cai `r*m`;
- desemprego: uma fracao crescente de `Working` passa a `Unemployed` (com o sentinela);
- piora de bureau: EXT_SOURCE_2/3 caem `r*m`.

Concept drift: a partir de `concept_start_month`, P(inadimplencia | X) sobe no segmento
`concept_segment_income_type` sem nenhuma mudanca nas features. So aparece quando os
rotulos chegam, pela queda de performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_guard.config import EMPLOYMENT_SENTINEL, TARGET_COLUMN

PRICE_COLUMNS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
BUREAU_COLUMNS = ["EXT_SOURCE_2", "EXT_SOURCE_3"]


def apply_data_drift(
    df: pd.DataFrame, month: int, sim: dict, rng: np.random.Generator
) -> pd.DataFrame:
    out = df.copy()
    out[PRICE_COLUMNS] = out[PRICE_COLUMNS] * (1 + sim["price_inflation_per_month"] * month)
    out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"] * (
        1 - sim["real_income_erosion_per_month"] * month
    )

    working = out.index[out["NAME_INCOME_TYPE"] == "Working"]
    n_shift = round(sim["unemployment_shift_per_month"] * month * len(working))
    if n_shift:
        chosen = rng.choice(working, size=n_shift, replace=False)
        out.loc[chosen, "NAME_INCOME_TYPE"] = "Unemployed"
        out.loc[chosen, "DAYS_EMPLOYED"] = EMPLOYMENT_SENTINEL

    decline = sim["ext_source_decline_per_month"] * month
    out[BUREAU_COLUMNS] = (out[BUREAU_COLUMNS] - decline).clip(0, 1)
    return out


def concept_flip_rate(month: int, sim: dict) -> float:
    start = sim["concept_start_month"]
    return 0.0 if month < start else sim["concept_flip_rate_per_month"] * (month - start + 1)


def apply_concept_drift(
    df: pd.DataFrame, month: int, sim: dict, rng: np.random.Generator
) -> pd.DataFrame:
    out = df.copy()
    rate = concept_flip_rate(month, sim)
    if rate <= 0:
        return out
    segment = out["NAME_INCOME_TYPE"] == sim["concept_segment_income_type"]
    flip = segment & (out[TARGET_COLUMN] == 0) & (rng.uniform(size=len(out)) < rate)
    out.loc[flip, TARGET_COLUMN] = 1
    return out
