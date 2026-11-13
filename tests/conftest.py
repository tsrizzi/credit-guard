"""Fixtures sinteticas no formato de entrada do Home Credit (nenhum dado real nos testes)."""

import numpy as np
import pandas as pd
import pytest

from credit_guard.config import EMPLOYMENT_SENTINEL


def make_raw_like(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income_types = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant"],
        size=n,
        p=[0.5, 0.2, 0.2, 0.1],
    )
    credit = rng.uniform(100_000, 1_500_000, n).round(1)
    annuity = (credit / rng.uniform(10, 40, n)).round(1)
    days_employed = -rng.integers(30, 10_000, n)
    days_employed = np.where(income_types == "Pensioner", EMPLOYMENT_SENTINEL, days_employed)
    ext2 = rng.uniform(0, 1, n)
    df = pd.DataFrame(
        {
            "client_id": [f"id{i:05d}" for i in range(n)],
            "AMT_INCOME_TOTAL": rng.uniform(40_000, 400_000, n).round(1),
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(1),
            "DAYS_BIRTH": -rng.integers(21 * 365, 68 * 365, n),
            "DAYS_EMPLOYED": days_employed,
            "DAYS_ID_PUBLISH": -rng.integers(0, 6000, n),
            "CNT_CHILDREN": rng.integers(0, 4, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n),
            "EXT_SOURCE_1": np.where(rng.uniform(size=n) < 0.5, np.nan, rng.uniform(0, 1, n)),
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.uniform(0, 1, n)),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "NAME_INCOME_TYPE": income_types,
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education"], n, p=[0.7, 0.3]
            ),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Widow"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.65, 0.35]),
        }
    )
    # alvo correlacionado com EXT_SOURCE_2 para os modelos terem sinal
    p_default = 0.25 - 0.2 * ext2
    df["TARGET"] = (rng.uniform(size=n) < p_default).astype(int)
    return df


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    return make_raw_like()
