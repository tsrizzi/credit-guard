"""Contrato de dados de entrada (Pandera).

O contrato valida o registro no formato de ENTRADA (colunas cruas do Home Credit
selecionadas), antes de qualquer engenharia de features. O mesmo schema e usado no
pipeline batch e na API. Os nomes das checagens sao estaveis: viram labels de metricas,
chaves de relatorio e ancoras de runbook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from credit_guard.config import (
    CATEGORICAL_INPUTS,
    EMPLOYMENT_SENTINEL,
    ID_COLUMN,
    NUMERIC_INPUTS,
    TARGET_COLUMN,
    load_params,
)

DOMAINS: dict[str, list] = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_INCOME_TYPE": [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "NAME_FAMILY_STATUS": [
        "Married",
        "Single / not married",
        "Civil marriage",
        "Separated",
        "Widow",
    ],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
}
PROTECTED_DOMAINS: dict[str, list] = {"CODE_GENDER": ["F", "M"]}

# Tipos de renda para os quais o sentinela de DAYS_EMPLOYED e uma codificacao legitima.
SENTINEL_INCOME_TYPES = ["Pensioner", "Unemployed"]

REQUIRED_NUMERIC = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
}

CHECK_NAMES = [
    "required_not_null",
    "coerce_dtype",
    "age_between_18_100",
    "income_positive_max",
    "amounts_positive_finite",
    "unique_client_id",
    *[f"domain_{c}" for c in [*DOMAINS, "REGION_RATING_CLIENT"]],
    "annuity_le_credit",
    "ext_source_unit_interval",
    "employment_sentinel_consistency",
    "days_employed_non_positive",
    "days_id_publish_non_positive",
    "children_range",
    "fam_members_min",
]

# colunas envolvidas nas regras entre colunas (aparecem no relatorio de violacoes)
FRAME_CHECK_COLUMNS = {
    "annuity_le_credit": ["AMT_ANNUITY", "AMT_CREDIT"],
    "employment_sentinel_consistency": ["DAYS_EMPLOYED", "NAME_INCOME_TYPE"],
}
AMOUNT_COLUMNS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


def _num(s: pd.Series) -> pd.Series:
    """Visao numerica tolerante: celula nao numerica vira NaN (quem a reporta e o coerce_dtype),
    para que uma celula invalida nao gere violacoes fantasmas em outras regras."""
    return pd.to_numeric(s, errors="coerce")


def _passes(mask: pd.Series, values: pd.Series) -> pd.Series:
    return mask | values.isna()


def _isin(values: list, name: str) -> pa.Check:
    return pa.Check(lambda s: s.isin(values), name=name)


def _rule(fn, name: str) -> pa.Check:
    return pa.Check(lambda s: _passes(fn(_num(s)), _num(s)), name=name)


def _numeric_checks(col: str, c: dict) -> list[pa.Check]:
    if col == "DAYS_BIRTH":
        lo, hi = c["age_min"], c["age_max"]
        return [_rule(lambda v: (-v / 365.25).between(lo, hi), "age_between_18_100")]
    if col == "AMT_INCOME_TOTAL":
        mx = c["income_max"]
        return [_rule(lambda v: (v > 0) & (v <= mx), "income_positive_max")]
    if col in AMOUNT_COLUMNS:
        mx = c["amount_max"]
        return [_rule(lambda v: (v > 0) & np.isfinite(v) & (v <= mx), "amounts_positive_finite")]
    if col.startswith("EXT_SOURCE_"):
        return [_rule(lambda v: v.between(0, 1), "ext_source_unit_interval")]
    if col == "DAYS_EMPLOYED":
        return [
            _rule(lambda v: (v <= 0) | (v == EMPLOYMENT_SENTINEL), "days_employed_non_positive")
        ]
    if col == "DAYS_ID_PUBLISH":
        return [_rule(lambda v: v <= 0, "days_id_publish_non_positive")]
    if col == "CNT_CHILDREN":
        mx = c["children_max"]
        return [_rule(lambda v: v.between(0, mx), "children_range")]
    if col == "CNT_FAM_MEMBERS":
        return [_rule(lambda v: v >= 1, "fam_members_min")]
    if col == "REGION_RATING_CLIENT":
        return [_rule(lambda v: v.isin([1, 2, 3]), "domain_REGION_RATING_CLIENT")]
    return []


def _annuity_le_credit(df):
    annuity, credit = _num(df["AMT_ANNUITY"]), _num(df["AMT_CREDIT"])
    return (annuity <= credit) | annuity.isna() | credit.isna()


def _sentinel_consistency(df):
    days = _num(df["DAYS_EMPLOYED"])
    return (days != EMPLOYMENT_SENTINEL) | df["NAME_INCOME_TYPE"].isin(SENTINEL_INCOME_TYPES)


def build_input_schema(
    contract_params: dict | None = None, include_protected: bool = False
) -> pa.DataFrameSchema:
    c = contract_params or load_params()["contract"]
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(object, nullable=False, unique=True),
        TARGET_COLUMN: pa.Column(
            int,
            nullable=False,
            required=False,
            coerce=True,
            checks=[_isin([0, 1], "domain_TARGET")],
        ),
    }
    for col in NUMERIC_INPUTS:
        columns[col] = pa.Column(
            float,
            nullable=col not in REQUIRED_NUMERIC,
            coerce=True,
            checks=_numeric_checks(col, c),
        )
    for col in CATEGORICAL_INPUTS:
        columns[col] = pa.Column(
            object, nullable=False, checks=[_isin(DOMAINS[col], f"domain_{col}")]
        )
    if include_protected:
        for col, values in PROTECTED_DOMAINS.items():
            columns[col] = pa.Column(
                object, nullable=False, checks=[_isin(values, f"domain_{col}")]
            )
    return pa.DataFrameSchema(
        columns,
        checks=[
            pa.Check(_annuity_le_credit, name="annuity_le_credit"),
            pa.Check(
                _sentinel_consistency,
                name="employment_sentinel_consistency",
            ),
        ],
        strict=False,
        name="credit_input_contract",
    )
