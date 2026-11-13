"""Contrato de dados de entrada (Pandera).

O contrato valida o registro no formato de ENTRADA (colunas cruas do Home Credit
selecionadas), antes de qualquer engenharia de features. O mesmo schema e usado no
pipeline batch e na API. Os nomes das checagens sao estaveis: viram labels de metricas,
chaves de relatorio e ancoras de runbook.
"""

from __future__ import annotations

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
    "age_between_18_100",
    "income_positive_max",
    "unique_client_id",
    *[f"domain_{c}" for c in [*DOMAINS, "REGION_RATING_CLIENT"]],
    "annuity_le_credit",
    "ext_source_unit_interval",
    "employment_sentinel_consistency",
    "days_employed_non_positive",
    "children_range",
    "fam_members_min",
]


def _isin(values: list, name: str) -> pa.Check:
    return pa.Check(lambda s: s.isin(values), name=name)


def _numeric_checks(col: str, c: dict) -> list[pa.Check]:
    if col == "DAYS_BIRTH":
        lo, hi = c["age_min"], c["age_max"]
        return [
            pa.Check(
                lambda s: (-s / 365.25).between(lo, hi),
                name="age_between_18_100",
            )
        ]
    if col == "AMT_INCOME_TOTAL":
        mx = c["income_max"]
        return [
            pa.Check(
                lambda s: (s > 0) & (s <= mx),
                name="income_positive_max",
            )
        ]
    if col.startswith("EXT_SOURCE_"):
        return [pa.Check(lambda s: s.between(0, 1), name="ext_source_unit_interval")]
    if col == "DAYS_EMPLOYED":
        return [
            pa.Check(
                lambda s: (s <= 0) | (s == EMPLOYMENT_SENTINEL),
                name="days_employed_non_positive",
            )
        ]
    if col == "CNT_CHILDREN":
        mx = c["children_max"]
        return [pa.Check(lambda s: s.between(0, mx), name="children_range")]
    if col == "CNT_FAM_MEMBERS":
        return [pa.Check(lambda s: s >= 1, name="fam_members_min")]
    if col == "REGION_RATING_CLIENT":
        return [_isin([1, 2, 3], "domain_REGION_RATING_CLIENT")]
    return []


def _annuity_le_credit(df):
    return (
        (df["AMT_ANNUITY"] <= df["AMT_CREDIT"]) | df["AMT_ANNUITY"].isna() | df["AMT_CREDIT"].isna()
    )


def _sentinel_consistency(df):
    return (df["DAYS_EMPLOYED"] != EMPLOYMENT_SENTINEL) | df["NAME_INCOME_TYPE"].isin(
        SENTINEL_INCOME_TYPES
    )


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
