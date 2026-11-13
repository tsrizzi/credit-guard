"""Lote sujo para demonstrar o contrato bloqueando a ingestao.

Mistura sujeira REAL do Home Credit (linhas da quarentena que violam regras nao protegidas,
ex.: renda de 117 milhoes, parcela nula) com defeitos injetados, um tipo por grupo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_guard.config import EMPLOYMENT_SENTINEL, ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
from credit_guard.contracts.validation import validate_batch

# defeito injetado -> regra do contrato que deve barra-lo
DEFECT_CHECKS = {
    "null_income": "required_not_null",
    "underage": "age_between_18_100",
    "annuity_gt_credit": "annuity_le_credit",
    "working_with_sentinel": "employment_sentinel_consistency",
    "unknown_contract_type": "domain_NAME_CONTRACT_TYPE",
    "ext_source_out_of_range": "ext_source_unit_interval",
    "duplicated_id": "unique_client_id",
}
BATCH_COLUMNS = [ID_COLUMN, *INPUT_COLUMNS, TARGET_COLUMN]


def _inject(group: pd.DataFrame, defect: str, donor_ids: pd.Series) -> pd.DataFrame:
    g = group.copy()
    if defect == "null_income":
        g["AMT_INCOME_TOTAL"] = np.nan
    elif defect == "underage":
        g["DAYS_BIRTH"] = -16 * 365
    elif defect == "annuity_gt_credit":
        g["AMT_ANNUITY"] = g["AMT_CREDIT"] * 1.5
    elif defect == "working_with_sentinel":
        g["NAME_INCOME_TYPE"] = "Working"
        g["DAYS_EMPLOYED"] = EMPLOYMENT_SENTINEL
    elif defect == "unknown_contract_type":
        g["NAME_CONTRACT_TYPE"] = "Leasing"
    elif defect == "ext_source_out_of_range":
        g["EXT_SOURCE_3"] = 1.7
    elif defect == "duplicated_id":
        g[ID_COLUMN] = donor_ids.iloc[: len(g)].to_numpy()
    return g


def make_dirty_batch(
    clean: pd.DataFrame, quarantined: pd.DataFrame, n_rows: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    q = quarantined[[c for c in BATCH_COLUMNS if c in quarantined.columns]]
    q_result = validate_batch(q)  # sem atributos protegidos: so sujeira que a API tambem barra
    real = q.loc[q.index.isin(q_result.invalid_index)]

    n_injected = n_rows - len(real)
    sample = clean[BATCH_COLUMNS].sample(n=n_injected, random_state=int(rng.integers(1 << 31)))
    groups = [sample.iloc[i] for i in np.array_split(np.arange(len(sample)), len(DEFECT_CHECKS))]
    donors = sample[ID_COLUMN]
    parts, manifest = [real], {"real_quarantined": len(real)}
    for defect, group in zip(DEFECT_CHECKS, groups, strict=True):
        parts.append(_inject(group, defect, donors))
        manifest[defect] = len(group)
    batch = pd.concat(parts, ignore_index=True)
    return batch.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(
        drop=True
    ), manifest
