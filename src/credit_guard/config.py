"""Caminhos, listas de colunas e parametros versionados do projeto."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
PARAMS_PATH = ROOT / "params.yaml"

ID_COLUMN = "client_id"
RAW_ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
PROTECTED_COLUMNS = ["CODE_GENDER"]

NUMERIC_INPUTS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
CATEGORICAL_INPUTS = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]
INPUT_COLUMNS = NUMERIC_INPUTS + CATEGORICAL_INPUTS

# Codificacao do Home Credit para "nao empregado" (aposentados/desempregados).
EMPLOYMENT_SENTINEL = 365243


def load_params(path: Path | None = None) -> dict:
    with open(path or PARAMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
