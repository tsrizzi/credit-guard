"""Estagio `prepare`: bruto -> quarentena / referencia (treino, teste) / holdout / protegidos.

- Minimizacao: so as colunas necessarias sao lidas do arquivo bruto.
- Pseudonimizacao: SK_ID_CURR vira `client_id` (sha256 com salt, 16 hex).
- Contrato aplicado ao bruto: linhas que violam vao para quarentena (com relatorio).
- Atributo protegido (CODE_GENDER) sai das tabelas de features e fica em tabela separada.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_guard.config import (
    DATA_DIR,
    ID_COLUMN,
    INPUT_COLUMNS,
    PROTECTED_COLUMNS,
    RAW_ID_COLUMN,
    REPORTS_DIR,
    TARGET_COLUMN,
    load_params,
)
from credit_guard.contracts.report import write_validation_report
from credit_guard.contracts.validation import ValidationResult, validate_batch
from credit_guard.logging_utils import get_logger
from credit_guard.observability.pipeline import StageContext, track_stage

log = get_logger("prepare")

DEFAULT_SALT = "fiap-fase4-dev-salt"  # apenas desenvolvimento; em producao vem de um cofre
PROTECTED_TABLE_COLUMNS = [ID_COLUMN, *PROTECTED_COLUMNS, "DAYS_BIRTH"]


def pseudonymize_ids(ids: pd.Series, key: str) -> pd.Series:
    """HMAC-SHA256 com chave secreta: sem a chave, nao da para recalcular o pseudonimo a partir
    de IDs candidatos (um hash com salt publico seria reversivel por forca bruta)."""
    k = key.encode()
    return ids.astype(str).map(lambda v: hmac.new(k, v.encode(), hashlib.sha256).hexdigest()[:16])


def resolve_id_key() -> str:
    key = os.environ.get("ID_SALT")
    if not key:
        log.warning(
            "public_dev_pseudonymization_key",
            extra={
                "detail": "ID_SALT ausente: usando a chave PUBLICA de desenvolvimento; os pseudonimos sao apenas ilustrativos"
            },
        )
        return DEFAULT_SALT
    return key


def load_raw(path: Path, salt: str | None = None) -> pd.DataFrame:
    cols = [RAW_ID_COLUMN, TARGET_COLUMN, *PROTECTED_COLUMNS, *INPUT_COLUMNS]
    df = pd.read_csv(path, usecols=cols)
    df.insert(0, ID_COLUMN, pseudonymize_ids(df.pop(RAW_ID_COLUMN), salt or resolve_id_key()))
    return df


def quarantine_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ValidationResult]:
    result = validate_batch(df, include_protected=True)
    bad = df.index.isin(result.invalid_index)
    return df.loc[~bad].copy(), df.loc[bad].copy(), result


def split_frames(clean: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    p, seed = params["prepare"], params["seed"]
    holdout_frac = 1 - p["ref_train_frac"] - p["ref_test_frac"]
    ref, holdout = train_test_split(
        clean, test_size=holdout_frac, stratify=clean[TARGET_COLUMN], random_state=seed
    )
    test_share = p["ref_test_frac"] / (p["ref_train_frac"] + p["ref_test_frac"])
    train, test = train_test_split(
        ref, test_size=test_share, stratify=ref[TARGET_COLUMN], random_state=seed
    )
    return {"train": train, "test": test, "holdout": holdout}


def separate_protected(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.drop(columns=PROTECTED_COLUMNS), df[PROTECTED_TABLE_COLUMNS].copy()


def _run(ctx: StageContext) -> None:
    params = load_params()
    raw = load_raw(DATA_DIR / "raw" / "application_train.csv")
    clean, quarantined, result = quarantine_invalid(raw)
    write_validation_report(result, REPORTS_DIR / "validation", "reference_raw")
    ctx.add_rows("accepted", len(clean))
    ctx.add_rows("quarantined", len(quarantined))

    for sub in ("reference", "holdout", "protected", "quarantine"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    quarantined.to_parquet(DATA_DIR / "quarantine" / "raw_quarantined.parquet", index=False)

    parts = split_frames(clean, params)
    feats, prot = {}, {}
    for name, frame in parts.items():
        feats[name], prot[name] = separate_protected(frame)
    feats["train"].to_parquet(DATA_DIR / "reference" / "train.parquet", index=False)
    feats["test"].to_parquet(DATA_DIR / "reference" / "test.parquet", index=False)
    feats["holdout"].to_parquet(DATA_DIR / "holdout" / "holdout.parquet", index=False)
    pd.concat([prot["train"], prot["test"]]).to_parquet(
        DATA_DIR / "protected" / "reference.parquet", index=False
    )
    prot["holdout"].to_parquet(DATA_DIR / "protected" / "holdout.parquet", index=False)

    log.info(
        "prepare_done",
        extra={
            "raw_rows": len(raw),
            "quarantined_rows": len(quarantined),
            "violations_by_check": result.violations_by_check,
            **{f"{k}_rows": len(v) for k, v in parts.items()},
        },
    )


def main() -> None:
    with track_stage("prepare") as ctx:
        _run(ctx)


if __name__ == "__main__":
    main()
