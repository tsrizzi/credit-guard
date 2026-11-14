"""Execucao do contrato com coleta de todas as violacoes (lazy) e resultado estruturado."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera.errors as pa_errors

from credit_guard.contracts.schema import FRAME_CHECK_COLUMNS, build_input_schema

VIOLATION_COLUMNS = ["column", "check", "failure_case", "row_index"]


@dataclass
class ValidationResult:
    passed: bool
    n_rows: int
    n_invalid_rows: int
    violations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=VIOLATION_COLUMNS)
    )
    invalid_index: list = field(default_factory=list)
    valid: pd.DataFrame | None = None

    @property
    def violations_by_check(self) -> dict[str, int]:
        if self.violations.empty:
            return {}
        return {str(k): int(v) for k, v in self.violations["check"].value_counts().items()}

    def to_dict(self, max_cases: int = 50, per_check: int = 5) -> dict:
        # amostra estratificada por regra: todas as regras violadas aparecem no relatorio
        cases = self.violations.groupby("check", sort=False).head(per_check).head(max_cases)
        cases = cases.astype(object)
        cases = cases.where(pd.notna(cases), None)
        return {
            "passed": self.passed,
            "verdict": "APROVADO" if self.passed else "BLOQUEADO",
            "n_rows": self.n_rows,
            "n_invalid_rows": self.n_invalid_rows,
            "violations_by_check": self.violations_by_check,
            "failure_cases": [
                {k: _jsonable(v) for k, v in r.items()} for r in cases.to_dict("records")
            ],
        }


class ContractViolation(Exception):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(f"contrato violado: {result.violations_by_check}")


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value if isinstance(value, (int, float, str, bool, type(None))) else str(value)


def _normalize_check(check: str) -> str:
    check = str(check)
    if check == "not_nullable":
        return "required_not_null"
    if check.startswith("coerce_dtype") or check.startswith("dtype("):
        return "coerce_dtype"
    if check == "field_uniqueness":
        return "unique_client_id"
    if check == "column_in_dataframe":
        return "required_column"
    return check


def _frame_violations(rows: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for i, r in out.iterrows():
        cols = FRAME_CHECK_COLUMNS.get(r["check"])
        if not cols or r["row_index"] not in df.index:
            out.at[i, "column"] = None
            continue
        values = df.loc[r["row_index"], cols]
        out.at[i, "column"] = ",".join(cols)
        out.at[i, "failure_case"] = ", ".join(f"{c}={_jsonable(values[c])}" for c in cols)
    return out


def validate_batch(
    df: pd.DataFrame, include_protected: bool = False, contract_params: dict | None = None
) -> ValidationResult:
    n_rows = len(df)
    if n_rows == 0:
        violations = pd.DataFrame(
            [
                {
                    "column": None,
                    "check": "empty_batch",
                    "failure_case": "0 linhas",
                    "row_index": None,
                }
            ]
        )
        return ValidationResult(False, 0, 0, violations, [])

    schema = build_input_schema(contract_params, include_protected=include_protected)
    try:
        valid = schema.validate(df, lazy=True)
    except pa_errors.SchemaErrors as err:
        fc = err.failure_cases.copy()
        violations = pd.DataFrame(
            {
                "column": fc["column"],
                "check": fc["check"].map(_normalize_check),
                "failure_case": fc["failure_case"],
                "row_index": fc["index"],
            }
        )
        # checagens entre colunas sao reportadas uma vez por coluna: vira uma por linha, com as
        # colunas envolvidas e seus valores reais como caso de falha
        frame_level = (fc["schema_context"] == "DataFrameSchema") & fc["index"].notna()
        violations = pd.concat(
            [
                violations[~frame_level],
                _frame_violations(
                    violations[frame_level].drop_duplicates(["check", "row_index"]), df
                ),
            ],
            ignore_index=True,
        )
        violations = violations.drop_duplicates(subset=["check", "row_index", "column"])
        violations = violations.reset_index(drop=True)
        if violations["row_index"].isna().any():
            # violacao no nivel do lote (ex.: coluna ausente) invalida todas as linhas
            invalid = list(df.index)
        else:
            invalid = sorted(set(violations["row_index"].tolist()), key=str)
        return ValidationResult(False, n_rows, len(invalid), violations, invalid)
    return ValidationResult(True, n_rows, 0, valid=valid)


def enforce(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    result = validate_batch(df, **kwargs)
    if not result.passed:
        raise ContractViolation(result)
    return result.valid
