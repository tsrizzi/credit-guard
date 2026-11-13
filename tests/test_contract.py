import numpy as np
import pandas as pd
import pytest

from credit_guard.contracts import ingest
from credit_guard.contracts.report import write_validation_report
from credit_guard.contracts.validation import ContractViolation, enforce, validate_batch


def _first_with(df: pd.DataFrame, income_type: str) -> int:
    return df.index[df["NAME_INCOME_TYPE"] == income_type][0]


def test_valid_batch_passes(raw_like_df):
    result = validate_batch(raw_like_df, include_protected=True)
    assert result.passed is True
    assert result.n_invalid_rows == 0


def _null_income(df):
    df.loc[0, "AMT_INCOME_TOTAL"] = np.nan


def _underage(df):
    df.loc[0, "DAYS_BIRTH"] = -16 * 365


def _huge_income(df):
    df.loc[0, "AMT_INCOME_TOTAL"] = 117_000_000


def _duplicated_id(df):
    df.loc[1, "client_id"] = df.loc[0, "client_id"]


def _unknown_family(df):
    df.loc[0, "NAME_FAMILY_STATUS"] = "Unknown"


def _annuity_gt_credit(df):
    df.loc[0, "AMT_ANNUITY"] = df.loc[0, "AMT_CREDIT"] + 1


def _ext_source_out(df):
    df.loc[0, "EXT_SOURCE_2"] = 1.4


def _working_sentinel(df):
    df.loc[_first_with(df, "Working"), "DAYS_EMPLOYED"] = 365243


def _gender_xna(df):
    df.loc[0, "CODE_GENDER"] = "XNA"


@pytest.mark.parametrize(
    ("mutate", "expected_check"),
    [
        (_null_income, "required_not_null"),
        (_underage, "age_between_18_100"),
        (_huge_income, "income_positive_max"),
        (_duplicated_id, "unique_client_id"),
        (_unknown_family, "domain_NAME_FAMILY_STATUS"),
        (_annuity_gt_credit, "annuity_le_credit"),
        (_ext_source_out, "ext_source_unit_interval"),
        (_working_sentinel, "employment_sentinel_consistency"),
        (_gender_xna, "domain_CODE_GENDER"),
    ],
)
def test_each_rule_rejects(raw_like_df, mutate, expected_check):
    df = raw_like_df.copy()
    mutate(df)
    result = validate_batch(df, include_protected=True)
    assert result.passed is False
    assert expected_check in set(result.violations["check"])


def test_pensioner_sentinel_is_valid(raw_like_df):
    pensioners = raw_like_df[raw_like_df["NAME_INCOME_TYPE"] == "Pensioner"]
    assert (pensioners["DAYS_EMPLOYED"] == 365243).all()
    assert validate_batch(pensioners).passed is True


def test_wrong_dtype_rejected(raw_like_df):
    df = raw_like_df.astype({"AMT_CREDIT": object})
    df.loc[0, "AMT_CREDIT"] = "abc"
    result = validate_batch(df)
    assert result.passed is False


def test_empty_batch_rejected(raw_like_df):
    result = validate_batch(raw_like_df.iloc[0:0])
    assert result.passed is False
    assert "empty_batch" in set(result.violations["check"])


def test_all_rows_invalid_report(raw_like_df, tmp_path):
    df = raw_like_df.copy()
    df["AMT_INCOME_TOTAL"] = np.nan
    result = validate_batch(df)
    assert result.n_invalid_rows == len(df)
    json_path, html_path = write_validation_report(result, tmp_path, "todos_invalidos")
    assert json_path.exists() and html_path.exists()
    assert "BLOQUEADO" in html_path.read_text(encoding="utf-8")


def test_enforce_raises(raw_like_df):
    df = raw_like_df.copy()
    _null_income(df)
    with pytest.raises(ContractViolation) as exc:
        enforce(df)
    assert exc.value.result.passed is False


def test_enforce_returns_valid_frame(raw_like_df):
    out = enforce(raw_like_df)
    assert len(out) == len(raw_like_df)


def test_ingest_cli_blocks(raw_like_df, tmp_path):
    df = raw_like_df.copy()
    _null_income(df)
    path = tmp_path / "lote.parquet"
    df.to_parquet(path)
    args = ["--input", str(path), "--report-dir", str(tmp_path), "--name", "lote"]
    assert ingest.main(args) == 1
    assert ingest.main([*args, "--expect-reject"]) == 0


def test_ingest_cli_accepts_valid(raw_like_df, tmp_path):
    path = tmp_path / "ok.parquet"
    raw_like_df.to_parquet(path)
    args = ["--input", str(path), "--report-dir", str(tmp_path), "--name", "ok"]
    assert ingest.main(args) == 0
    assert ingest.main([*args, "--expect-reject"]) == 1


def test_dataframe_level_check_counts_one_violation_per_row(raw_like_df):
    df = raw_like_df.copy()
    df.loc[[0, 1, 2], "AMT_ANNUITY"] = df.loc[[0, 1, 2], "AMT_CREDIT"] + 1
    result = validate_batch(df)
    assert result.violations_by_check["annuity_le_credit"] == 3
    assert result.n_invalid_rows == 3
