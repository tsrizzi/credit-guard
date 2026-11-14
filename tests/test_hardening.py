"""Testes da rodada de correcoes da revisao final (cada um reproduz um achado)."""

import hashlib
import hmac
import json

import numpy as np
import pandas as pd
import pytest

from credit_guard.contracts.validation import validate_batch
from tests.conftest import make_raw_like


# I1 - casos de violacao de regras entre colunas mostram as colunas envolvidas
def test_frame_level_failure_case_shows_relevant_columns(raw_like_df):
    df = raw_like_df.copy()
    df.loc[0, "AMT_ANNUITY"] = df.loc[0, "AMT_CREDIT"] + 1
    v = validate_batch(df).violations
    row = v[v["check"] == "annuity_le_credit"].iloc[0]
    assert row["column"] == "AMT_ANNUITY,AMT_CREDIT"
    assert str(round(df.loc[0, "AMT_ANNUITY"], 1)) in str(row["failure_case"])


# I1 - a amostra de casos do relatorio cobre todas as regras violadas
def test_failure_cases_sampled_per_rule(raw_like_df):
    df = pd.concat([raw_like_df] * 3, ignore_index=True)
    df["client_id"] = [f"{i:016x}" for i in range(len(df))]
    working = df.index[df["NAME_INCOME_TYPE"] == "Working"]
    df.loc[working, "DAYS_EMPLOYED"] = 365243  # > 50 violacoes de uma regra so
    df.loc[0, "DAYS_BIRTH"] = -16 * 365
    cases = validate_batch(df).to_dict()["failure_cases"]
    assert {"employment_sentinel_consistency", "age_between_18_100"} <= {c["check"] for c in cases}


# I2 - uma celula numerica invalida nao condena o lote inteiro nem cria regra fantasma
def test_single_bad_numeric_cell_flags_only_that_row(raw_like_df):
    df = raw_like_df.astype({"AMT_CREDIT": object})
    df.loc[5, "AMT_CREDIT"] = "abc"
    result = validate_batch(df)
    assert result.passed is False
    assert result.n_invalid_rows == 1
    assert set(result.violations["check"]) == {"coerce_dtype"}


# I3 - valores de emprestimo negativos, infinitos ou absurdos sao barrados
@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("AMT_CREDIT", -5e5, "amounts_positive_finite"),
        ("AMT_ANNUITY", -1.0, "amounts_positive_finite"),
        ("AMT_GOODS_PRICE", np.inf, "amounts_positive_finite"),
        ("AMT_CREDIT", 1e308, "amounts_positive_finite"),
        ("DAYS_ID_PUBLISH", 100, "days_id_publish_non_positive"),
    ],
)
def test_amounts_and_days_rules(raw_like_df, column, value, check):
    df = raw_like_df.copy()
    df.loc[0, column] = value
    result = validate_batch(df)
    assert result.passed is False
    assert check in set(result.violations["check"])


# I4 - o relatorio do Evidently usa PSI (mesma metrica dos testes proprios)
def test_evidently_report_uses_psi(tmp_path):
    from credit_guard.monitoring.evidently_reports import build_drift_report

    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "c": rng.choice(["A", "B"], 2000)})
    cur = pd.DataFrame({"x": rng.normal(1.5, 1, 2000), "c": rng.choice(["A", "B"], 2000)})
    summary = build_drift_report(ref, cur, ["x"], ["c"], tmp_path / "r.html")
    assert all(v["method"].lower() == "psi" for v in summary["per_feature"].values())
    assert summary["per_feature"]["x"]["threshold"] == pytest.approx(0.2)
    assert summary["drifted_features"] == ["x"]


# I6 - MLflow indisponivel nao derruba o pipeline
def test_mlflow_best_effort_swallows_failures(capsys):
    from mlflow.exceptions import MlflowException

    from credit_guard.observability.tracking import mlflow_best_effort

    def boom():
        raise MlflowException("server down")

    assert mlflow_best_effort(boom, "log_month") is False
    events = [json.loads(ln) for ln in capsys.readouterr().err.splitlines() if ln.startswith("{")]
    assert any(e["event"] == "mlflow_unavailable" and e["action"] == "log_month" for e in events)
    assert mlflow_best_effort(lambda: None, "ok") is True


# I9 - pseudonimizacao com HMAC-SHA256 (chave secreta) e aviso quando a chave e a publica de dev
def test_pseudonymize_uses_hmac_with_secret_key():
    from credit_guard.data.prepare import pseudonymize_ids

    out = pseudonymize_ids(pd.Series([100002]), "chave-secreta")
    expected = hmac.new(b"chave-secreta", b"100002", hashlib.sha256).hexdigest()[:16]
    assert out.iloc[0] == expected


def test_public_dev_key_emits_warning(monkeypatch, capsys):
    from credit_guard.data.prepare import resolve_id_key

    monkeypatch.delenv("ID_SALT", raising=False)
    resolve_id_key()
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    assert any(e["event"] == "public_dev_pseudonymization_key" for e in events)


# M1 - configuracao invalida do Datadog nao derruba a API
def test_statsd_bad_config_falls_back_to_none(monkeypatch):
    from credit_guard.observability import publish

    monkeypatch.setattr(publish, "_statsd", None)
    monkeypatch.setenv("DD_ENABLED", "true")
    monkeypatch.setenv("DD_DOGSTATSD_PORT", "abc")
    assert publish.get_statsd() is None


# M3 - intervencao conjunta em grupo misto preserva combinacoes coerentes da referencia
def test_mixed_group_restored_jointly_from_reference_rows():
    from credit_guard.monitoring.causal import _restore_group

    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {"kind": ["Pensioner"] * 500 + ["Working"] * 500, "days": [365243] * 500 + [-1000] * 500}
    )
    cur = pd.DataFrame({"kind": ["Working"] * 1000, "days": [-500] * 1000})
    restored = _restore_group(ref, cur, ["kind", "days"], rng)
    combos = set(map(tuple, restored[["kind", "days"]].to_numpy().tolist()))
    assert combos <= {("Pensioner", 365243), ("Working", -1000)}


def test_make_raw_like_ids_are_pseudonymized_format():
    ids = make_raw_like(n=5)["client_id"]
    assert ids.str.fullmatch(r"[0-9a-f]{16}").all()
