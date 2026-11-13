import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_guard.api.main import create_app
from credit_guard.config import ID_COLUMN, INPUT_COLUMNS, load_params
from credit_guard.model.features import build_pipeline
from tests.conftest import make_raw_like


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = make_raw_like(n=400, seed=31)
    params = {**load_params()["model"], "n_estimators": 10}
    params.pop("target_approval_rate")
    model = build_pipeline(params, seed=42).fit(df, df["TARGET"])
    joblib.dump(model, d / "model.joblib")
    (d / "model_meta.json").write_text(
        json.dumps({"threshold": 0.15, "model_version": "test-v1"}), encoding="utf-8"
    )
    return d


@pytest.fixture
def client(models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    return TestClient(create_app(models_dir))


def _records(n=2):
    df = make_raw_like(n=n, seed=32)[[ID_COLUMN, *INPUT_COLUMNS]]
    df = df.astype(object).where(df.notna(), None)
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}
        for r in df.to_dict("records")
    ]


def test_predict_valid_returns_200(client, tmp_path):
    resp = client.post("/predict", json={"records": _records(2), "batch_id": "m01"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_version"] == "test-v1"
    assert len(body["predictions"]) == 2
    assert {p["decision"] for p in body["predictions"]} <= {"approve", "reject"}
    log_lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2 and json.loads(log_lines[0])["batch_id"] == "m01"


def test_predict_contract_violation_returns_422(client):
    records = _records(2)
    records[0]["AMT_INCOME_TOTAL"] = None
    resp = client.post("/predict", json={"records": records})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "contract_violation"
    assert "required_not_null" in detail["violations_by_check"]


def test_predict_wrong_type_returns_422(client):
    records = _records(1)
    records[0]["AMT_INCOME_TOTAL"] = "abc"
    assert client.post("/predict", json={"records": records}).status_code == 422


def test_predict_missing_field_returns_422(client):
    records = _records(1)
    del records[0]["AMT_CREDIT"]
    assert client.post("/predict", json={"records": records}).status_code == 422


def test_predict_rejects_protected_field(client):
    records = _records(1)
    records[0]["CODE_GENDER"] = "F"
    assert client.post("/predict", json={"records": records}).status_code == 422


def test_predict_empty_records_returns_422(client):
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_metrics_exposes_counters(client):
    client.post("/predict", json={"records": _records(1)})
    bad = _records(1)
    bad[0]["AMT_INCOME_TOTAL"] = None
    client.post("/predict", json={"records": bad})
    text = client.get("/metrics").text
    assert "credit_contract_rejections_total" in text
    assert "credit_predictions_total" in text
    assert "credit_request_latency_seconds" in text


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "test-v1"}


def test_health_degraded_without_model(tmp_path):
    resp = TestClient(create_app(tmp_path)).get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
