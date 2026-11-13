import json

import pytest

from credit_guard.observability.pipeline import track_stage
from credit_guard.observability.publish import get_statsd, push_metrics


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().err.strip().splitlines() if line]


def test_push_unreachable_gateway_does_not_raise():
    assert push_metrics("j", {"m": (1.0, {})}, gateway="http://127.0.0.1:1") is False


def test_push_without_gateway_returns_false(monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    assert push_metrics("j", {"m": (1.0, {})}) is False


def test_statsd_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DD_ENABLED", raising=False)
    assert get_statsd() is None


def test_statsd_enabled_returns_client(monkeypatch):
    monkeypatch.setenv("DD_ENABLED", "true")
    assert get_statsd() is not None


def test_track_stage_logs_and_reraises(capsys, monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    with pytest.raises(ValueError):
        with track_stage("demo") as ctx:
            ctx.add_rows("accepted", 5)
            raise ValueError("boom")
    failed = [e for e in _events(capsys) if e["event"] == "stage_failed"]
    assert failed and failed[0]["stage"] == "demo"
    assert failed[0]["rows"] == {"accepted": 5}
    assert "boom" in failed[0]["error"]


def test_track_stage_success_logs(capsys, monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    with track_stage("demo") as ctx:
        ctx.add_rows("accepted", 2)
    events = _events(capsys)
    assert [e["event"] for e in events] == ["stage_started", "stage_succeeded"]
    assert events[1]["duration_seconds"] >= 0
