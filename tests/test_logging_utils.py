import json

from credit_guard.logging_utils import get_logger


def test_json_logger_emits_json(capsys):
    get_logger("t").info("stage_done", extra={"stage": "prepare", "rows": 3})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "stage_done"
    assert payload["stage"] == "prepare"
    assert payload["rows"] == 3
    assert payload["level"] == "INFO"
