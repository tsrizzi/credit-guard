from credit_guard.config import INPUT_COLUMNS, load_params


def test_load_params_has_seed():
    params = load_params()
    assert params["seed"] == 42
    assert params["monitoring"]["auc_drop_alert"] == 0.03


def test_input_columns_exclude_protected_and_target():
    assert "CODE_GENDER" not in INPUT_COLUMNS
    assert "TARGET" not in INPUT_COLUMNS
    assert len(INPUT_COLUMNS) == 19
