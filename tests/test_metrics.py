import numpy as np
import pytest

from credit_guard.model.metrics import evaluate_scores, ks_statistic


def test_ks_statistic_perfect_and_random():
    y = np.array([0] * 500 + [1] * 500)
    perfect = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert ks_statistic(y, rng.uniform(size=1000)) < 0.1


def test_evaluate_scores_keys_and_approval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    proba = np.clip(y * 0.3 + rng.uniform(size=1000) * 0.7, 0, 1)
    m = evaluate_scores(y, proba, threshold=0.5)
    assert set(m) >= {"roc_auc", "gini", "ks", "pr_auc", "approval_rate", "default_rate_approved"}
    assert m["approval_rate"] == pytest.approx((proba < 0.5).mean())
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
