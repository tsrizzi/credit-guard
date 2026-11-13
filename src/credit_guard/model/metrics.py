"""Metricas de modelo de credito: AUC, Gini, KS, PR-AUC e impacto de negocio no threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ks_statistic(y_true, proba) -> float:
    """Distancia maxima entre as CDFs de score de bons (0) e maus (1)."""
    y, p = np.asarray(y_true), np.asarray(proba)
    grid = np.sort(np.unique(p))
    cdf_bad = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / max((y == 1).sum(), 1)
    cdf_good = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / max((y == 0).sum(), 1)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def evaluate_scores(y_true, proba, threshold: float) -> dict:
    y, p = np.asarray(y_true), np.asarray(proba)
    approved = p < threshold
    auc = float(roc_auc_score(y, p))
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y[approved].mean()) if approved.any() else 0.0,
        "default_rate": float(y.mean()),
    }
