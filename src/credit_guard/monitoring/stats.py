"""Testes estatisticos de drift implementados de forma transparente (alem do Evidently).

- PSI (Population Stability Index): metrica padrao da industria de credito.
  Faixas: < 0,10 estavel | 0,10-0,20 moderado (investigar) | >= 0,20 significativo (retreinar).
- KS (Kolmogorov-Smirnov, duas amostras): significancia estatistica da mudanca.
- Jensen-Shannon (distancia, base 2): simetrica e limitada em [0, 1].
- Qui-quadrado: categoricas.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

EPS = 1e-4
NAN_BUCKET = "__nan__"


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _distributions(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Proporcoes por bucket nas duas amostras (bins por quantis da referencia; NaN = bucket)."""
    if _is_numeric(reference) and _is_numeric(current):
        ref_valid = reference.dropna().to_numpy(dtype=float)
        edges = (
            np.unique(np.quantile(ref_valid, np.linspace(0, 1, bins + 1))) if len(ref_valid) else []
        )
        inner = np.asarray(edges[1:-1]) if len(edges) > 2 else np.array([])

        def counts(s: pd.Series) -> np.ndarray:
            vals = s.to_numpy(dtype=float)
            nan = np.isnan(vals)
            idx = np.searchsorted(inner, vals[~nan], side="right")
            c = np.bincount(idx, minlength=len(inner) + 1).astype(float)
            return np.append(c, nan.sum())

        ref_c, cur_c = counts(reference), counts(current)
    else:
        ref_s = reference.astype(object).where(reference.notna(), NAN_BUCKET)
        cur_s = current.astype(object).where(current.notna(), NAN_BUCKET)
        cats = sorted(set(ref_s.unique()) | set(cur_s.unique()), key=str)
        ref_c = ref_s.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
        cur_c = cur_s.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    ref_p = ref_c / max(ref_c.sum(), 1)
    cur_p = cur_c / max(cur_c.sum(), 1)
    return ref_p, cur_p


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref_p, cur_p = _distributions(reference, current, bins)
    ref_p, cur_p = np.maximum(ref_p, EPS), np.maximum(cur_p, EPS)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_band(value: float) -> Literal["stable", "moderate", "significant"]:
    if value < 0.10:
        return "stable"
    if value < 0.20:
        return "moderate"
    return "significant"


def ks_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    res = ks_2samp(reference.dropna().to_numpy(float), current.dropna().to_numpy(float))
    return float(res.statistic), float(res.pvalue)


def js_distance(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref_p, cur_p = _distributions(reference, current, bins)
    value = float(jensenshannon(ref_p, cur_p, base=2))
    return 0.0 if np.isnan(value) else value


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    ref_s, cur_s = (
        reference.astype(object).fillna(NAN_BUCKET),
        current.astype(object).fillna(NAN_BUCKET),
    )
    cats = sorted(set(ref_s.unique()) | set(cur_s.unique()), key=str)
    table = np.vstack(
        [
            ref_s.value_counts().reindex(cats, fill_value=0).to_numpy(),
            cur_s.value_counts().reindex(cats, fill_value=0).to_numpy(),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(table)[1])


def feature_drift_table(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    bins: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for col in numeric:
        value = psi(ref[col], cur[col], bins)
        ks_stat, ks_p = ks_test(ref[col], cur[col])
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "psi": value,
                "psi_band": psi_band(value),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "js": js_distance(ref[col], cur[col], bins),
                "chi2_pvalue": np.nan,
            }
        )
    for col in categorical:
        value = psi(ref[col], cur[col], bins)
        rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "psi": value,
                "psi_band": psi_band(value),
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "js": js_distance(ref[col], cur[col], bins),
                "chi2_pvalue": chi2_pvalue(ref[col], cur[col]),
            }
        )
    table = pd.DataFrame(rows)
    table["drift"] = table["psi_band"] == "significant"
    return table.sort_values("psi", ascending=False, ignore_index=True)
