"""Atribuicao empirica do deslocamento de score a cada feature (analise causal intervencionista).

Para cada feature, "desfaz" o drift substituindo seus valores atuais pelos valores da
referencia no mesmo quantil (numericas) ou reamostrando a distribuicao de referencia
(categoricas), mantendo as demais features como estao, e re-pontua. A parcela do
deslocamento medio de score que some com a intervencao e atribuida a essa feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_guard.monitoring.stats import _is_numeric


def _restore(ref: pd.Series, cur: pd.Series, rng: np.random.Generator) -> pd.Series:
    if _is_numeric(ref) and _is_numeric(cur):
        ref_valid = ref.dropna().to_numpy(float)
        ranks = cur.rank(pct=True, method="average")
        mapped = pd.Series(np.nan, index=cur.index)
        ok = ranks.notna()
        mapped[ok] = np.quantile(ref_valid, ranks[ok].clip(0, 1).to_numpy())
        return mapped
    freq = ref.value_counts(normalize=True, dropna=False)
    return pd.Series(
        rng.choice(freq.index.to_numpy(), size=len(cur), p=freq.to_numpy()), index=cur.index
    )


def drift_attribution(
    model,
    meta: dict,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    threshold = meta["threshold"]
    score_ref = model.predict_proba(reference)[:, 1].mean()
    score_cur_all = model.predict_proba(current)[:, 1]
    score_cur = score_cur_all.mean()
    approval_cur = (score_cur_all < threshold).mean()
    total_shift = score_cur - score_ref

    rows = []
    for feat in features:
        restored = current.copy()
        restored[feat] = _restore(reference[feat], current[feat], rng)
        s = model.predict_proba(restored)[:, 1]
        delta = score_cur - s.mean()
        rows.append(
            {
                "feature": feat,
                "mean_score_drifted": float(score_cur),
                "mean_score_restored": float(s.mean()),
                "delta_score": float(delta),
                "delta_approval": float((s < threshold).mean() - approval_cur),
                "share_of_score_shift": float(delta / total_shift) if total_shift else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "share_of_score_shift", ascending=False, ignore_index=True
    )
