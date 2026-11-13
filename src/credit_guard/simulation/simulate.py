"""Estagio `simulate`: gera os meses de producao (drift gradual) e o lote sujo."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from credit_guard.config import DATA_DIR, REPORTS_DIR, TARGET_COLUMN, load_params
from credit_guard.logging_utils import get_logger
from credit_guard.observability.pipeline import StageContext, track_stage
from credit_guard.simulation.dirty import make_dirty_batch
from credit_guard.simulation.drift import apply_concept_drift, apply_data_drift, concept_flip_rate

log = get_logger("simulate")


def month_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "mean_income": float(df["AMT_INCOME_TOTAL"].mean()),
        "mean_credit": float(df["AMT_CREDIT"].mean()),
        "unemployed_share": float((df["NAME_INCOME_TYPE"] == "Unemployed").mean()),
        "mean_ext_source_2": float(df["EXT_SOURCE_2"].mean()),
        "default_rate": float(df[TARGET_COLUMN].mean()),
    }


def _run(ctx: StageContext) -> None:
    params = load_params()
    sim, seed = params["simulation"], params["seed"]
    rng = np.random.default_rng(seed)
    holdout = pd.read_parquet(DATA_DIR / "holdout" / "holdout.parquet")
    holdout = holdout.sample(frac=1, random_state=seed).reset_index(drop=True)

    n, months = sim["rows_per_month"], sim["n_months"]
    if n * months > len(holdout):
        raise ValueError(f"holdout pequeno demais: {len(holdout)} < {n * months}")

    out_dir = DATA_DIR / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"params": sim, "months": {}}
    for m in range(1, months + 1):
        base = holdout.iloc[(m - 1) * n : m * n]
        drifted = apply_concept_drift(apply_data_drift(base, m, sim, rng), m, sim, rng)
        drifted.to_parquet(out_dir / f"month_{m:02d}.parquet", index=False)
        ctx.add_rows("generated", len(drifted))
        manifest["months"][f"{m:02d}"] = {
            "factors": {
                "price_multiplier": 1 + sim["price_inflation_per_month"] * m,
                "income_multiplier": 1 - sim["real_income_erosion_per_month"] * m,
                "working_to_unemployed_share": sim["unemployment_shift_per_month"] * m,
                "ext_source_decline": sim["ext_source_decline_per_month"] * m,
                "concept_flip_rate": concept_flip_rate(m, sim),
            },
            "before": month_profile(base),
            "after": month_profile(drifted),
        }

    rest = holdout.iloc[months * n :]
    quarantined = pd.read_parquet(DATA_DIR / "quarantine" / "raw_quarantined.parquet")
    dirty, dirty_manifest = make_dirty_batch(rest, quarantined, sim["dirty_rows"], rng)
    (DATA_DIR / "dirty").mkdir(parents=True, exist_ok=True)
    dirty.to_parquet(DATA_DIR / "dirty" / "dirty_batch.parquet", index=False)
    manifest["dirty_batch"] = dirty_manifest

    (REPORTS_DIR / "simulation").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "simulation" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info(
        "simulate_done", extra={"months": months, "rows_per_month": n, "dirty": dirty_manifest}
    )


def main() -> None:
    with track_stage("simulate") as ctx:
        _run(ctx)


if __name__ == "__main__":
    main()
