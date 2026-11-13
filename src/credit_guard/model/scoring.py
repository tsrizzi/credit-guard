"""Carga do modelo versionado e pontuacao de um lote."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_guard.config import ID_COLUMN, MODELS_DIR


def load_model(models_dir: Path | None = None) -> tuple[Pipeline, dict]:
    d = Path(models_dir or MODELS_DIR)
    model = joblib.load(d / "model.joblib")
    meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
    return model, meta


def score_frame(model: Pipeline, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df)[:, 1]
    return pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].to_numpy(),
            "probability_default": proba,
            "decision": np.where(proba < meta["threshold"], "approve", "reject"),
        },
        index=df.index,
    )
