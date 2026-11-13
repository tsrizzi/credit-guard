"""Estagio `train`: treina o baseline na referencia, fixa o threshold de negocio e registra no MLflow."""

from __future__ import annotations

import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from credit_guard.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ROOT, TARGET_COLUMN, load_params
from credit_guard.logging_utils import get_logger
from credit_guard.model.features import MODEL_FEATURES, base_feature_importance, build_pipeline
from credit_guard.model.metrics import evaluate_scores
from credit_guard.observability.pipeline import StageContext, track_stage

log = get_logger("train")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
TOP_N_FEATURES = 5


def fit_threshold(proba_val, target_approval_rate: float) -> float:
    """Threshold de PD tal que a fracao aprovada (proba < threshold) seja a meta de negocio."""
    return float(np.quantile(np.asarray(proba_val), target_approval_rate))


def _file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(ctx: StageContext) -> None:
    import mlflow
    from mlflow.models import infer_signature

    params = load_params()
    model_params = dict(params["model"])
    target_rate = model_params.pop("target_approval_rate")
    train_path = DATA_DIR / "reference" / "train.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(DATA_DIR / "reference" / "test.parquet")

    pipe = build_pipeline(model_params, seed=params["seed"])
    pipe.fit(train, train[TARGET_COLUMN])
    ctx.add_rows("trained", len(train))
    proba_test = pipe.predict_proba(test)[:, 1]
    threshold = fit_threshold(proba_test, target_rate)
    metrics = evaluate_scores(test[TARGET_COLUMN], proba_test, threshold)

    importance = base_feature_importance(pipe)
    version_src = json.dumps(params["model"], sort_keys=True) + _file_md5(train_path)
    model_version = hashlib.sha256(version_src.encode()).hexdigest()[:10]
    meta = {
        "model_version": model_version,
        "threshold": threshold,
        "target_approval_rate": target_rate,
        "features": MODEL_FEATURES,
        "top_features": importance.head(TOP_N_FEATURES).index.tolist(),
        "reference_metrics": metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = REPORTS_DIR / "model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"threshold": threshold, **metrics}, indent=2), encoding="utf-8"
    )
    importance.rename("importance").to_frame().to_csv(
        out / "feature_importance.csv", index_label="feature"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("credit-training")
    with mlflow.start_run(run_name=f"baseline-{model_version}"):
        mlflow.log_params(
            {**model_params, "target_approval_rate": target_rate, "seed": params["seed"]}
        )
        mlflow.log_metrics({**metrics, "threshold": threshold})
        mlflow.set_tags(
            {"model_version": model_version, "dataset": "home-credit/application_train"}
        )
        sample = test.head(5).astype({c: float for c in test.select_dtypes("number").columns})
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
            registered_model_name="credit-scoring",
        )
        mlflow.log_artifact(str(out / "feature_importance.csv"))

    log.info(
        "train_done", extra={"model_version": model_version, **metrics, "threshold": threshold}
    )


def main() -> None:
    with track_stage("train") as ctx:
        _run(ctx)


if __name__ == "__main__":
    main()
