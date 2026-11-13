"""Schemas da API. Tipos propositalmente permissivos: a validacao de negocio e do contrato
Pandera (que devolve 422 com o nome da regra violada). O pydantic so garante a FORMA:
campos obrigatorios presentes e nenhum campo extra (atributos protegidos sao recusados).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

from credit_guard.config import INPUT_COLUMNS

Value = float | int | str | None

CreditApplication = create_model(
    "CreditApplication",
    __config__=ConfigDict(extra="forbid"),
    client_id=(str, Field(..., description="ID pseudonimizado do cliente")),
    **{col: (Value, Field(...)) for col in INPUT_COLUMNS},
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CreditApplication] = Field(..., min_length=1, max_length=5000)  # type: ignore[valid-type]
    batch_id: str | None = None


class Prediction(BaseModel):
    client_id: str
    probability_default: float
    decision: str


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]
