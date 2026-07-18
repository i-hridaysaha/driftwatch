from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_db
from driftwatch.api.idempotency import compute_payload_hash
from driftwatch.api.validation import extract_segment_values, validate_features
from driftwatch.config.loader import ModelConfigNotFoundError, load_model_config
from driftwatch.db.models import Model, Prediction

router = APIRouter()


class PredictionIn(BaseModel):
    model_config = {"extra": "forbid"}

    prediction_id: str
    predicted_at: datetime
    features: dict[str, Any]
    prediction_value: dict[str, Any]
    prediction_score: float | None = None


class PredictionIngestRequest(BaseModel):
    model_config = {"extra": "forbid"}

    predictions: list[PredictionIn] = Field(min_length=1)


class PredictionIngestResponse(BaseModel):
    inserted: int
    skipped_duplicate: int


def _payload(record: PredictionIn) -> dict[str, Any]:
    return {
        "predicted_at": record.predicted_at.isoformat(),
        "features": record.features,
        "prediction_value": record.prediction_value,
        "prediction_score": record.prediction_score,
    }


@router.post(
    "/models/{model_id}/predictions",
    response_model=PredictionIngestResponse,
    status_code=201,
)
def ingest_predictions(
    model_id: str, body: PredictionIngestRequest, db: Session = Depends(get_db)
) -> PredictionIngestResponse:
    try:
        config = load_model_config(model_id)
    except ModelConfigNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if db.get(Model, model_id) is None:
        raise HTTPException(
            status_code=404, detail=f"model {model_id!r} has no registered baseline yet"
        )

    validation_errors: list[str] = []
    for i, record in enumerate(body.predictions):
        validation_errors.extend(
            f"predictions[{i}]: {message}"
            for message in validate_features(record.features, config.schema_)
        )
    if validation_errors:
        raise HTTPException(status_code=422, detail=validation_errors)

    # Same prediction_id can legitimately repeat within a batch (harmless retry);
    # a repeat with a *different* payload is a client bug, caught before any writes.
    payload_hashes: dict[str, str] = {}
    representative: dict[str, PredictionIn] = {}
    batch_conflicts: set[str] = set()
    for record in body.predictions:
        payload_hash = compute_payload_hash(_payload(record))
        prior_hash = payload_hashes.get(record.prediction_id)
        if prior_hash is not None and prior_hash != payload_hash:
            batch_conflicts.add(record.prediction_id)
            continue
        payload_hashes[record.prediction_id] = payload_hash
        representative[record.prediction_id] = record
    if batch_conflicts:
        raise HTTPException(
            status_code=422,
            detail=(
                "request contains conflicting duplicate prediction_ids: "
                f"{sorted(batch_conflicts)}"
            ),
        )

    existing = {
        p.prediction_id: p
        for p in db.scalars(
            select(Prediction).where(
                Prediction.model_id == model_id,
                Prediction.prediction_id.in_(representative.keys()),
            )
        )
    }

    to_insert: dict[str, PredictionIn] = {}
    conflicts: list[str] = []
    skipped = 0
    for prediction_id, record in representative.items():
        existing_row = existing.get(prediction_id)
        if existing_row is None:
            to_insert[prediction_id] = record
        elif existing_row.payload_hash == payload_hashes[prediction_id]:
            skipped += 1
        else:
            conflicts.append(prediction_id)

    if conflicts:
        raise HTTPException(
            status_code=409,
            detail=f"prediction_id already exists with a different payload: {sorted(conflicts)}",
        )

    for prediction_id, record in to_insert.items():
        segment_values = extract_segment_values(record.features, config.segments.dimensions)
        db.add(
            Prediction(
                prediction_id=prediction_id,
                model_id=model_id,
                predicted_at=record.predicted_at,
                features=record.features,
                prediction_value=record.prediction_value,
                prediction_score=record.prediction_score,
                segment_values=segment_values,
                payload_hash=payload_hashes[prediction_id],
            )
        )

    return PredictionIngestResponse(inserted=len(to_insert), skipped_duplicate=skipped)
