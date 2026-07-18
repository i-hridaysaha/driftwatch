from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_db
from driftwatch.config.loader import ModelConfigNotFoundError, load_model_config
from driftwatch.db.models import EvaluationWindow, Label, Model, Prediction
from driftwatch.evaluation.performance import recompute_performance_for_window
from driftwatch.hashing import compute_payload_hash

router = APIRouter()


class LabelIn(BaseModel):
    model_config = {"extra": "forbid"}

    prediction_id: str
    label_value: float
    labeled_at: datetime


class LabelIngestRequest(BaseModel):
    model_config = {"extra": "forbid"}

    labels: list[LabelIn] = Field(min_length=1)


class LabelIngestResponse(BaseModel):
    inserted: int
    skipped_duplicate: int
    windows_recomputed: list[int]


def _payload(record: LabelIn) -> dict[str, Any]:
    return {"label_value": record.label_value, "labeled_at": record.labeled_at.isoformat()}


@router.post("/models/{model_id}/labels", response_model=LabelIngestResponse, status_code=201)
def ingest_labels(
    model_id: str, body: LabelIngestRequest, db: Session = Depends(get_db)
) -> LabelIngestResponse:
    try:
        load_model_config(model_id)
    except ModelConfigNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if db.get(Model, model_id) is None:
        raise HTTPException(
            status_code=404, detail=f"model {model_id!r} has no registered baseline yet"
        )

    incoming_ids = sorted({record.prediction_id for record in body.labels})
    predictions = {
        p.prediction_id: p
        for p in db.scalars(
            select(Prediction).where(
                Prediction.model_id == model_id,
                Prediction.prediction_id.in_(incoming_ids),
            )
        )
    }
    missing = sorted(set(incoming_ids) - set(predictions))
    if missing:
        raise HTTPException(
            status_code=422, detail=f"labels reference unknown prediction_id(s): {missing}"
        )

    existing_hashes: dict[str, set[str]] = {}
    for label in db.scalars(
        select(Label).where(Label.model_id == model_id, Label.prediction_id.in_(incoming_ids))
    ):
        existing_hashes.setdefault(label.prediction_id, set()).add(label.payload_hash)

    inserted = 0
    skipped = 0
    touched_window_ids: set[int] = set()
    seen_in_batch: set[tuple[str, str]] = set()

    for record in body.labels:
        payload_hash = compute_payload_hash(_payload(record))
        batch_key = (record.prediction_id, payload_hash)
        already_known = payload_hash in existing_hashes.get(record.prediction_id, set())
        if already_known or batch_key in seen_in_batch:
            skipped += 1
            continue
        seen_in_batch.add(batch_key)

        db.add(
            Label(
                prediction_id=record.prediction_id,
                model_id=model_id,
                label_value=record.label_value,
                labeled_at=record.labeled_at,
                payload_hash=payload_hash,
            )
        )
        inserted += 1

        predicted_at = predictions[record.prediction_id].predicted_at
        window = db.scalars(
            select(EvaluationWindow).where(
                EvaluationWindow.model_id == model_id,
                EvaluationWindow.window_start <= predicted_at,
                EvaluationWindow.window_end > predicted_at,
            )
        ).first()
        if window is not None:
            touched_window_ids.add(window.id)

    db.flush()
    for window_id in touched_window_ids:
        recompute_performance_for_window(db, window_id)

    return LabelIngestResponse(
        inserted=inserted,
        skipped_duplicate=skipped,
        windows_recomputed=sorted(touched_window_ids),
    )
