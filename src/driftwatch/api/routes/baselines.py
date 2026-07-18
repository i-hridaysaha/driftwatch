from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_db
from driftwatch.api.validation import extract_segment_values, validate_features
from driftwatch.config.loader import ModelConfigNotFoundError, load_model_config
from driftwatch.db.models import Baseline, BaselineRecord, Model
from driftwatch.stats.binning import compute_baseline_binning

router = APIRouter()


class BaselineRecordIn(BaseModel):
    model_config = {"extra": "forbid"}

    features: dict[str, Any]
    prediction_score: float | None = None


class BaselineRegisterRequest(BaseModel):
    model_config = {"extra": "forbid"}

    records: list[BaselineRecordIn] = Field(min_length=1)


class BaselineRegisterResponse(BaseModel):
    baseline_id: int
    n_records: int


@router.post(
    "/models/{model_id}/baseline",
    response_model=BaselineRegisterResponse,
    status_code=201,
)
def register_baseline(
    model_id: str, body: BaselineRegisterRequest, db: Session = Depends(get_db)
) -> BaselineRegisterResponse:
    try:
        config = load_model_config(model_id)
    except ModelConfigNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    errors: list[str] = []
    for i, record in enumerate(body.records):
        record_errors = validate_features(record.features, config.schema_)
        errors.extend(f"records[{i}]: {message}" for message in record_errors)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    if db.get(Model, model_id) is None:
        db.add(Model(model_id=model_id))
        db.flush()

    db.execute(
        update(Baseline).where(Baseline.model_id == model_id, Baseline.is_active.is_(True)).values(
            is_active=False
        )
    )

    binning_config = compute_baseline_binning(
        features=[record.features for record in body.records],
        prediction_scores=[record.prediction_score for record in body.records],
        schema=config.schema_,
    )
    baseline = Baseline(model_id=model_id, is_active=True, binning_config=binning_config)
    db.add(baseline)
    db.flush()

    for record in body.records:
        segment_values = extract_segment_values(record.features, config.segments.dimensions)
        db.add(
            BaselineRecord(
                baseline_id=baseline.id,
                features=record.features,
                prediction_score=record.prediction_score,
                segment_values=segment_values,
            )
        )

    return BaselineRegisterResponse(baseline_id=baseline.id, n_records=len(body.records))
