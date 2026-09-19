import logging

from fastapi import Depends, FastAPI

from driftwatch.api.deps import require_api_key
from driftwatch.api.routes.baselines import router as baselines_router
from driftwatch.api.routes.health import router as health_router
from driftwatch.api.routes.labels import router as labels_router
from driftwatch.api.routes.predictions import router as predictions_router
from driftwatch.settings import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="driftwatch")
    app.include_router(health_router)
    # every write goes through the key check; the liveness probe does not
    write = [Depends(require_api_key)]
    app.include_router(baselines_router, dependencies=write)
    app.include_router(predictions_router, dependencies=write)
    app.include_router(labels_router, dependencies=write)
    if get_settings().api_key is None:
        logger.warning(
            "API_KEY is not set: the baseline, prediction and label endpoints "
            "are open to anything that can reach this process"
        )
    return app


app = create_app()
