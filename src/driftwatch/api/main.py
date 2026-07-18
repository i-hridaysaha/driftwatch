from fastapi import FastAPI

from driftwatch.api.routes.baselines import router as baselines_router
from driftwatch.api.routes.health import router as health_router
from driftwatch.api.routes.labels import router as labels_router
from driftwatch.api.routes.predictions import router as predictions_router


def create_app() -> FastAPI:
    app = FastAPI(title="driftwatch")
    app.include_router(health_router)
    app.include_router(baselines_router)
    app.include_router(predictions_router)
    app.include_router(labels_router)
    return app


app = create_app()
