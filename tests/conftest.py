from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_db
from driftwatch.api.main import create_app
from driftwatch.settings import get_settings


@pytest.fixture(scope="module")
def alembic_config() -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


@pytest.fixture(scope="module")
def migrated_db(alembic_config: Config) -> Iterator[None]:
    command.upgrade(alembic_config, "head")
    yield
    command.downgrade(alembic_config, "base")


@pytest.fixture
def db_session(migrated_db: None) -> Iterator[Session]:
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app = create_app()

    def override_get_db() -> Iterator[Session]:
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
