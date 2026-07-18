import pytest
from fastapi.testclient import TestClient

from driftwatch.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
