"""실제 DB·운영 비밀값 없이 배포 준비 상태 응답을 검증한다."""

from unittest.mock import MagicMock

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend import main, readiness


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    connect = MagicMock()
    connection = connect.return_value.__enter__.return_value
    connection.execute.return_value.fetchone.return_value = (True,)
    monkeypatch.setattr(readiness.psycopg, "connect", connect)
    return connect, connection


def test_ready_with_recommendable_track(database):
    connect, connection = database
    response = TestClient(main.app).get("/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert connect.call_args.args[0] == "postgresql://test:test@localhost/test"
    assert connect.call_args.kwargs["connect_timeout"] == 2
    query = connection.execute.call_args.args[0]
    assert "emb_gemini IS NOT NULL AND store_url IS NOT NULL" in query


def test_not_ready_without_recommendable_track(database):
    _, connection = database
    connection.execute.return_value.fetchone.return_value = (False,)
    response = TestClient(main.app).get("/readiness")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


@pytest.mark.parametrize("stage", ["connect", "table", "query_timeout"])
def test_not_ready_on_database_failure(database, stage):
    connect, connection = database
    if stage == "connect":
        connect.side_effect = psycopg.OperationalError("private connection details")
    elif stage == "table":
        connection.execute.side_effect = psycopg.errors.UndefinedTable("private table details")
    else:
        connection.execute.side_effect = psycopg.errors.QueryCanceled("private query details")
    client = TestClient(main.app)
    response = client.get("/readiness")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("url", [None, "", "invalid-url", "sqlite:///test.db"])
def test_not_ready_with_invalid_configuration(database, monkeypatch, url):
    connect, _ = database
    if url is None:
        monkeypatch.delenv("DATABASE_URL")
    else:
        monkeypatch.setenv("DATABASE_URL", url)
    assert TestClient(main.app).get("/readiness").status_code == 503
    connect.assert_not_called()
