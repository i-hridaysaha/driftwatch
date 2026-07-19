from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from driftwatch.settings import get_settings


@st.cache_resource
def _engine() -> Engine:
    """`-c timezone=utc` pins every connection's Postgres session timezone
    to UTC, regardless of the server's or host's own configured default.
    Every DateTime column in this project is timezone(True) and always
    written as UTC (driftwatch.db.models), but Postgres returns a
    TIMESTAMPTZ value converted into the SESSION's timezone on read, not
    necessarily UTC -- the returned instant is identical either way, but a
    non-UTC session would silently attach the wrong offset, and code that
    extracts a calendar date from it (default_time_range's `.date()`) could
    land on the wrong day for timestamps near a UTC midnight boundary.
    Forcing UTC here removes that dependency on wherever this happens to be
    deployed."""
    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=utc"},
    )


@contextmanager
def read_only_session() -> Iterator[Session]:
    """Every dashboard query goes through this session. `SET TRANSACTION
    READ ONLY` is issued as the first statement of a fresh transaction, so
    Postgres itself rejects any write the dashboard might ever accidentally
    issue -- enforcement at the database level, not just "the code we wrote
    happens not to contain an INSERT." The session is always rolled back on
    exit, never committed, even though a read-only transaction has nothing
    to commit.

    An ORM Session (not a bare Connection) so dashboard code can reuse
    existing ORM-level building blocks where it matters most --
    driftwatch.alerting.streaks.classify_drift/classify_performance for the
    suppressed-alerts view -- rather than reimplementing that classification
    logic a second time against raw rows.

    This connects with the same application role as the rest of the
    service rather than a dedicated read-only DB role/replica -- a real
    deployment would provision one; that's infra scope beyond this phase,
    and the transaction-level guard above is what actually makes "the
    dashboard issues no writes" true regardless.
    """
    engine = _engine()
    connection = engine.connect()
    connection.execute(text("SET TRANSACTION READ ONLY"))
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.rollback()
        connection.close()
