"""Connection smoke test (Migration 014 leftover from Phase A's seam).

The driver seam introduced in Phase A and inlined in Phase G is gone;
this test stays as a basic ``SELECT 1`` round-trip + SQLAlchemy
``postgresql+psycopg`` dialect-driver assertion.  It pins:

- ``psycopg.connect()`` works against the testcontainer.
- ``sa.engine.URL.create(drivername="postgresql+psycopg")`` produces an
  engine whose ``dialect.driver == "psycopg"`` -- guards against an
  accidental dialect regression now that we own the dialect string
  inline rather than via a ``SA_DRIVERNAME`` constant.

The file name is preserved for reviewer continuity with Phase A; the
contents are now driver-direct.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.integration


def test_psycopg_connect_round_trips_select_one(pg_connection_params: dict[str, Any]) -> None:
    """A bare ``SELECT 1`` round-trips through ``psycopg.connect()``."""
    conn = psycopg.connect(**pg_connection_params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
        assert row == (1,)
    finally:
        conn.close()


def test_sa_dialect_resolves_to_psycopg(pg_connection_params: dict[str, Any]) -> None:
    """SQLAlchemy ``postgresql+psycopg`` dialect URL produces a psycopg-backed engine."""
    url = sa.engine.URL.create(
        drivername="postgresql+psycopg",
        username=pg_connection_params["user"],
        password=pg_connection_params["password"],
        host=pg_connection_params["host"],
        port=pg_connection_params["port"],
        database=pg_connection_params["dbname"],
    )
    engine = sa.create_engine(url)
    try:
        assert engine.dialect.driver == "psycopg"
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1")).scalar()
        assert result == 1
    finally:
        engine.dispose()
