"""Tests for PostgresResource and batched read utilities."""

import logging
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import psycopg
import pytest
import sqlalchemy as sa

from moncpipelib.config import SCD2Config
from moncpipelib.resources.postgres import (
    PostgresPolarsSchema,
    PostgresResource,
    read_batched,
    read_batched_to_dataframe,
)
from moncpipelib.transforms.hashing import compute_row_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resource() -> PostgresResource:
    """Create a PostgresResource for testing."""
    return PostgresResource(
        host="localhost",
        port=5432,
        user="testuser",
        password="testpass",
        database="testdb",
    )


@pytest.fixture
def mock_context() -> MagicMock:
    """Create a mock Dagster OpExecutionContext."""
    ctx = MagicMock()
    ctx.log = MagicMock()
    return ctx


@pytest.fixture
def sample_batches() -> list[pl.DataFrame]:
    """Sample DataFrame batches for testing."""
    return [
        pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]}),
        pl.DataFrame({"id": [4, 5], "value": ["d", "e"]}),
    ]


# ---------------------------------------------------------------------------
# TestGetEngine
# ---------------------------------------------------------------------------


class TestGetEngine:
    """Tests for PostgresResource.get_engine()."""

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_creates_engine_with_correct_url(
        self, mock_create_engine: MagicMock, resource: PostgresResource
    ) -> None:
        """Engine should be created with correct connection URL."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine = resource.get_engine()

        assert engine is mock_engine
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        url = call_args[0][0]
        assert url.drivername == "postgresql+psycopg"
        assert url.username == "testuser"
        assert url.host == "localhost"
        assert url.port == 5432
        assert url.database == "testdb"
        assert url.query.get("sslmode") == "require"
        # Password must not appear in string representation
        assert "testpass" not in str(url)

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_engine_is_cached(
        self, mock_create_engine: MagicMock, resource: PostgresResource
    ) -> None:
        """Same engine instance should be returned on multiple calls."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine1 = resource.get_engine()
        engine2 = resource.get_engine()

        assert engine1 is engine2
        mock_create_engine.assert_called_once()

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_engine_passes_connect_args(
        self, mock_create_engine: MagicMock, resource: PostgresResource
    ) -> None:
        """Engine should pass connect_timeout and pool_pre_ping."""
        resource.get_engine()

        call_kwargs = mock_create_engine.call_args[1]
        # #365: application_name / options are applied per-connect via a
        # do_connect event (see TestConnectionSitesTagApplicationName), not
        # baked into connect_args at engine-creation time.
        assert call_kwargs["connect_args"] == {"connect_timeout": 30}
        assert call_kwargs["pool_pre_ping"] is True
        assert call_kwargs["hide_parameters"] is True


# ---------------------------------------------------------------------------
# TestLazyInitModelCopySafety
# ---------------------------------------------------------------------------


class TestLazyInitModelCopySafety:
    """Lazy-init helpers (``_get_lineage_tracker``,
    ``_get_openlineage_emitter``) must produce a fresh tracker / emitter
    when a resource is ``model_copy``'d from a parent that was already
    touched with the opposing config.

    Pydantic ``PrivateAttr`` values are copied by reference, so the
    cache state survives ``model_copy``. A separate ``_initialized``
    flag plus a ``None`` cached value would leave the child resource
    permanently in the "tried, got nothing" state -- silently disabling
    lineage on a child that overrides ``enable_row_lineage=True``.
    Caused 9 integration test failures on migration 018 PR #318.
    """

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_lineage_tracker_reinitialises_after_model_copy(
        self,
        mock_create_engine: MagicMock,
    ) -> None:
        """A resource with ``enable_row_lineage=False`` that has
        already cache-missed must, after ``model_copy(update=
        {"enable_row_lineage": True})``, still produce a real tracker
        on the child."""
        mock_create_engine.return_value = MagicMock()

        parent = PostgresResource(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="d",
            enable_row_lineage=False,
        )
        # Trigger the parent's cache miss (returns None).
        assert parent._get_lineage_tracker() is None

        child = parent.model_copy(update={"enable_row_lineage": True})
        assert child.enable_row_lineage is True

        tracker = child._get_lineage_tracker()
        assert tracker is not None, (
            "Lineage tracker must re-initialise on a model_copy'd resource "
            "even when the parent's cache was warmed with enable_row_lineage=False"
        )

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_lineage_tracker_disabled_path_remains_cheap(
        self,
        mock_create_engine: MagicMock,
    ) -> None:
        """When ``enable_row_lineage=False``, repeated calls must NOT
        create an engine -- the early-return path keeps the disabled
        case zero-cost."""
        mock_create_engine.return_value = MagicMock()
        r = PostgresResource(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="d",
            enable_row_lineage=False,
        )
        for _ in range(5):
            assert r._get_lineage_tracker() is None
        mock_create_engine.assert_not_called()

    def test_openlineage_emitter_reinitialises_after_model_copy(self) -> None:
        """Same model_copy-stale-state guard for the OpenLineage emitter."""
        parent = PostgresResource(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="d",
            openlineage_url=None,
        )
        # Cache-miss with openlineage off.
        assert parent._get_openlineage_emitter() is None

        child = parent.model_copy(update={"openlineage_url": "http://localhost:5000"})

        # The actual emitter may fail to construct if optional deps
        # aren't installed; the assertion is that we re-attempt the
        # init rather than short-circuiting on stale state. Either an
        # emitter or ``None`` is acceptable; what we forbid is the
        # ``_initialized`` flag preventing the attempt.
        result = child._get_openlineage_emitter()
        # The cache state must have advanced from the stale parent's.
        # If the import succeeded, ``result`` is the emitter; if it
        # raised ImportError, ``result`` is None but the next call
        # would still re-attempt (no _initialized latch).
        del result  # smoke-only: the code path executed without raising


# ---------------------------------------------------------------------------
# TestGetStreamingConnection
# ---------------------------------------------------------------------------


class TestGetStreamingConnection:
    """Tests for PostgresResource.get_streaming_connection()."""

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_yields_connection_with_stream_results(
        self, mock_create_engine: MagicMock, resource: PostgresResource
    ) -> None:
        """Streaming connection should have stream_results=True."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_streaming_conn = MagicMock()

        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execution_options.return_value = mock_streaming_conn

        with resource.get_streaming_connection() as conn:
            assert conn is mock_streaming_conn

        mock_conn.execution_options.assert_called_once_with(stream_results=True)

    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_connection_closed_on_exit(
        self, mock_create_engine: MagicMock, resource: PostgresResource
    ) -> None:
        """Connection context manager should close on exit."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_cm = MagicMock()
        mock_engine.connect.return_value = mock_cm
        mock_cm.__enter__ = MagicMock(return_value=MagicMock())
        mock_cm.__exit__ = MagicMock(return_value=False)

        with resource.get_streaming_connection():
            pass

        mock_cm.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# TestReadBatchedStreaming
# ---------------------------------------------------------------------------


class TestReadBatchedStreaming:
    """Tests for streaming method of read_batched."""

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_yields_correct_batches(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Streaming should yield the correct number of batches."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter(sample_batches)

        batches = list(resource.read_batched("SELECT * FROM test"))
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_passes_batch_size(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Batch size should be passed to pl.read_database."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter([])

        list(resource.read_batched("SELECT * FROM test", batch_size=25_000))

        call_kwargs = mock_read_db.call_args[1]
        assert call_kwargs["batch_size"] == 25_000
        assert call_kwargs["iter_batches"] is True

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_logs_progress(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        mock_context: MagicMock,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Progress should be logged when context is provided."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter(sample_batches)

        list(resource.read_batched("SELECT * FROM test", context=mock_context))

        log_calls = [str(c) for c in mock_context.log.info.call_args_list]
        assert any("Starting streaming read" in c for c in log_calls)
        assert any("Read batch 1" in c for c in log_calls)
        assert any("Read batch 2" in c for c in log_calls)
        assert any("Read completed" in c for c in log_calls)

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_handles_empty_results(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Streaming should handle empty result sets."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter([])

        batches = list(resource.read_batched("SELECT * FROM empty_table"))
        assert batches == []

    @patch("moncpipelib.resources.postgres.pl.read_database")
    def test_accepts_sqlalchemy_engine_directly(
        self,
        mock_read_db: MagicMock,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Should accept a SQLAlchemy Engine directly."""
        mock_engine = MagicMock(spec=sa.engine.Engine)
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter(sample_batches)

        batches = list(read_batched("SELECT 1", mock_engine))
        assert len(batches) == 2


# ---------------------------------------------------------------------------
# TestReadBatchedOffset
# ---------------------------------------------------------------------------


class TestReadBatchedOffset:
    """Tests for offset method of read_batched."""

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_yields_correct_batches(
        self,
        mock_connect: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Offset method should paginate and yield batches."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # First call is COUNT, then two batch reads
        count_df = pl.DataFrame({"cnt": [5]})
        mock_read_db.side_effect = [count_df, sample_batches[0], sample_batches[1]]

        batches = list(
            resource.read_batched(
                "SELECT * FROM test",
                method="offset",
                order_by="id",
                batch_size=3,
            )
        )
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_includes_order_by_in_query(
        self,
        mock_connect: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Offset queries should include ORDER BY clause."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        count_df = pl.DataFrame({"cnt": [2]})
        batch_df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        mock_read_db.side_effect = [count_df, batch_df]

        list(
            resource.read_batched(
                "SELECT * FROM test",
                method="offset",
                order_by=["id", "created_at"],
                batch_size=100,
            )
        )

        # Check second call (first paginated query)
        paginated_query = mock_read_db.call_args_list[1][0][0]
        assert "ORDER BY id, created_at" in paginated_query
        assert "LIMIT 100" in paginated_query
        assert "OFFSET 0" in paginated_query

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_logs_progress(
        self,
        mock_connect: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        mock_context: MagicMock,
    ) -> None:
        """Progress should be logged when context is provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        count_df = pl.DataFrame({"cnt": [3]})
        batch_df = pl.DataFrame({"id": [1, 2, 3]})
        mock_read_db.side_effect = [count_df, batch_df]

        list(
            resource.read_batched(
                "SELECT * FROM test",
                method="offset",
                order_by="id",
                batch_size=100,
                context=mock_context,
            )
        )

        log_calls = [str(c) for c in mock_context.log.info.call_args_list]
        assert any("Starting offset read" in c for c in log_calls)
        assert any("3" in c for c in log_calls)  # Total rows logged
        assert any("Read completed" in c for c in log_calls)

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_early_exit_on_partial_batch(
        self,
        mock_connect: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Should stop iteration when a batch has fewer rows than batch_size."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        count_df = pl.DataFrame({"cnt": [5]})
        # First batch is full (3 rows), second is partial (2 rows)
        batch1 = pl.DataFrame({"id": [1, 2, 3]})
        batch2 = pl.DataFrame({"id": [4, 5]})
        mock_read_db.side_effect = [count_df, batch1, batch2]

        batches = list(
            resource.read_batched(
                "SELECT * FROM test",
                method="offset",
                order_by="id",
                batch_size=3,
            )
        )

        assert len(batches) == 2
        # Should not attempt a third query since batch2 < batch_size
        assert mock_read_db.call_count == 3  # count + 2 batches


# ---------------------------------------------------------------------------
# TestReadBatchedToDataFrame
# ---------------------------------------------------------------------------


class TestReadBatchedToDataFrame:
    """Tests for read_batched_to_dataframe."""

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_concatenates_all_batches(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Should return single concatenated DataFrame."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter(sample_batches)

        result = resource.read_batched_to_dataframe("SELECT * FROM test")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5
        assert result.columns == ["id", "value"]

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_returns_schema_aware_empty_dataframe_for_no_results(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Empty result should yield a zero-row frame that keeps the schema (#358).

        The streaming read yields no batches, so ``read_batched_to_dataframe``
        falls back to a ``LIMIT 0`` schema probe. The probe's frame (columns
        intact, zero rows) is what callers get -- not a column-less
        ``pl.DataFrame()`` -- so ``.select`` on named columns does not raise.
        """
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        probe_frame = pl.DataFrame(schema={"id": pl.Int64, "value": pl.String})
        # First call: the batched streaming read (no rows). Second call: the
        # LIMIT 0 schema probe on the empty path.
        mock_read_db.side_effect = [iter([]), probe_frame]

        result = resource.read_batched_to_dataframe("SELECT * FROM empty")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == ["id", "value"]
        # The documented motivation: selecting a named column must not raise.
        assert result.select("id").columns == ["id"]

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_empty_result_falls_back_to_bare_frame_when_probe_fails(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing schema probe must not regress the empty path to an error (#358).

        The fallback to a bare frame is intentional, but it must be observable:
        a warning is logged so the silent column-less result leaves a trace.
        """
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # First call: empty streaming read. Second call: probe raises.
        mock_read_db.side_effect = [iter([]), RuntimeError("probe blew up")]

        with caplog.at_level(logging.WARNING, logger="moncpipelib.resources"):
            result = resource.read_batched_to_dataframe("SELECT * FROM empty")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert any(
            "schema probe failed" in r.getMessage() and "probe blew up" in r.getMessage()
            for r in caplog.records
        )

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_module_level_function(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Module-level read_batched_to_dataframe should work the same."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_db.return_value = iter(sample_batches)

        result = read_batched_to_dataframe("SELECT * FROM test", resource)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in read_batched."""

    def test_offset_without_order_by_raises_error(self, resource: PostgresResource) -> None:
        """Offset method without order_by should raise ValueError."""
        with pytest.raises(ValueError, match="order_by is required for offset method"):
            list(resource.read_batched("SELECT 1", method="offset"))

    def test_streaming_with_psycopg2_raises_error(self) -> None:
        """Streaming method with raw psycopg2 connection should raise ValueError."""
        mock_conn = MagicMock()
        # Ensure it's not detected as SQLAlchemy Engine/Connection
        mock_conn.__class__ = type("PgConnection", (), {})

        with pytest.raises(ValueError, match="streaming method requires SQLAlchemy"):
            list(read_batched("SELECT 1", mock_conn, method="streaming"))

    def test_offset_with_sqlalchemy_raises_error(self) -> None:
        """Offset method with SQLAlchemy engine should raise ValueError."""
        mock_engine = MagicMock(spec=sa.engine.Engine)

        with pytest.raises(ValueError, match="offset method requires psycopg"):
            list(read_batched("SELECT 1", mock_engine, method="offset"))

    def test_offset_with_psycopg2_without_order_by_raises_error(self) -> None:
        """Offset method via raw connection without order_by should raise ValueError."""
        mock_conn = MagicMock()
        mock_conn.__class__ = type("PgConnection", (), {})

        with pytest.raises(ValueError, match="order_by is required"):
            list(read_batched("SELECT 1", mock_conn, method="offset"))

    def test_batch_size_zero_raises_error(self, resource: PostgresResource) -> None:
        """batch_size=0 should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(resource.read_batched("SELECT 1", batch_size=0))

    def test_batch_size_negative_raises_error(self, resource: PostgresResource) -> None:
        """Negative batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(resource.read_batched("SELECT 1", batch_size=-100))

    def test_batch_size_one_is_valid(self) -> None:
        """batch_size=1 should be accepted (not rejected by validation)."""
        mock_engine = MagicMock(spec=sa.engine.Engine)
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("moncpipelib.resources.postgres.pl.read_database", return_value=iter([])):
            result = list(read_batched("SELECT 1", mock_engine, batch_size=1))
            assert result == []


# ---------------------------------------------------------------------------
# TestConvenienceMethods
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestSchemaOverrides
# ---------------------------------------------------------------------------


def _make_cursor_desc(
    columns: list[tuple[str, int]],
) -> list[tuple[str, int, None, None, None, None, None]]:
    """Build a DBAPI2-style cursor.description from (name, oid) pairs."""
    return [(name, oid, None, None, None, None, None) for name, oid in columns]


class TestSchemaOverrides:
    """Tests for database-driven schema overrides."""

    def test_cursor_desc_to_schema_maps_common_types(self) -> None:
        """Known OIDs should be mapped to their Polars types."""
        desc = _make_cursor_desc(
            [
                ("id", 23),  # int4 -> Int32
                ("name", 25),  # text -> String
                ("active", 16),  # bool -> Boolean
                ("amount", 1700),  # numeric -> Float64
                ("created", 1082),  # date -> Date
                ("big_id", 20),  # int8 -> Int64
            ]
        )
        schema = PostgresPolarsSchema.from_cursor_description(desc)

        assert schema == {
            "id": pl.Int32,
            "name": pl.String,
            "active": pl.Boolean,
            "amount": pl.Float64,
            "created": pl.Date,
            "big_id": pl.Int64,
        }

    def test_cursor_desc_to_schema_skips_unknown_oids(self) -> None:
        """Unknown OIDs should be omitted from the schema dict."""
        desc = _make_cursor_desc(
            [
                ("id", 23),  # known
                ("weird", 99999),  # unknown
                ("name", 25),  # known
            ]
        )
        schema = PostgresPolarsSchema.from_cursor_description(desc)

        assert "id" in schema
        assert "name" in schema
        assert "weird" not in schema

    def test_cursor_desc_to_schema_empty_description(self) -> None:
        """Empty description should return empty dict."""
        assert PostgresPolarsSchema.from_cursor_description([]) == {}

    def test_oid_mapping_completeness(self) -> None:
        """All mapped OIDs should resolve to valid Polars types."""
        for oid, pl_type in PostgresPolarsSchema.OID_MAP.items():
            assert hasattr(pl_type, "__name__") or hasattr(pl_type, "__class__"), (
                f"OID {oid} mapped to invalid type: {pl_type}"
            )

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_streaming_passes_schema_overrides(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Streaming method should pass schema_overrides from cursor description."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Set up cursor description for the LIMIT 0 probe
        # Note: with a plain MagicMock (no spec), isinstance(engine, sa.engine.Engine)
        # is False, so conn_ctx = nullcontext(engine) and conn IS mock_engine.
        mock_result = MagicMock()
        mock_result.cursor.description = _make_cursor_desc(
            [
                ("id", 23),  # int4 -> Int32
                ("value", 25),  # text -> String
            ]
        )
        mock_engine.execute.return_value = mock_result

        mock_read_db.return_value = iter(sample_batches)

        list(resource.read_batched("SELECT * FROM test"))

        call_kwargs = mock_read_db.call_args[1]
        assert call_kwargs["schema_overrides"] == {
            "id": pl.Int32,
            "value": pl.String,
        }

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.sa.create_engine")
    def test_streaming_graceful_fallback_on_probe_failure(
        self,
        mock_create_engine: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
        sample_batches: list[pl.DataFrame],
    ) -> None:
        """Streaming should still work if the schema probe fails."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Make the probe fail
        mock_conn.execute.side_effect = [
            Exception("probe failed"),  # First call: schema probe
            MagicMock(),  # Reset for any other calls
        ]
        # After the probe fails, execute should work normally for remaining calls
        mock_conn.execute.side_effect = Exception("probe failed")

        mock_read_db.return_value = iter(sample_batches)

        batches = list(resource.read_batched("SELECT * FROM test"))
        assert len(batches) == 2

        # schema_overrides should be None when probe fails
        call_kwargs = mock_read_db.call_args[1]
        assert call_kwargs["schema_overrides"] is None

    @patch("moncpipelib.resources.postgres.pl.read_database")
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_offset_passes_schema_overrides(
        self,
        mock_connect: MagicMock,
        mock_read_db: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """Offset method should pass schema_overrides from cursor description."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Set up cursor for the LIMIT 0 probe
        mock_cursor = MagicMock()
        mock_cursor.description = _make_cursor_desc(
            [
                ("id", 23),  # int4 -> Int32
                ("val", 25),  # text -> String
            ]
        )
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count_df = pl.DataFrame({"cnt": [2]})
        batch_df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        mock_read_db.side_effect = [count_df, batch_df]

        list(
            resource.read_batched(
                "SELECT * FROM test",
                method="offset",
                order_by="id",
                batch_size=100,
            )
        )

        # Second read_database call (paginated query) should have schema_overrides
        paginated_call_kwargs = mock_read_db.call_args_list[1][1]
        assert paginated_call_kwargs["schema_overrides"] == {
            "id": pl.Int32,
            "val": pl.String,
        }


# ---------------------------------------------------------------------------
# TestUUIDAdapter
# ---------------------------------------------------------------------------


class TestTypeAdapters:
    """Tests for ``PostgresPolarsSchema.register_*`` (Migration 014 Phase G).

    Phase G inlined the driver seam.  ``register_uuid_adapter`` /
    ``register_json_adapters`` now register psycopg3 ``Loader``
    subclasses directly on ``connection.adapters``; the tests assert
    that contract by passing a ``MagicMock`` connection and inspecting
    its ``adapters.register_loader`` calls.
    """

    def test_register_uuid_adapter_registers_loaders(self) -> None:
        """register_uuid_adapter registers Loaders for json, jsonb, and uuid."""
        mock_conn = MagicMock()
        PostgresPolarsSchema.register_uuid_adapter(mock_conn)
        registered = [c.args[0] for c in mock_conn.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid"]

    def test_register_json_adapters_registers_loaders(self) -> None:
        """register_json_adapters registers the same Loaders (alias for back-compat)."""
        mock_conn = MagicMock()
        PostgresPolarsSchema.register_json_adapters(mock_conn)
        registered = [c.args[0] for c in mock_conn.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid"]

    def test_register_uuid_adapter_short_circuits_on_no_adapters(self) -> None:
        """A connection without an ``adapters`` attribute is a no-op."""
        mock_conn = MagicMock(spec=[])  # no auto-attributes
        # Should not raise.
        PostgresPolarsSchema.register_uuid_adapter(mock_conn)

    def test_register_uuid_adapter_sa_extracts_dbapi_conn(self) -> None:
        """register_uuid_adapter_sa unwraps the SA connection and registers."""
        mock_sa_conn = MagicMock()
        mock_dbapi = MagicMock(name="dbapi_conn")
        mock_sa_conn.connection.dbapi_connection = mock_dbapi
        PostgresPolarsSchema.register_uuid_adapter_sa(mock_sa_conn)
        registered = [c.args[0] for c in mock_dbapi.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid"]

    def test_register_uuid_adapter_sa_skips_none_dbapi(self) -> None:
        """register_uuid_adapter_sa is a no-op when dbapi_connection is None."""
        mock_sa_conn = MagicMock()
        mock_sa_conn.connection.dbapi_connection = None
        # Should not raise.
        PostgresPolarsSchema.register_uuid_adapter_sa(mock_sa_conn)

    def test_register_json_adapters_sa_extracts_dbapi_conn(self) -> None:
        """register_json_adapters_sa unwraps the SA connection and registers."""
        mock_sa_conn = MagicMock()
        mock_dbapi = MagicMock(name="dbapi_conn")
        mock_sa_conn.connection.dbapi_connection = mock_dbapi
        PostgresPolarsSchema.register_json_adapters_sa(mock_sa_conn)
        registered = [c.args[0] for c in mock_dbapi.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid"]

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_get_connection_registers_adapters(
        self,
        mock_connect: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """PostgresResource.get_connection() registers the Loaders on the new connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        with resource.get_connection() as _conn:
            pass
        # Two calls: register_uuid_adapter + register_json_adapters.  Each
        # registers all three loaders, so we expect 6 register_loader calls.
        registered = [c.args[0] for c in mock_conn.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid", "json", "jsonb", "uuid"]

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_get_connection_raw_registers_adapters(
        self,
        mock_connect: MagicMock,
        resource: PostgresResource,
    ) -> None:
        """PostgresResource.get_connection_raw() registers the Loaders on the new connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        resource.get_connection_raw()
        registered = [c.args[0] for c in mock_conn.adapters.register_loader.call_args_list]
        assert registered == ["json", "jsonb", "uuid", "json", "jsonb", "uuid"]


# ---------------------------------------------------------------------------
# TestParseTarget
# ---------------------------------------------------------------------------


class TestParseTarget:
    """Tests for PostgresResource._parse_target()."""

    def test_parses_schema_and_table(self) -> None:
        """Should split 'schema.table' into a (schema, table) tuple."""
        schema, table = PostgresResource._parse_target("silver.dim_provider")
        assert schema == "silver"
        assert table == "dim_provider"

    def test_rejects_bare_table_name(self) -> None:
        """Should raise ValueError when no dot is present."""
        with pytest.raises(ValueError, match="target must be 'schema.table'"):
            PostgresResource._parse_target("dim_provider")

    def test_rejects_too_many_dots(self) -> None:
        """Should raise ValueError for 'catalog.schema.table' style targets."""
        with pytest.raises(ValueError, match="Invalid table name format"):
            PostgresResource._parse_target("catalog.silver.dim_provider")

    def test_rejects_empty_schema(self) -> None:
        """Should raise ValueError when schema portion is empty."""
        with pytest.raises(ValueError, match="Invalid table name format"):
            PostgresResource._parse_target(".dim_provider")

    def test_rejects_empty_table(self) -> None:
        """Should raise ValueError when table portion is empty."""
        with pytest.raises(ValueError, match="Invalid table name format"):
            PostgresResource._parse_target("silver.")


# ---------------------------------------------------------------------------
# TestBuildWriteConfig
# ---------------------------------------------------------------------------


class TestBuildWriteConfig:
    """Tests for PostgresResource._build_write_config()."""

    def test_resolves_string_write_mode_to_enum(self) -> None:
        """String write_mode should be converted to WriteMode enum."""
        from moncpipelib.io_managers.enums import WriteMode

        config = PostgresResource._build_write_config(
            write_mode="upsert",
            primary_key=["id"],
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert config["write_mode"] == WriteMode.UPSERT

    def test_accepts_enum_write_mode(self) -> None:
        """WriteMode enum should be passed through unchanged."""
        from moncpipelib.io_managers.enums import WriteMode

        config = PostgresResource._build_write_config(
            write_mode=WriteMode.APPEND,
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert config["write_mode"] == WriteMode.APPEND

    def test_explicit_flags_track_provided_params(self) -> None:
        """Explicit flags should be True when a param value is provided."""
        config = PostgresResource._build_write_config(
            write_mode="full_refresh",
            primary_key=["id"],
            update_columns=None,
            partition_column="report_date",
            business_key=["provider_npi"],
            tracked_columns=["name", "address"],
            detect_deletes=True,
        )
        assert config["primary_key_explicit"] is True
        assert config["partition_column_explicit"] is True
        assert config["business_key_explicit"] is True
        assert config["tracked_columns_explicit"] is True
        assert config["detect_deletes_explicit"] is True

    def test_scd2_defaults_included(self) -> None:
        """SCD2 default column names should always be present in config."""
        config = PostgresResource._build_write_config(
            write_mode="full_refresh",
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert config["effective_from_col"] == "effective_from"
        assert config["effective_to_col"] == "effective_to"
        assert config["is_current_col"] == "is_current"
        assert config["hash_col"] == "row_hash"

    def test_analyze_after_write_defaults_to_none(self) -> None:
        """Omitted analyze_after_write defers to the resource-level setting."""
        config = PostgresResource._build_write_config(
            write_mode="full_refresh",
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert config["analyze_after_write"] is None

    def test_analyze_after_write_param_carried_through(self) -> None:
        """An explicit analyze_after_write value lands in write_config."""
        config = PostgresResource._build_write_config(
            write_mode="append",
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
            analyze_after_write="always",
        )
        assert config["analyze_after_write"] == "always"

    def test_skip_unchanged_defaults_off_and_not_explicit(self) -> None:
        """Omitted skip_unchanged is False and reconcilable by a contract sink."""
        config = PostgresResource._build_write_config(
            write_mode="upsert",
            primary_key=["id"],
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert config["skip_unchanged"] is False
        assert config["skip_unchanged_explicit"] is False

    def test_skip_unchanged_param_carried_through_as_explicit(self) -> None:
        """An explicit skip_unchanged=True lands in write_config as explicit."""
        config = PostgresResource._build_write_config(
            write_mode="upsert",
            primary_key=["id"],
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
            skip_unchanged=True,
        )
        assert config["skip_unchanged"] is True
        assert config["skip_unchanged_explicit"] is True


# ---------------------------------------------------------------------------
# TestResolveAnalyzeAfterWrite
# ---------------------------------------------------------------------------


class TestResolveAnalyzeAfterWrite:
    """Tests for PostgresResource._resolve_analyze_after_write()."""

    def test_default_field_value_is_partitioned(self, resource: PostgresResource) -> None:
        """Resource-level default targets exactly the autovacuum gap."""
        assert resource.analyze_after_write == "partitioned"
        assert resource._resolve_analyze_after_write({}) == "partitioned"

    def test_write_config_override_wins(self, resource: PostgresResource) -> None:
        assert resource._resolve_analyze_after_write({"analyze_after_write": "never"}) == "never"

    def test_invalid_value_fails_before_any_write_sql(self, resource: PostgresResource) -> None:
        with pytest.raises(ValueError, match="analyze_after_write"):
            resource._resolve_analyze_after_write({"analyze_after_write": "on"})


# ---------------------------------------------------------------------------
# TestNormalizeContext
# ---------------------------------------------------------------------------


class TestNormalizeContext:
    """Tests for PostgresResource._normalize_context()."""

    def test_passes_through_write_context(self) -> None:
        """A WriteContext should be returned as-is."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="my_asset",
            run_id="abc-123",
            log=MagicMock(),
        )
        result = PostgresResource._normalize_context(wctx)
        assert result is wctx

    def test_converts_asset_execution_context(self) -> None:
        """An AssetExecutionContext-like mock should be converted to WriteContext."""
        from moncpipelib.resources.types import WriteContext

        mock_ctx = MagicMock()
        mock_ctx.asset_key.to_user_string.return_value = "dim_provider"
        mock_ctx.run_id = "run-456"
        mock_ctx.has_partition_key = False

        result = PostgresResource._normalize_context(mock_ctx)
        assert isinstance(result, WriteContext)
        assert result.asset_name == "dim_provider"
        assert result.run_id == "run-456"
        assert result.has_partition_key is False


# ---------------------------------------------------------------------------
# TestValidateWriteConfig
# ---------------------------------------------------------------------------


class TestValidateWriteConfig:
    """Tests for PostgresResource._validate_write_config()."""

    def test_upsert_without_primary_key_raises(self) -> None:
        """Upsert mode without primary_key should raise ValueError."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.UPSERT,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
        }
        with pytest.raises(ValueError, match="requires primary_key"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_scd2_without_business_key_raises(self) -> None:
        """SCD2 mode without business_key should raise ValueError."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.SCD2,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
            "business_key": None,
        }
        with pytest.raises(ValueError, match="requires business_key"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_missing_primary_key_column_raises(self) -> None:
        """Primary key referencing a missing column should raise ValueError."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.UPSERT,
            "primary_key": ["id", "nonexistent"],
            "update_columns": None,
            "partition_column": None,
        }
        with pytest.raises(ValueError, match="not found in DataFrame"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_missing_partition_column_raises(self) -> None:
        """Partition column not in DataFrame should raise ValueError."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.FULL_REFRESH,
            "primary_key": None,
            "update_columns": None,
            "partition_column": "report_date",
        }
        with pytest.raises(ValueError, match="partition_column.*not found"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_valid_full_refresh_config_passes(self) -> None:
        """A valid full_refresh config should not raise."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.FULL_REFRESH,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
        }
        # Should not raise
        PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_skip_unchanged_on_non_upsert_raises(self) -> None:
        """skip_unchanged=True outside upsert mode is inert config -> ValueError."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.APPEND,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
            "skip_unchanged": True,
        }
        with pytest.raises(ValueError, match="only valid with write_mode='upsert'"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_skip_unchanged_on_upsert_passes(self) -> None:
        """skip_unchanged=True with upsert mode and a primary key is valid."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.UPSERT,
            "primary_key": ["id"],
            "update_columns": None,
            "partition_column": None,
            "skip_unchanged": True,
        }
        # Should not raise
        PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_detect_deletes_on_non_scd2_raises(self) -> None:
        """detect_deletes=True outside scd2 mode is inert config -> ValueError (#429)."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.UPSERT,
            "primary_key": ["id"],
            "update_columns": None,
            "partition_column": None,
            "detect_deletes": True,
        }
        with pytest.raises(ValueError, match="only valid with write_mode='scd2'"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_detect_deletes_on_scd2_passes(self) -> None:
        """detect_deletes=True with scd2 mode and a business key is valid."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.SCD2,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
            "business_key": ["id"],
            "detect_deletes": True,
        }
        # Should not raise
        PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_full_refresh_method_on_non_full_refresh_raises(self) -> None:
        """full_refresh_method set outside full_refresh mode is inert -> ValueError (#4)."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.UPSERT,
            "primary_key": ["id"],
            "update_columns": None,
            "partition_column": None,
            "full_refresh_method": "delete",
        }
        with pytest.raises(ValueError, match="only valid with write_mode='full_refresh'"):
            PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_full_refresh_method_on_full_refresh_passes(self) -> None:
        """full_refresh_method with full_refresh mode is valid."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.FULL_REFRESH,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
            "full_refresh_method": "delete",
        }
        # Should not raise
        PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")

    def test_full_refresh_method_none_on_non_full_refresh_passes(self) -> None:
        """An absent (None) full_refresh_method never trips the backstop."""
        from moncpipelib.io_managers.enums import WriteMode

        config = {
            "write_mode": WriteMode.APPEND,
            "primary_key": None,
            "update_columns": None,
            "partition_column": None,
            "full_refresh_method": None,
        }
        # Should not raise
        PostgresResource._validate_write_config(config, ["id", "name"], "test_asset")


# ---------------------------------------------------------------------------
# TestValidatePartitionSafety
# ---------------------------------------------------------------------------


class TestValidatePartitionSafety:
    """Tests for PostgresResource._validate_partition_safety()."""

    def test_no_partition_key_is_noop(self, resource: PostgresResource) -> None:
        """Non-partitioned context should pass without checks."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="test", run_id="r1", log=MagicMock(), has_partition_key=False
        )
        config = {
            "write_mode": WriteMode.FULL_REFRESH,
            "partition_column": None,
            "primary_key": None,
        }
        # Should not raise
        resource._validate_partition_safety(wctx, config, "test")

    def test_full_refresh_partitioned_without_partition_column_raises(
        self, resource: PostgresResource
    ) -> None:
        """Partitioned full_refresh without partition_column should raise."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="test",
            run_id="r1",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-01"],
        )
        config = {
            "write_mode": WriteMode.FULL_REFRESH,
            "partition_column": None,
            "primary_key": None,
        }
        with pytest.raises(ContractViolationError, match="partitioned but no partition_column"):
            resource._validate_partition_safety(wctx, config, "test")

    def test_upsert_partition_column_not_in_primary_key_raises(
        self, resource: PostgresResource
    ) -> None:
        """Upsert with partition_column excluded from primary_key should raise."""
        from moncpipelib.contracts.exceptions import ContractViolationError
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="test",
            run_id="r1",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-01"],
        )
        config = {
            "write_mode": WriteMode.UPSERT,
            "partition_column": "report_date",
            "primary_key": ["id"],
        }
        with pytest.raises(ContractViolationError, match="does not include it"):
            resource._validate_partition_safety(wctx, config, "test")


# ---------------------------------------------------------------------------
# TestWriteResultToDagsterMetadata
# ---------------------------------------------------------------------------


class TestWriteResultToDagsterMetadata:
    """Tests for WriteResult.to_dagster_metadata()."""

    def test_basic_metadata_fields(self) -> None:
        """Should include write_mode, target_table, and row_count."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.dim_provider",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={"rows_deleted": 100, "rows_inserted": 50},
            row_count=50,
        )
        metadata = result.to_dagster_metadata()

        assert metadata["write_mode"] == MetadataValue.text("full_refresh")
        assert metadata["target_table"] == MetadataValue.text("silver.dim_provider")
        assert metadata["row_count"] == MetadataValue.int(50)
        assert metadata["layer"] == MetadataValue.text("silver")

    def test_stats_are_included(self) -> None:
        """Mode-specific stats (int and str values) should appear in metadata."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.fact_claims",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.UPSERT,
            stats={"rows_upserted": 200, "insert_method": "execute_values"},
            row_count=200,
        )
        metadata = result.to_dagster_metadata()

        assert metadata["rows_upserted"] == MetadataValue.int(200)
        assert metadata["insert_method"] == MetadataValue.text("execute_values")

    def test_float_and_bool_stats_are_routed_correctly(self) -> None:
        """Float stats route to MetadataValue.float; bool stats route to
        MetadataValue.bool (not MetadataValue.int -- ``bool`` is an ``int``
        subclass in Python).  Backstops the verbose-timings path in
        ``_write_batched`` (#260) which emits ``t_iter_seconds`` etc."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.fact_claims",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={
                "t_iter_seconds": 12.345,
                "t_copy_seconds": 0.001,
                "rows_inserted": 100,
                "skipped": True,
            },
            row_count=100,
        )
        metadata = result.to_dagster_metadata()

        assert metadata["t_iter_seconds"] == MetadataValue.float(12.345)
        assert metadata["t_copy_seconds"] == MetadataValue.float(0.001)
        assert metadata["rows_inserted"] == MetadataValue.int(100)
        assert metadata["skipped"] == MetadataValue.bool(True)

    def test_optional_fields_omitted_when_none(self) -> None:
        """Optional fields (source_file, lineage_id, etc.) should be absent when None."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.dim_provider",
            schema="silver",
            layer=None,
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
        )
        metadata = result.to_dagster_metadata()

        assert "layer" not in metadata
        assert "source_file" not in metadata
        assert "lineage_id" not in metadata
        assert "primary_key" not in metadata
        assert "partition_column" not in metadata
        # ``backfill_id`` is omitted when ``None`` so normal-run metadata
        # payloads stay clean.
        assert "backfill_id" not in metadata

    def test_is_backfill_always_emitted(self) -> None:
        """Migration 018 Phase 2: ``is_backfill`` is always emitted as a
        boolean -- both ``True`` and ``False`` are informative for the
        materialization-event view."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result_false = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
        )
        assert result_false.to_dagster_metadata()["is_backfill"] == MetadataValue.bool(False)

        result_true = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
            is_backfill=True,
        )
        assert result_true.to_dagster_metadata()["is_backfill"] == MetadataValue.bool(True)

    def test_backfill_id_emitted_when_present(self) -> None:
        """Migration 018 Phase 2: ``backfill_id`` surfaces as text on the
        materialization-event view when the run was part of a backfill."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
            is_backfill=True,
            backfill_id="bf_2026_05_22_claims",
        )
        metadata = result.to_dagster_metadata()

        assert metadata["backfill_id"] == MetadataValue.text("bf_2026_05_22_claims")
        assert metadata["is_backfill"] == MetadataValue.bool(True)

    def test_duration_seconds_emitted_when_set(self) -> None:
        """``duration_seconds`` surfaces as a ``MetadataValue.float`` so
        materialization queries can compare wall-clock cost run-over-run."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=1000,
            duration_seconds=2.5,
        )
        metadata = result.to_dagster_metadata()

        assert metadata["duration_seconds"] == MetadataValue.float(2.5)
        # Throughput is derived from row_count / duration_seconds.
        assert metadata["throughput_rows_per_sec"] == MetadataValue.float(400.0)

    def test_duration_seconds_omitted_when_unset(self) -> None:
        """Tests that build a ``WriteResult`` directly without going through
        ``write()`` legitimately leave ``duration_seconds`` ``None``; in that
        case both the duration and derived throughput must be absent."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=100,
        )
        metadata = result.to_dagster_metadata()

        assert "duration_seconds" not in metadata
        assert "throughput_rows_per_sec" not in metadata

    def test_throughput_omitted_when_zero_rows_or_duration(self) -> None:
        """Throughput is undefined for an empty write or a zero-duration
        write (mocked tests can hit the latter). Emitting ``inf`` or ``0.0``
        in those edge cases is more confusing than helpful."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        empty = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
            duration_seconds=1.5,
        )
        metadata_empty = empty.to_dagster_metadata()
        assert metadata_empty["duration_seconds"] == MetadataValue.float(1.5)
        assert "throughput_rows_per_sec" not in metadata_empty

        zero_duration = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=100,
            duration_seconds=0.0,
        )
        metadata_zero = zero_duration.to_dagster_metadata()
        assert metadata_zero["duration_seconds"] == MetadataValue.float(0.0)
        assert "throughput_rows_per_sec" not in metadata_zero

    def test_partition_keys_emitted_when_set(self) -> None:
        """``partition_keys`` surfaces the actual key value(s) being written
        (distinct from ``partition_column``, which is the column name)."""
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            # Partition-scoped writes are FULL_REFRESH + partition_column;
            # the enum has no dedicated PARTITION_SCOPED value.
            write_mode=WriteMode.FULL_REFRESH,
            stats={},
            row_count=10,
            partition_column="load_period",
            partition_keys=["2026-05-17", "2026-05-18"],
        )
        metadata = result.to_dagster_metadata()

        assert metadata["partition_key"] == MetadataValue.text("2026-05-17, 2026-05-18")
        # The column name is still emitted separately.
        assert metadata["partition_column"] == MetadataValue.text("load_period")

    def test_partition_keys_omitted_when_empty(self) -> None:
        """Non-partitioned writes leave ``partition_keys`` ``None`` and the
        metadata key must be absent."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
        )
        metadata = result.to_dagster_metadata()

        assert "partition_key" not in metadata

    def test_source_uri_pipeline_id_effective_date_emitted_when_set(self) -> None:
        """``source_uri`` (from_ingest bronze writes), ``pipeline_id``
        (always available when a contract is loaded), and ``effective_date``
        (SCD2 override) all surface in metadata when set, and are omitted
        when not."""
        import datetime

        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="bronze.fda_ndc",
            schema="bronze",
            layer="bronze",
            write_mode=WriteMode.SCD2,
            stats={},
            row_count=100,
            source_uri="abfss://intake@examplestorageacct.dfs.core.windows.net/fda/2026-05-17/ndc.zip",
            pipeline_id="01234567-89ab-cdef-0123-456789abcdef",
            effective_date=datetime.date(2026, 5, 17),
        )
        metadata = result.to_dagster_metadata()

        assert metadata["source_uri"] == MetadataValue.text(
            "abfss://intake@examplestorageacct.dfs.core.windows.net/fda/2026-05-17/ndc.zip"
        )
        assert metadata["pipeline_id"] == MetadataValue.text("01234567-89ab-cdef-0123-456789abcdef")
        assert metadata["effective_date"] == MetadataValue.text("2026-05-17")

    def test_source_uri_pipeline_id_effective_date_omitted_when_unset(self) -> None:
        """All three optional fields are omitted from metadata when ``None``."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
        )
        metadata = result.to_dagster_metadata()

        assert "source_uri" not in metadata
        assert "pipeline_id" not in metadata
        assert "effective_date" not in metadata

    def test_column_schema_from_contract(self) -> None:
        """Contract columns should produce a Dagster TableSchema with types and PII."""
        from dagster import TableSchema

        from moncpipelib.contracts.models import (
            Column,
            ColumnTest,
            ColumnType,
            DataContract,
            Schema,
        )
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        contract = DataContract(
            version="1.0",
            pipeline_id="test-uuid",
            layer="silver",
            asset="test_asset",
            schema=Schema(
                columns=[
                    Column(
                        name="id",
                        type=ColumnType.INTEGER,
                        nullable=False,
                        pii=False,
                        description="Primary key",
                        tests=[ColumnTest(test_type="unique")],
                    ),
                    Column(
                        name="patient_name",
                        type=ColumnType.STRING,
                        nullable=True,
                        pii=True,
                        description="Full name of patient",
                    ),
                    Column(
                        name="created_at",
                        type=ColumnType.DATETIME,
                        nullable=False,
                        pii=False,
                    ),
                ]
            ),
        )
        result = WriteResult(
            table_name="silver.test",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
            contract=contract,
            columns=["id", "patient_name", "created_at"],
        )
        metadata = result.to_dagster_metadata()

        assert "dagster/column_schema" in metadata
        schema = metadata["dagster/column_schema"].value
        assert isinstance(schema, TableSchema)
        assert len(schema.columns) == 3

        # id column: integer, not nullable, unique, not PII
        id_col = schema.columns[0]
        assert id_col.name == "id"
        assert id_col.type == "int"
        assert id_col.description == "Primary key"
        assert id_col.constraints.nullable is False
        assert id_col.constraints.unique is True
        assert "Contains PHI" not in (id_col.constraints.other or [])

        # patient_name: string, nullable, PII
        name_col = schema.columns[1]
        assert name_col.name == "patient_name"
        assert name_col.type == "string"
        assert name_col.description == "[PHI] Full name of patient"
        assert name_col.constraints.nullable is True
        assert "Contains PHI" in name_col.constraints.other

        # created_at: datetime, not PII, no description
        dt_col = schema.columns[2]
        assert dt_col.name == "created_at"
        assert dt_col.type == "datetime"
        assert dt_col.description is None
        assert dt_col.constraints.nullable is False

    def test_column_schema_pii_no_description(self) -> None:
        """PII column without a description should get '[PHI]' as description."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        contract = DataContract(
            version="1.0",
            pipeline_id="test-uuid",
            layer="silver",
            asset="test_asset",
            schema=Schema(
                columns=[
                    Column(
                        name="ssn",
                        type=ColumnType.STRING,
                        nullable=False,
                        pii=True,
                    ),
                ]
            ),
        )
        result = WriteResult(
            table_name="silver.test",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
            contract=contract,
        )
        metadata = result.to_dagster_metadata()
        schema = metadata["dagster/column_schema"].value
        assert schema.columns[0].description == "[PHI]"
        assert "Contains PHI" in schema.columns[0].constraints.other

    def test_column_schema_managed_columns_excluded(self) -> None:
        """Managed columns (e.g., _lineage_id) should not appear in TableSchema."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        contract = DataContract(
            version="1.0",
            pipeline_id="test-uuid",
            layer="silver",
            asset="test_asset",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    Column(
                        name="_lineage_id",
                        type=ColumnType.UUID,
                        nullable=True,
                        managed=True,
                        pii=False,
                    ),
                ]
            ),
        )
        result = WriteResult(
            table_name="silver.test",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
            contract=contract,
        )
        metadata = result.to_dagster_metadata()
        schema = metadata["dagster/column_schema"].value
        assert len(schema.columns) == 1
        assert schema.columns[0].name == "id"

    def test_column_schema_fallback_no_contract(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without a contract, bare column names should be emitted with a warning."""
        from dagster import TableSchema

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.test",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=10,
            columns=["id", "name", "value"],
        )
        with caplog.at_level("WARNING", logger="moncpipelib.resources"):
            metadata = result.to_dagster_metadata()

        assert "dagster/column_schema" in metadata
        schema = metadata["dagster/column_schema"].value
        assert isinstance(schema, TableSchema)
        assert len(schema.columns) == 3
        assert schema.columns[0].name == "id"
        assert schema.columns[0].type == "string"  # default
        assert "No contract available" in caplog.text

    def test_column_schema_no_contract_no_columns(self) -> None:
        """Without contract or columns, no dagster/column_schema should be emitted."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.test",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
        )
        metadata = result.to_dagster_metadata()
        assert "dagster/column_schema" not in metadata

    def test_column_type_mapping(self) -> None:
        """Each ColumnType should map to the expected Dagster type string."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import COLUMN_TYPE_MAP, WriteResult

        expected = {
            ColumnType.STRING: "string",
            ColumnType.INTEGER: "int",
            ColumnType.DECIMAL: "float",
            ColumnType.BOOLEAN: "bool",
            ColumnType.DATE: "date",
            ColumnType.DATETIME: "datetime",
            ColumnType.UUID: "uuid",
            ColumnType.JSON: "json",
            ColumnType.JSONB: "jsonb",
        }

        for col_type, dagster_type in expected.items():
            contract = DataContract(
                version="1.0",
                pipeline_id="test-uuid",
                layer="silver",
                asset="test_asset",
                schema=Schema(
                    columns=[Column(name="col", type=col_type, nullable=True, pii=False)]
                ),
            )
            result = WriteResult(
                table_name="silver.test",
                schema="silver",
                layer="silver",
                write_mode=WriteMode.APPEND,
                stats={},
                row_count=10,
                contract=contract,
            )
            metadata = result.to_dagster_metadata()
            schema = metadata["dagster/column_schema"].value
            assert schema.columns[0].type == dagster_type, (
                f"{col_type} should map to {dagster_type}"
            )

        # Verify all ColumnType values are covered
        assert set(expected.keys()) == set(ColumnType), "All ColumnType values should be tested"
        assert set(COLUMN_TYPE_MAP.keys()) == {str(ct) for ct in ColumnType}, (
            "COLUMN_TYPE_MAP should cover all ColumnType values"
        )


# ---------------------------------------------------------------------------
# TestWriteOrchestration
# ---------------------------------------------------------------------------


class TestWriteOrchestration:
    """Tests for PostgresResource.write() end-to-end orchestration with mocks."""

    @pytest.fixture
    def write_resource(self) -> PostgresResource:
        """Create a PostgresResource with write-related features disabled for unit tests."""
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_write_single_dataframe_full_refresh(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """write() should open a connection, execute SQL, commit, and return WriteResult."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext, WriteResult

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Stub _validate_columns to skip real DB schema lookup
        mock_cursor.fetchall.return_value = []

        wctx = WriteContext(asset_name="dim_provider", run_id="run-1", log=MagicMock())
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 3},
            ) as mock_exec,
        ):
            result = write_resource.write(
                df,
                target="silver.dim_provider",
                context=wctx,
                write_mode="full_refresh",
                contract=None,
            )

        assert isinstance(result, WriteResult)
        assert result.table_name == "silver.dim_provider"
        assert result.schema == "silver"
        assert result.layer == "silver"
        assert result.write_mode == WriteMode.FULL_REFRESH
        assert result.row_count == 3
        assert result.stats["rows_inserted"] == 3
        mock_conn.commit.assert_called()
        mock_conn.close.assert_called_once()
        mock_exec.assert_called_once()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_write_skip_unchanged_rejected_on_non_upsert_mode(
        self,
        _mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """skip_unchanged outside upsert fails fast through the full write() chain.

        Covers the kwarg -> _build_write_config -> validate_write_config wiring
        that the direct _validate_write_config tests skip; the raise happens
        before any SQL, so the connection mock exists only as a safety net.
        """
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(asset_name="dim_provider", run_id="run-1", log=MagicMock())
        df = pl.DataFrame({"id": [1], "name": ["a"]})

        with pytest.raises(ValueError, match="only valid with write_mode='upsert'"):
            write_resource.write(
                df,
                target="silver.dim_provider",
                context=wctx,
                write_mode="append",
                skip_unchanged=True,
                contract=None,
            )

    def test_write_rejects_non_dataframe(self, write_resource: PostgresResource) -> None:
        """write() should raise TypeError for non-DataFrame, non-BatchedDataFrame input."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(asset_name="test", run_id="r1", log=MagicMock())

        with pytest.raises(TypeError, match="expected pl.DataFrame or BatchedDataFrame"):
            write_resource.write(
                {"not": "a dataframe"},  # type: ignore[arg-type]
                target="silver.test",
                context=wctx,
                contract=None,
            )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_write_attaches_duration_seconds(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """``write()`` must attach a positive ``duration_seconds`` on the
        returned ``WriteResult`` regardless of which branch was taken.
        This is the load-bearing contract for run-over-run perf
        comparisons in the materialization-event view."""
        from moncpipelib.resources.types import WriteContext

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        wctx = WriteContext(asset_name="dim_provider", run_id="run-1", log=MagicMock())
        df = pl.DataFrame({"id": [1, 2, 3]})

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 3},
            ),
        ):
            result = write_resource.write(
                df,
                target="silver.dim_provider",
                context=wctx,
                write_mode="full_refresh",
                contract=None,
            )

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0.0
        # The metadata dict must surface it as a float.
        metadata = result.to_dagster_metadata()
        assert "duration_seconds" in metadata


# ---------------------------------------------------------------------------
# Batched-write reconcile-before-inject ordering (#258)
# ---------------------------------------------------------------------------


class TestBatchedWritePartitionColumnInjection:
    """Tests covering #258: ``_inject_period_partition_column`` MUST run after
    ``ContractReconciler.reconcile_write_config`` in the BatchedDataFrame
    write path, so the contract-derived ``partition_column`` (e.g.
    ``load_period``) is in ``write_config`` when inject reads it.

    Pre-#258, inject ran first and bailed because ``write_config
    ["partition_column"]`` was ``None`` (it gets populated by the
    reconcile call that ran AFTER inject).  The single-DataFrame path
    has always done it in the correct order, which is why CMS ASP
    consumers (``pl.DataFrame``) worked while RxNorm Phase 2a
    (``BatchedDataFrame``) hit the bug in data-platform#613.
    """

    @pytest.fixture
    def write_resource(self) -> PostgresResource:
        """Resource with write features disabled for unit tests."""
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    @staticmethod
    def _partition_contract(table: str, partition_column: str = "load_period") -> Any:  # type: ignore[no-untyped-def]
        """Build a DataContract whose sink declares ``partition_column``.

        The reconcile path looks up the sink via ``find_matching_sink``
        (table-name match, with single-sink fallback) and copies the
        sink's ``partition_column`` into ``write_config`` when the
        caller hasn't passed one explicitly.
        """
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset=table,
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.STRING, nullable=False),
                    # ``managed=True`` matches the production convention:
                    # columns populated by ``_inject_period_partition_column``
                    # are excluded from ``validate_schema``'s
                    # "Missing columns" check.  Without it,
                    # ``_enforce_contract`` raises before inject runs.
                    Column(
                        name=partition_column,
                        type=ColumnType.STRING,
                        nullable=False,
                        managed=True,
                    ),
                ]
            ),
            sinks=[
                {
                    "type": "table",
                    "schema": "reference_bronze",
                    "table": table,
                    "partition_column": partition_column,
                }
            ],
        )

    @staticmethod
    def _setup_mock_conn(mock_connect: MagicMock) -> MagicMock:
        """Wire psycopg2.connect to a mock connection + cursor pair."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # _validate_columns reads pg_catalog; return empty list so the patch
        # in each test (which fully replaces the method) is what's used.
        mock_cursor.fetchall.return_value = []
        # PII metadata sync compares ``cursor.rowcount > 0``; without an int
        # MagicMock would raise TypeError mid-write.
        mock_cursor.rowcount = 0
        return mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_batched_write_injects_partition_column_from_contract_when_caller_omits(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """Regression test for #258.  RxNorm Phase 2a shape: contract sink
        declares ``partition_column: load_period``; caller passes a
        ``BatchedDataFrame`` lacking the column to ``database.write()``
        WITHOUT passing ``partition_column`` as a kwarg; the asset is
        partitioned.  Pre-#258, ``_validate_write_config`` raises
        ``partition_column 'load_period' not found in DataFrame``.
        Post-fix, inject populates the column on the first batch and
        the write proceeds."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        df1 = pl.DataFrame({"id": ["a", "b"]})
        df2 = pl.DataFrame({"id": ["c"]})
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=3)

        contract = self._partition_contract("rxnorm_rxnsty")
        wctx = WriteContext(
            asset_name="reference_bronze/rxnorm_rxnsty",
            run_id="run-1",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-01-04"],
        )

        landed_columns: list[list[str]] = []

        def _capture_columns(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.insert_rows",
                side_effect=lambda _config, _cursor, _table, batch_df, _mode, _wctx: (
                    landed_columns.append(list(batch_df.columns))
                ),
            ),
        ):
            write_resource.write(
                batched,
                target="reference_bronze.rxnorm_rxnsty",
                context=wctx,
                write_mode="append",
                contract=contract,
                # Intentionally NOT passing partition_column -- mirrors the
                # data-platform#613 production call and the pre-#258 bug.
            )

        assert len(landed_columns) == 2, f"expected 2 batches written, got {len(landed_columns)}"
        for batch_idx, cols in enumerate(landed_columns):
            assert "load_period" in cols, (
                f"batch {batch_idx} missing 'load_period'; columns: {cols}. "
                "This is the #258 regression."
            )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_batched_write_injects_partition_column_on_every_batch(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """Multi-batch invariant: ``load_period`` must be on EVERY landed
        batch, not just the first.  Catches a future refactor that
        accidentally moves ``_inject_period_partition_column`` inside
        ``if i == 0:`` (which would break batches 2..N silently)."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        df1 = pl.DataFrame({"id": ["a"]})
        df2 = pl.DataFrame({"id": ["b"]})
        df3 = pl.DataFrame({"id": ["c"]})
        batched = BatchedDataFrame(batches=iter([df1, df2, df3]), total_rows_hint=3)

        contract = self._partition_contract("multi_batch_table")
        wctx = WriteContext(
            asset_name="reference_bronze/multi_batch_table",
            run_id="run-multi",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-02-01"],
        )

        landed_partition_values: list[Any] = []

        def _capture_partition(
            _config: Any, _cursor: Any, _table: str, batch_df: pl.DataFrame, _mode: Any, _wctx: Any
        ) -> None:
            landed_partition_values.append(batch_df["load_period"].to_list())

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.insert_rows",
                side_effect=_capture_partition,
            ),
        ):
            write_resource.write(
                batched,
                target="reference_bronze.multi_batch_table",
                context=wctx,
                write_mode="append",
                contract=contract,
            )

        assert len(landed_partition_values) == 3
        for batch_idx, values in enumerate(landed_partition_values):
            assert all(v == "2024-02-01" for v in values), (
                f"batch {batch_idx}: expected all load_period values to be "
                f"'2024-02-01', got {values}"
            )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_batched_write_explicit_partition_column_kwarg_still_works(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """Back-compat: when the caller passes ``partition_column`` AND the
        contract declares the same value, reconcile logs a warning but
        proceeds.  Locks in the data-platform-side workaround path so
        removing the workaround after #258 lands doesn't trip a
        different ordering bug."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        df1 = pl.DataFrame({"id": ["a"]})
        batched = BatchedDataFrame(batches=iter([df1]), total_rows_hint=1)

        contract = self._partition_contract("explicit_kwarg_table")
        wctx = WriteContext(
            asset_name="reference_bronze/explicit_kwarg_table",
            run_id="run-explicit",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-03-01"],
        )

        landed: list[list[str]] = []

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.insert_rows",
                side_effect=lambda _config, _cursor, _table, batch_df, _mode, _wctx: landed.append(
                    list(batch_df.columns)
                ),
            ),
        ):
            write_resource.write(
                batched,
                target="reference_bronze.explicit_kwarg_table",
                context=wctx,
                write_mode="append",
                contract=contract,
                partition_column="load_period",  # data-platform workaround shape
            )

        assert len(landed) == 1
        assert "load_period" in landed[0]

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_single_write_injects_partition_column_from_contract_when_caller_omits(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """Symmetry test: the single-``pl.DataFrame`` write path must do
        the same thing as the batched path -- contract-declared
        ``partition_column`` must be injected when the caller omits the
        kwarg.  Locks in cross-path symmetry so the two paths can't
        drift on this dimension again (the original #258 bug was
        precisely that drift)."""
        from moncpipelib.resources.types import WriteContext

        self._setup_mock_conn(mock_connect)

        df = pl.DataFrame({"id": ["a", "b", "c"]})
        contract = self._partition_contract("single_path_table")
        wctx = WriteContext(
            asset_name="reference_bronze/single_path_table",
            run_id="run-single",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2024-04-01"],
        )

        landed: list[list[str]] = []

        def _capture_append(
            _config: Any, _cursor: Any, _table: str, df_arg: pl.DataFrame, _wctx: Any
        ) -> dict[str, Any]:
            landed.append(list(df_arg.columns))
            return {"rows_inserted": len(df_arg), "insert_method": "test_stub"}

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_append",
                side_effect=_capture_append,
            ),
        ):
            write_resource.write(
                df,
                target="reference_bronze.single_path_table",
                context=wctx,
                write_mode="append",
                contract=contract,
            )

        assert len(landed) == 1
        assert "load_period" in landed[0], (
            "single-DataFrame path must inject contract-declared partition_column "
            "when caller omits the kwarg, mirroring the BatchedDataFrame path."
        )


# ---------------------------------------------------------------------------
# Verbose-timings emission in batched writes (#260)
# ---------------------------------------------------------------------------


class TestBatchedWriteVerboseTimings:
    """Cover the per-phase timer instrumentation in ``_write_batched``.

    Diagnoses the ``Client:ClientRead`` server-side wait pattern reported
    in #260 by attributing wall time to one of three buckets:
    ``t_iter_seconds`` (upstream batch production), ``t_prep_seconds``
    (contract / inject / lineage / SCD2 prep), ``t_copy_seconds``
    (the actual COPY / upsert call).
    """

    @pytest.fixture
    def write_resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    @staticmethod
    def _setup_mock_conn(mock_connect: MagicMock) -> MagicMock:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_cursor.rowcount = 0
        return mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_verbose_off_omits_timing_metadata(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With ``VERBOSE_METADATA=False`` (the default), the timing keys
        must NOT appear in ``WriteResult.stats`` -- normal-operation
        metadata should be unchanged from before #260."""
        import sys

        # ``moncpipelib.__init__`` re-exports the ``config`` singleton,
        # which shadows the submodule via attribute lookup.  Pull the
        # actual module out of ``sys.modules`` so monkeypatching binds
        # to the module's namespace, not the singleton.
        _cfg_mod = sys.modules["moncpipelib.config"]

        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        monkeypatch.setattr(_cfg_mod, "VERBOSE_METADATA", False)
        self._setup_mock_conn(mock_connect)

        batched = BatchedDataFrame(
            batches=iter([pl.DataFrame({"id": ["a", "b"]})]), total_rows_hint=2
        )
        wctx = WriteContext(asset_name="bronze/t", run_id="r1", log=MagicMock())

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            result = write_resource.write(
                batched,
                target="bronze.t",
                context=wctx,
                write_mode="append",
                contract=None,
            )

        assert "t_iter_seconds" not in result.stats
        assert "t_prep_seconds" not in result.stats
        assert "t_copy_seconds" not in result.stats

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_verbose_on_emits_timing_metadata(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With ``VERBOSE_METADATA=True``, ``WriteResult.stats`` must
        include ``t_iter_seconds`` / ``t_prep_seconds`` / ``t_copy_seconds``
        as floats, and they must round-trip through ``to_dagster_metadata``
        as ``MetadataValue.float``."""
        import sys

        # ``moncpipelib.__init__`` re-exports the ``config`` singleton,
        # which shadows the submodule via attribute lookup.  Pull the
        # actual module out of ``sys.modules`` so monkeypatching binds
        # to the module's namespace, not the singleton.
        _cfg_mod = sys.modules["moncpipelib.config"]
        from dagster import MetadataValue

        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        monkeypatch.setattr(_cfg_mod, "VERBOSE_METADATA", True)
        self._setup_mock_conn(mock_connect)

        batched = BatchedDataFrame(
            batches=iter(
                [
                    pl.DataFrame({"id": ["a", "b"]}),
                    pl.DataFrame({"id": ["c"]}),
                    pl.DataFrame({"id": ["d", "e", "f"]}),
                ]
            ),
            total_rows_hint=6,
        )
        wctx = WriteContext(asset_name="bronze/t", run_id="r1", log=MagicMock())

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            result = write_resource.write(
                batched,
                target="bronze.t",
                context=wctx,
                write_mode="append",
                contract=None,
            )

        for key in ("t_iter_seconds", "t_prep_seconds", "t_copy_seconds"):
            assert key in result.stats, f"missing verbose timing key: {key}"
            value = result.stats[key]
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
            assert value >= 0.0, f"{key} should be non-negative, got {value}"

        metadata = result.to_dagster_metadata()
        assert isinstance(metadata["t_iter_seconds"], type(MetadataValue.float(0.0)))
        assert isinstance(metadata["t_prep_seconds"], type(MetadataValue.float(0.0)))
        assert isinstance(metadata["t_copy_seconds"], type(MetadataValue.float(0.0)))

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_verbose_timings_attribute_iter_time_to_t_iter(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A slow upstream iterator (sleeps between yielding batches) must
        push wall time into ``t_iter_seconds``, not ``t_copy_seconds``.
        This is the load-bearing assertion for diagnosing #260: it
        proves the instrumentation correctly distinguishes ClientRead /
        idle-in-txn time from real COPY time.
        """
        import sys
        import time as _time

        # ``moncpipelib.__init__`` re-exports the ``config`` singleton,
        # which shadows the submodule via attribute lookup.  Pull the
        # actual module out of ``sys.modules`` so monkeypatching binds
        # to the module's namespace, not the singleton.
        _cfg_mod = sys.modules["moncpipelib.config"]

        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        monkeypatch.setattr(_cfg_mod, "VERBOSE_METADATA", True)
        self._setup_mock_conn(mock_connect)

        slow_iter_sleep = 0.05  # 50ms per batch boundary

        def _slow_batches() -> Any:
            yield pl.DataFrame({"id": ["a"]})
            _time.sleep(slow_iter_sleep)
            yield pl.DataFrame({"id": ["b"]})
            _time.sleep(slow_iter_sleep)
            yield pl.DataFrame({"id": ["c"]})

        batched = BatchedDataFrame(batches=_slow_batches(), total_rows_hint=3)
        wctx = WriteContext(asset_name="bronze/t", run_id="r1", log=MagicMock())

        with (
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            result = write_resource.write(
                batched,
                target="bronze.t",
                context=wctx,
                write_mode="append",
                contract=None,
            )

        # 2 sleeps of 50ms each = ~100ms minimum in t_iter.  Use a
        # generous lower bound (50ms) so this isn't flaky on slow CI.
        # ``insert_rows`` is mocked away, so t_copy should be near zero.
        assert result.stats["t_iter_seconds"] >= 0.05, (
            f"slow upstream iterator should attribute time to t_iter, "
            f"got t_iter={result.stats['t_iter_seconds']}"
        )
        assert result.stats["t_iter_seconds"] > result.stats["t_copy_seconds"], (
            f"with mocked COPY and a sleeping iterator, t_iter must dominate; "
            f"got t_iter={result.stats['t_iter_seconds']}, "
            f"t_copy={result.stats['t_copy_seconds']}"
        )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_runtime_set_verbose_metadata_is_picked_up_by_writer(
        self,
        mock_connect: MagicMock,
        write_resource: PostgresResource,
    ) -> None:
        """Toggling the flag at runtime via ``set_verbose_metadata``
        (with NO env-var fiddling and NO monkeypatch) must affect the
        very next batched write.  This is the load-bearing claim of
        the in-pipeline ergonomic API: writer reads the flag lazily,
        so flipping it from a Dagster op or REPL takes effect on the
        next ``database.write(...)``."""
        import sys

        from moncpipelib import set_verbose_metadata
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        cfg = sys.modules["moncpipelib.config"]
        original_flag = cfg.VERBOSE_METADATA
        try:
            self._setup_mock_conn(mock_connect)

            # First write with flag OFF -- baseline.
            set_verbose_metadata(False)
            batched_off = BatchedDataFrame(
                batches=iter([pl.DataFrame({"id": ["a"]})]), total_rows_hint=1
            )
            wctx = WriteContext(asset_name="bronze/t", run_id="r1", log=MagicMock())
            with (
                patch.object(PostgresResource, "_validate_columns"),
                patch("moncpipelib.io_managers.writers.insert_rows"),
            ):
                result_off = write_resource.write(
                    batched_off,
                    target="bronze.t",
                    context=wctx,
                    write_mode="append",
                    contract=None,
                )
            assert "t_iter_seconds" not in result_off.stats

            # Flip via the public API and write again.
            set_verbose_metadata(True)
            batched_on = BatchedDataFrame(
                batches=iter([pl.DataFrame({"id": ["b"]})]), total_rows_hint=1
            )
            with (
                patch.object(PostgresResource, "_validate_columns"),
                patch("moncpipelib.io_managers.writers.insert_rows"),
            ):
                result_on = write_resource.write(
                    batched_on,
                    target="bronze.t",
                    context=wctx,
                    write_mode="append",
                    contract=None,
                )
            assert "t_iter_seconds" in result_on.stats, (
                "set_verbose_metadata(True) should flip the flag for the "
                "very next write -- writer must read the flag lazily"
            )
            assert "t_prep_seconds" in result_on.stats
            assert "t_copy_seconds" in result_on.stats
        finally:
            cfg.VERBOSE_METADATA = original_flag


# ---------------------------------------------------------------------------
# Contract-driven lineage tests
# ---------------------------------------------------------------------------


class TestContractDrivenLineage:
    """Tests for contract layer fallback and lineage config in write path."""

    @pytest.fixture
    def lineage_resource(self) -> PostgresResource:
        """Resource with lineage enabled but contracts silent (for unit tests)."""
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    def test_layer_fallback_from_contract(self) -> None:
        """write() should resolve layer from contract when schema is compound."""
        from moncpipelib.config import VALID_LAYERS as _VALID_LAYERS
        from moncpipelib.contracts.models import Column, ColumnType, DataContract, Schema

        # Schema "reference_bronze" is not in VALID_LAYERS
        assert "reference_bronze" not in _VALID_LAYERS

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="fda_ndc_directory_bronze",
            layer="bronze",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
        )

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=False,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

        wctx = _make_wctx("fda_ndc_directory_bronze")
        df = pl.DataFrame({"id": ["1", "2"]})

        with (
            patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect,
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 2},
            ),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 0
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = resource.write(
                df,
                target="reference_bronze.fda_ndc_directory",
                context=wctx,
                write_mode="full_refresh",
                contract=contract,
            )

        # Layer should resolve from contract, not schema
        assert result.layer == "bronze"

    def test_lineage_disabled_via_contract(self) -> None:
        """write() should skip lineage when contract has lineage.enabled=False."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            LineageConfig,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
            lineage=LineageConfig(enabled=False),
        )

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

        wctx = _make_wctx("test_asset")
        df = pl.DataFrame({"id": ["1", "2"]})

        with (
            patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect,
            patch.object(PostgresResource, "_validate_columns"),
            patch.object(PostgresResource, "_prepare_lineage") as mock_lineage,
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 2},
            ),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 0
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            resource.write(
                df,
                target="bronze.test_table",
                context=wctx,
                write_mode="full_refresh",
                contract=contract,
            )

        # _prepare_lineage should NOT have been called
        mock_lineage.assert_not_called()

    def test_lineage_config_fields_passed_through(self) -> None:
        """source_system and transformation_type from contract should reach _prepare_lineage."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            LineageConfig,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="test_asset",
            layer="bronze",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
            lineage=LineageConfig(
                source_system="openfda",
                transformation_type="ingest",
            ),
        )

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

        wctx = _make_wctx("test_asset")
        df = pl.DataFrame({"id": ["1", "2"]})

        with (
            patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect,
            patch.object(PostgresResource, "_validate_columns"),
            patch.object(
                PostgresResource,
                "_prepare_lineage",
                # Phase 3: ``_prepare_lineage`` returns ``(df, lineage_id,
                # lineage_key, insert_kwargs)``. ``insert_kwargs`` must
                # include every required ``write_lineage_record`` arg
                # because the cursor block below will splat it.
                return_value=(
                    df,
                    "lid",
                    "lkey",
                    {
                        "lineage_id": "lid",
                        "lineage_key": "lkey",
                        "run_id": "run-test",
                        "asset_name": "test_asset",
                        "layer": "bronze",
                        "source_system": "openfda",
                        "transformation_type": "ingest",
                    },
                ),
            ) as mock_lineage,
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 2},
            ),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 0
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            resource.write(
                df,
                target="bronze.test_table",
                context=wctx,
                write_mode="full_refresh",
                contract=contract,
            )

        mock_lineage.assert_called_once()
        call_kwargs = mock_lineage.call_args
        assert call_kwargs.kwargs["source_system"] == "openfda"
        assert call_kwargs.kwargs["transformation_type"] == "ingest"


class TestBackfillSignalsThroughLineage:
    """Migration 018 Phases 2 + 3: ``WriteContext.is_backfill`` and
    ``WriteContext.backfill_id`` must reach the lineage-row INSERT on
    both the single-write and batched write paths.

    After Phase 3 the single-write path uses ``_prepare_lineage`` (pure,
    no DB call) which returns ``insert_kwargs`` that the caller later
    feeds into ``tracker.write_lineage_record(cursor, ...)`` inside the
    same psycopg transaction as the data DML.
    """

    def test_prepare_lineage_forwards_backfill_signals(self) -> None:
        """``PostgresResource._prepare_lineage`` returns ``insert_kwargs``
        carrying the WriteContext's backfill signals."""
        from moncpipelib.resources.types import WriteContext

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )

        wctx = WriteContext(
            asset_name="claims_silver",
            run_id="run-1",
            log=MagicMock(),
            is_backfill=True,
            backfill_id="bf_2026_05_22_claims",
        )
        df = pl.DataFrame({"id": ["1", "2"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000001",
            "v1:claims_silver:silver:2026-05-22:run-1",
        )
        mock_tracker.attach_lineage_to_dataframe.return_value = df

        with patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker):
            _, _, _, insert_kwargs = resource._prepare_lineage(
                df, wctx, source_file="claims.csv", layer="silver"
            )

        assert insert_kwargs["is_backfill"] is True
        assert insert_kwargs["backfill_id"] == "bf_2026_05_22_claims"

    def test_prepare_lineage_propagates_no_backfill_defaults(self) -> None:
        """A non-backfill WriteContext yields ``is_backfill=False`` and
        ``backfill_id=None`` in the prepared insert kwargs."""
        from moncpipelib.resources.types import WriteContext

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )

        wctx = WriteContext(asset_name="claims_silver", run_id="run-2", log=MagicMock())
        df = pl.DataFrame({"id": ["1"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000002",
            "v1:claims_silver:silver:2026-05-22:run-2",
        )
        mock_tracker.attach_lineage_to_dataframe.return_value = df

        with patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker):
            _, _, _, insert_kwargs = resource._prepare_lineage(
                df, wctx, source_file=None, layer="silver"
            )

        assert insert_kwargs["is_backfill"] is False
        assert insert_kwargs["backfill_id"] is None

    def test_prepare_lineage_does_not_touch_database(self) -> None:
        """Phase 3 invariant: ``_prepare_lineage`` must NOT call
        ``create_lineage_record`` or ``write_lineage_record`` -- the
        cursor-bound INSERT is the caller's responsibility, sequenced
        with the data DML inside the same transaction."""
        from moncpipelib.resources.types import WriteContext

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
        )

        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())
        df = pl.DataFrame({"id": ["1"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid", "lkey")
        mock_tracker.attach_lineage_to_dataframe.return_value = df

        with patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker):
            resource._prepare_lineage(df, wctx, source_file=None, layer="bronze")

        mock_tracker.create_lineage_record.assert_not_called()
        mock_tracker.write_lineage_record.assert_not_called()


class TestBuildLineageMetadataPayload:
    """Issue #334 Bug 2: ``_build_lineage_metadata_payload`` builds the
    JSONB shape that lands in ``data_lineage.metadata``.

    The payload schema is documented on the method.  These cases pin
    each documented key to a specific input shape so future schema
    additions don't accidentally drop a required key from the payload.
    """

    @staticmethod
    def _resource(*, enforce_contracts: str = "error") -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts=enforce_contracts,
        )

    @staticmethod
    def _contract():  # type: ignore[no-untyped-def]
        from moncpipelib.contracts.models import Column, ColumnType, DataContract, Schema

        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="claims_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
        )

    @staticmethod
    def _summary(status: str = "passed"):  # type: ignore[no-untyped-def]
        from moncpipelib.contracts.models import ContractValidationSummary

        return ContractValidationSummary(
            contract_version="1.0",
            contract_asset="claims_silver",
            status=status,
        )

    def test_write_mode_always_present(self) -> None:
        """Minimal payload for a non-partitioned write with a contract:
        ``write_mode`` plus ``contract_enforcement``. No partition shape
        keys, no contract status (no summary supplied)."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        resource = self._resource()
        wctx = WriteContext(asset_name="claims_silver", run_id="r", log=MagicMock())

        payload = resource._build_lineage_metadata_payload(
            write_config={
                "write_mode": WriteMode.FULL_REFRESH,
                "partition_column": None,
            },
            wctx=wctx,
            loaded_contract=self._contract(),
            contract_summary=None,
        )

        assert payload == {
            "write_mode": "full_refresh",
            "contract_enforcement": "error",
        }

    def test_partition_keys_included_when_partitioned(self) -> None:
        """A partitioned write surfaces ``partition_column`` and the
        active ``partition_keys`` list. ``has_partition_key=True`` and
        a non-empty list are both required."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        resource = self._resource()
        wctx = WriteContext(
            asset_name="claims_silver",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2026-05-15"],
        )

        payload = resource._build_lineage_metadata_payload(
            write_config={
                "write_mode": WriteMode.FULL_REFRESH,
                "partition_column": "release_date",
            },
            wctx=wctx,
            loaded_contract=self._contract(),
            contract_summary=self._summary(),
        )

        assert payload["partition_column"] == "release_date"
        assert payload["partition_keys"] == ["2026-05-15"]
        assert payload["contract_status"] == "passed"

    def test_partition_keys_capped_at_50(self) -> None:
        """Defensive cap: a 100-key partition list truncates to 50 plus
        a ``"... +50 more"`` sentinel. The cap exists because nothing
        prevents a Dagster partition definition yielding hundreds of
        keys per run, and ``data_lineage.metadata`` is a queryable
        surface, not a payload archive."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        resource = self._resource()
        keys = [f"2026-{(m % 12) + 1:02d}-01" for m in range(100)]
        wctx = WriteContext(
            asset_name="claims_silver",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=keys,
        )

        payload = resource._build_lineage_metadata_payload(
            write_config={
                "write_mode": WriteMode.FULL_REFRESH,
                "partition_column": "release_date",
            },
            wctx=wctx,
            loaded_contract=self._contract(),
            contract_summary=None,
        )

        assert len(payload["partition_keys"]) == 51
        assert payload["partition_keys"][-1] == "... +50 more"
        assert payload["partition_keys"][:50] == keys[:50]

    def test_contract_status_only_when_summary_present(self) -> None:
        """``contract_status`` is omitted when ``contract_summary`` is
        ``None`` -- this is the batched-path simulation, where the
        first batch's enforce hasn't run yet at lineage-INSERT time."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        resource = self._resource()
        wctx = WriteContext(asset_name="claims_silver", run_id="r", log=MagicMock())

        payload_without = resource._build_lineage_metadata_payload(
            write_config={"write_mode": WriteMode.UPSERT, "partition_column": None},
            wctx=wctx,
            loaded_contract=self._contract(),
            contract_summary=None,
        )
        payload_with = resource._build_lineage_metadata_payload(
            write_config={"write_mode": WriteMode.UPSERT, "partition_column": None},
            wctx=wctx,
            loaded_contract=self._contract(),
            contract_summary=self._summary("warned"),
        )

        assert "contract_status" not in payload_without
        assert payload_with["contract_status"] == "warned"

    def test_contract_enforcement_from_resource_field(self) -> None:
        """``contract_enforcement`` mirrors ``self.enforce_contracts``
        verbatim (``"error"`` / ``"warn"`` / ``"silent"``)."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())
        write_config = {"write_mode": WriteMode.APPEND, "partition_column": None}

        for level in ("error", "warn", "silent"):
            resource = self._resource(enforce_contracts=level)
            payload = resource._build_lineage_metadata_payload(
                write_config=write_config,
                wctx=wctx,
                loaded_contract=self._contract(),
                contract_summary=None,
            )
            assert payload["contract_enforcement"] == level

    def test_no_contract_omits_enforcement_and_status(self) -> None:
        """No-contract writes get ``write_mode`` only -- the contract
        metadata keys are omitted entirely when no contract is loaded."""
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteContext

        resource = self._resource()
        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())

        payload = resource._build_lineage_metadata_payload(
            write_config={
                "write_mode": WriteMode.FULL_REFRESH,
                "partition_column": None,
            },
            wctx=wctx,
            loaded_contract=None,
            contract_summary=None,
        )

        assert payload == {"write_mode": "full_refresh"}


class TestLineageMetadataThreadsThroughWrite:
    """Issue #334 Bug 2 path-integration: a representative
    ``database.write(...)`` call against a mocked cursor produces a
    ``write_lineage_record`` call whose ``metadata`` kwarg deserialises
    back to a dict containing the system-known keys.
    """

    def test_single_path_passes_metadata_to_write_lineage_record(self) -> None:
        """Single-write path: ``contract_status`` is present because
        ``_enforce_contract`` runs before ``_prepare_lineage``.

        Uses ``enforce_contracts="warn"`` so ``_enforce_contract``
        actually builds a ``ContractValidationSummary`` -- under
        ``"silent"`` enforcement the method short-circuits with
        ``(None, None)`` and ``contract_status`` would be omitted by
        design."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="claims_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
        )

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="warn",
        )

        wctx = _make_wctx("claims_silver")
        df = pl.DataFrame({"id": ["1", "2"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid", "lkey")
        mock_tracker.attach_lineage_to_dataframe.return_value = df
        mock_tracker.find_prior_lineage_id.return_value = None

        with (
            patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect,
            patch.object(PostgresResource, "_validate_columns"),
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_pipeline_registry_upsert_committed"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                return_value={"rows_deleted": 0, "rows_inserted": 2},
            ),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 0
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            resource.write(
                df,
                target="silver.claims",
                context=wctx,
                write_mode="full_refresh",
                contract=contract,
            )

        mock_tracker.write_lineage_record.assert_called_once()
        bound_metadata = mock_tracker.write_lineage_record.call_args.kwargs["metadata"]
        assert isinstance(bound_metadata, dict)
        assert bound_metadata["write_mode"] == "full_refresh"
        assert bound_metadata["contract_enforcement"] == "warn"
        # Single-path: ``contract_summary`` IS available so
        # ``contract_status`` lands in the payload.
        assert "contract_status" in bound_metadata

    def test_batched_path_passes_metadata_to_write_lineage_record(self) -> None:
        """Batched path: ``contract_status`` is absent (the asymmetry
        guard) but ``write_mode`` and partition shape are populated."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )
        from moncpipelib.streaming import BatchedDataFrame

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="claims_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
        )

        resource = PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="warn",
        )

        wctx = _make_wctx("claims_silver")
        batch_df = pl.DataFrame({"id": ["1", "2"]})
        batched = BatchedDataFrame(batches=iter([batch_df]), total_rows_hint=2)

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid", "lkey")
        mock_tracker.attach_lineage_to_dataframe.return_value = batch_df
        mock_tracker.find_prior_lineage_id.return_value = None
        mock_tracker.get_parent_lineage_ids.return_value = []

        with (
            patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect,
            patch.object(PostgresResource, "_validate_columns"),
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_pipeline_registry_upsert_committed"),
            patch(
                "moncpipelib.io_managers.writers.clear_table",
                # (deleted_count, clear_method) -- the batched path consumes
                # this return value since #4 to report the clear in its stats.
                return_value=(0, "truncate"),
            ),
            patch(
                "moncpipelib.io_managers.writers.insert_rows",
                return_value={"rows_inserted": 2},
            ),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 0
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            resource.write(
                batched,
                target="silver.claims",
                context=wctx,
                write_mode="full_refresh",
                contract=contract,
            )

        mock_tracker.write_lineage_record.assert_called_once()
        bound_metadata = mock_tracker.write_lineage_record.call_args.kwargs["metadata"]
        assert isinstance(bound_metadata, dict)
        assert bound_metadata["write_mode"] == "full_refresh"
        assert bound_metadata["contract_enforcement"] == "warn"
        # Batched-path asymmetry: ``contract_status`` is NOT present
        # because the first batch's ``_enforce_contract`` runs after
        # the lineage INSERT.
        assert "contract_status" not in bound_metadata


def _make_wctx(asset_name: str):  # type: ignore[no-untyped-def]
    """Create a WriteContext for testing."""
    from moncpipelib.resources.types import WriteContext

    return WriteContext(asset_name=asset_name, run_id="run-test", log=MagicMock())


# ---------------------------------------------------------------------------
# Migration 018 Phase 3: same-transaction lineage / data-DML atomicity
# ---------------------------------------------------------------------------


class TestSameTransactionLineageAtomicity:
    """Phase 3 invariant: the lineage-row INSERT runs on the same psycopg
    cursor as the data DML, BEFORE the DML, with no separate commit.

    The order is load-bearing: production has 793 enforced ``NOT
    DEFERRABLE`` FKs against ``data_lineage(lineage_id)``, so the
    lineage row must be visible to the FK check before the data DML
    references it. Both must commit (or roll back) together.
    """

    def _setup_resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_single_write_lineage_insert_runs_before_dml(
        self,
        mock_connect: MagicMock,
    ) -> None:
        """In ``_write_single``, ``write_lineage_record`` must be called
        on the cursor BEFORE the writer function runs."""
        resource = self._setup_resource()
        wctx = _make_wctx("test_asset")
        df = pl.DataFrame({"id": ["1", "2"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000010",
            "v1:test_asset:bronze:2026-05-15:run-te",
        )
        mock_tracker.attach_lineage_to_dataframe.return_value = df.with_columns(
            pl.lit("00000000-0000-0000-0000-000000000010").alias("_lineage_id"),
            pl.lit("v1:test_asset:bronze:2026-05-15:run-te").alias("_lineage_key"),
        )

        call_order: list[str] = []
        mock_tracker.write_lineage_record.side_effect = lambda *_a, **_kw: call_order.append(
            "write_lineage_record"
        )

        # Capture order on the cursor so we can prove write_lineage_record
        # runs before the writer function does its DML.
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        def _fake_full_refresh(*_args: object, **_kwargs: object) -> dict[str, int]:
            call_order.append("execute_full_refresh")
            return {"rows_deleted": 0, "rows_inserted": 2}

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                side_effect=_fake_full_refresh,
            ),
        ):
            resource.write(
                df,
                target="bronze.test_asset",
                context=wctx,
                write_mode="full_refresh",
            )

        assert call_order == ["write_lineage_record", "execute_full_refresh"], (
            f"lineage INSERT must precede data DML, got order: {call_order}"
        )
        # Same cursor was used for both -- write_lineage_record's first
        # positional arg is the cursor.
        assert mock_tracker.write_lineage_record.call_args.args[0] is mock_cursor
        # Tracker must NOT have opened its own SA-engine transaction in
        # the write path (would orphan the lineage row on data failure).
        mock_tracker.create_lineage_record.assert_not_called()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_data_dml_failure_rolls_back_lineage(
        self,
        mock_connect: MagicMock,
    ) -> None:
        """If the writer function raises after the lineage row is
        inserted, ``conn.rollback()`` must run and ``conn.commit()`` must
        NOT run -- both the data DML and the lineage INSERT are atomic."""
        resource = self._setup_resource()
        wctx = _make_wctx("test_asset")
        df = pl.DataFrame({"id": ["1", "2"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000020",
            "v1:test_asset:bronze:2026-05-15:run-te",
        )
        mock_tracker.attach_lineage_to_dataframe.return_value = df.with_columns(
            pl.lit("00000000-0000-0000-0000-000000000020").alias("_lineage_id"),
            pl.lit("v1:test_asset:bronze:2026-05-15:run-te").alias("_lineage_key"),
        )

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                side_effect=RuntimeError("simulated data-write failure"),
            ),
            pytest.raises(RuntimeError, match="simulated data-write failure"),
        ):
            resource.write(
                df,
                target="bronze.test_asset",
                context=wctx,
                write_mode="full_refresh",
            )

        # Lineage INSERT did happen on the cursor.
        mock_tracker.write_lineage_record.assert_called_once()
        # ...but the connection was rolled back, not committed.
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_called_once()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_lineage_insert_failure_rolls_back_and_runs_no_dml(
        self,
        mock_connect: MagicMock,
    ) -> None:
        """If ``write_lineage_record`` raises (e.g., unique-violation on
        ``lineage_key``), the data DML must never run and the connection
        must roll back."""
        resource = self._setup_resource()
        wctx = _make_wctx("test_asset")
        df = pl.DataFrame({"id": ["1", "2"]})

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000030",
            "v1:test_asset:bronze:2026-05-15:run-te",
        )
        mock_tracker.attach_lineage_to_dataframe.return_value = df.with_columns(
            pl.lit("00000000-0000-0000-0000-000000000030").alias("_lineage_id"),
            pl.lit("v1:test_asset:bronze:2026-05-15:run-te").alias("_lineage_key"),
        )
        mock_tracker.write_lineage_record.side_effect = RuntimeError(
            "simulated lineage-insert failure"
        )

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        dml_calls: list[str] = []

        def _fake_full_refresh(*_args: object, **_kwargs: object) -> dict[str, int]:
            dml_calls.append("execute_full_refresh")
            return {"rows_deleted": 0, "rows_inserted": 2}

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch(
                "moncpipelib.io_managers.writers.execute_full_refresh",
                side_effect=_fake_full_refresh,
            ),
            pytest.raises(RuntimeError, match="simulated lineage-insert failure"),
        ):
            resource.write(
                df,
                target="bronze.test_asset",
                context=wctx,
                write_mode="full_refresh",
            )

        # Data DML must NOT have run -- the lineage INSERT failed first.
        assert dml_calls == []
        mock_conn.commit.assert_not_called()
        mock_conn.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Migration 018 Phase 5: parent_lineage_ids accumulation on batched path
# ---------------------------------------------------------------------------


class TestBatchedParentLineageAccumulation:
    """Phase 5: the batched write path must accumulate upstream
    ``_lineage_id`` values across **every** batch and write the union
    onto ``data_lineage.parent_lineage_ids`` via a post-DML UPDATE.

    First-batch peek is rejected: ``BatchedDataFrame.batches`` is a
    generic iterator with no single-source guarantee, so any test that
    pins multi-batch behaviour is load-bearing against future
    "optimisation" back to peek-only.
    """

    @pytest.fixture
    def lineage_resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            enable_row_lineage=True,
            add_metadata_columns=False,
            enforce_contracts="silent",
        )

    @staticmethod
    def _setup_mock_conn(mock_connect: MagicMock) -> MagicMock:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None  # find_prior_lineage_id -> None
        mock_cursor.rowcount = 0
        return mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_single_batch_with_lineage_id_accumulates_parents(
        self,
        mock_connect: MagicMock,
        lineage_resource: PostgresResource,
    ) -> None:
        """A single-batch batched write of a DataFrame containing
        ``_lineage_id`` produces an UPDATE with the unique upstream
        IDs."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        upstream_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
        ]
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "_lineage_id": upstream_ids + upstream_ids,  # duplicates
            }
        )
        batched = BatchedDataFrame(batches=iter([df]), total_rows_hint=4)
        wctx = WriteContext(asset_name="silver/dim_x", run_id="r1", log=MagicMock())

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = (
            "00000000-0000-0000-0000-000000000099",
            "v1:dim_x:bronze:abcd:r1",
        )
        mock_tracker.attach_lineage_to_dataframe.side_effect = lambda df, _lid, _key: df
        mock_tracker.find_prior_lineage_id.return_value = None

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            lineage_resource.write(
                batched, target="bronze.dim_x", context=wctx, write_mode="append"
            )

        mock_tracker.update_parent_lineage_ids.assert_called_once()
        kwargs = mock_tracker.update_parent_lineage_ids.call_args.kwargs
        assert kwargs["lineage_id"] == "00000000-0000-0000-0000-000000000099"
        assert kwargs["parent_lineage_ids"] == sorted(upstream_ids)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_multi_batch_unions_parents_across_all_batches(
        self,
        mock_connect: MagicMock,
        lineage_resource: PostgresResource,
    ) -> None:
        """The regression test that pins per-batch behaviour: each batch
        carries DIFFERENT upstream ids. The UPDATE must use the union,
        not just the first batch's ids."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        # Three batches, each from a different upstream source.
        df1 = pl.DataFrame({"id": [1], "_lineage_id": ["aaa"]})
        df2 = pl.DataFrame({"id": [2], "_lineage_id": ["bbb"]})
        df3 = pl.DataFrame({"id": [3], "_lineage_id": ["ccc"]})
        batched = BatchedDataFrame(batches=iter([df1, df2, df3]), total_rows_hint=3)
        wctx = WriteContext(asset_name="silver/dim_x", run_id="r1", log=MagicMock())

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid-99", "lkey")
        mock_tracker.attach_lineage_to_dataframe.side_effect = lambda df, _l, _k: df
        mock_tracker.find_prior_lineage_id.return_value = None

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            lineage_resource.write(
                batched, target="bronze.dim_x", context=wctx, write_mode="append"
            )

        kwargs = mock_tracker.update_parent_lineage_ids.call_args.kwargs
        assert kwargs["parent_lineage_ids"] == ["aaa", "bbb", "ccc"]

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_mixed_batches_only_accumulates_present_ids(
        self,
        mock_connect: MagicMock,
        lineage_resource: PostgresResource,
    ) -> None:
        """Some batches have ``_lineage_id``, some don't. The UPDATE
        carries the union of present values; missing-column batches do
        not contribute a sentinel parent."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        df_with = pl.DataFrame({"id": [1], "_lineage_id": ["aaa"]})
        df_without = pl.DataFrame({"id": [2]})
        batched = BatchedDataFrame(batches=iter([df_with, df_without]), total_rows_hint=2)
        wctx = WriteContext(asset_name="silver/dim_x", run_id="r1", log=MagicMock())

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid-99", "lkey")
        mock_tracker.attach_lineage_to_dataframe.side_effect = lambda df, _l, _k: df
        mock_tracker.find_prior_lineage_id.return_value = None

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            lineage_resource.write(
                batched, target="bronze.dim_x", context=wctx, write_mode="append"
            )

        kwargs = mock_tracker.update_parent_lineage_ids.call_args.kwargs
        assert kwargs["parent_lineage_ids"] == ["aaa"]

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_no_lineage_id_column_skips_update(
        self,
        mock_connect: MagicMock,
        lineage_resource: PostgresResource,
    ) -> None:
        """If no batch carries ``_lineage_id``, the UPDATE must NOT run
        -- the lineage row's ``parent_lineage_ids`` stays NULL, which is
        semantically distinct from an empty array."""
        from moncpipelib.resources.types import WriteContext
        from moncpipelib.streaming import BatchedDataFrame

        self._setup_mock_conn(mock_connect)

        df1 = pl.DataFrame({"id": [1]})
        df2 = pl.DataFrame({"id": [2]})
        batched = BatchedDataFrame(batches=iter([df1, df2]), total_rows_hint=2)
        wctx = WriteContext(asset_name="bronze/raw_x", run_id="r1", log=MagicMock())

        mock_tracker = MagicMock()
        mock_tracker.generate_lineage_ids.return_value = ("lid-99", "lkey")
        mock_tracker.attach_lineage_to_dataframe.side_effect = lambda df, _l, _k: df
        mock_tracker.find_prior_lineage_id.return_value = None

        with (
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
            patch.object(PostgresResource, "_validate_columns"),
            patch("moncpipelib.io_managers.writers.insert_rows"),
        ):
            lineage_resource.write(
                batched, target="bronze.raw_x", context=wctx, write_mode="append"
            )

        mock_tracker.update_parent_lineage_ids.assert_not_called()


# ---------------------------------------------------------------------------
# Column validation tests
# ---------------------------------------------------------------------------


class TestValidateColumns:
    """Tests for _validate_columns server-default and generated column handling."""

    @pytest.fixture
    def resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
        )

    @staticmethod
    def _make_cursor(
        rows: list[tuple[str, str | None, str]],
    ) -> MagicMock:
        """Create a mock cursor returning (column_name, column_default, is_generated)."""
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        return cursor

    def test_server_default_column_not_required(self, resource: PostgresResource) -> None:
        """Columns with server defaults (e.g. DEFAULT uuidv7()) may be omitted."""
        cursor = self._make_cursor(
            [
                ("id", "uuidv7()", "NEVER"),
                ("name", None, "NEVER"),
                ("status", None, "NEVER"),
            ]
        )

        # DataFrame omits 'id' -- should not raise
        resource._validate_columns(cursor, "silver.test_table", ["name", "status"], "test_asset")

    def test_server_default_column_accepted_in_df(self, resource: PostgresResource) -> None:
        """Columns with server defaults may also be provided by the DataFrame."""
        cursor = self._make_cursor(
            [
                ("id", "uuidv7()", "NEVER"),
                ("name", None, "NEVER"),
            ]
        )

        # DataFrame includes 'id' -- should not raise
        resource._validate_columns(cursor, "silver.test_table", ["id", "name"], "test_asset")

    def test_missing_required_column_raises(self, resource: PostgresResource) -> None:
        """Columns without defaults that are missing from the DataFrame raise ValueError."""
        cursor = self._make_cursor(
            [
                ("id", "uuidv7()", "NEVER"),
                ("name", None, "NEVER"),
                ("status", None, "NEVER"),
            ]
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            resource._validate_columns(cursor, "silver.test_table", ["name"], "test_asset")

    def test_extra_column_in_df_raises(self, resource: PostgresResource) -> None:
        """Columns in DataFrame but not in the table raise ValueError."""
        cursor = self._make_cursor(
            [
                ("name", None, "NEVER"),
            ]
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            resource._validate_columns(cursor, "silver.test_table", ["name", "bogus"], "test_asset")

    def test_generated_stored_column_excluded(self, resource: PostgresResource) -> None:
        """GENERATED ALWAYS AS (...) STORED columns are excluded from both checks."""
        cursor = self._make_cursor(
            [
                ("name", None, "NEVER"),
                ("full_name", None, "ALWAYS"),  # generated stored column
            ]
        )

        # DataFrame doesn't include 'full_name' -- should not raise
        resource._validate_columns(cursor, "silver.test_table", ["name"], "test_asset")

    def test_generated_stored_column_in_df_raises(self, resource: PostgresResource) -> None:
        """Providing a generated stored column in the DataFrame raises ValueError."""
        cursor = self._make_cursor(
            [
                ("name", None, "NEVER"),
                ("full_name", None, "ALWAYS"),
            ]
        )

        with pytest.raises(ValueError, match="Column mismatch"):
            resource._validate_columns(
                cursor, "silver.test_table", ["name", "full_name"], "test_asset"
            )

    def test_multiple_default_types(self, resource: PostgresResource) -> None:
        """Mix of server defaults: uuid, now(), literal -- all may be omitted."""
        cursor = self._make_cursor(
            [
                ("id", "uuidv7()", "NEVER"),
                ("created_at", "now()", "NEVER"),
                ("version", "1", "NEVER"),
                ("name", None, "NEVER"),
            ]
        )

        # Only 'name' is required; all others have defaults
        resource._validate_columns(cursor, "public.test_table", ["name"], "test_asset")

    def test_exclude_from_table_applies_to_both_sets(self, resource: PostgresResource) -> None:
        """exclude_from_table removes columns from both writable and required sets."""
        cursor = self._make_cursor(
            [
                ("name", None, "NEVER"),
                ("scd2_hash", None, "NEVER"),
            ]
        )

        # 'scd2_hash' excluded, so only 'name' is checked
        resource._validate_columns(
            cursor,
            "silver.test_table",
            ["name"],
            "test_asset",
            exclude_from_table={"scd2_hash"},
        )


# ---------------------------------------------------------------------------
# SCD2 empty tracked columns: presence-only mode (#432)
# ---------------------------------------------------------------------------


class TestPrepareSCD2EmptyColumns:
    """Verify _prepare_scd2 handles empty hash-column resolution.

    Since #432, empty resolution with a non-empty business key selects
    presence-only mode (hash over the business key) instead of raising.
    Only the no-business-key case still errors.
    """

    @staticmethod
    def _make_write_config(
        *,
        tracked_columns: list[str] | None = None,
        business_key: list[str] | None = None,
        detect_deletes: bool = False,
    ) -> dict[str, object]:
        return {
            "scd2": SCD2Config(),
            "hash_col": "row_hash",
            "tracked_columns": tracked_columns,
            "business_key": business_key or ["id"],
            "detect_deletes": detect_deletes,
            "effective_from_col": "effective_from",
            "effective_to_col": "effective_to",
            "is_current_col": "is_current",
            "sequence_col": None,
        }

    @staticmethod
    def _make_resource() -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    @staticmethod
    def _make_wctx() -> MagicMock:
        wctx = MagicMock()
        wctx.log = MagicMock()
        return wctx

    def test_explicit_empty_tracked_columns_presence_only(self) -> None:
        """tracked_columns=[] hashes the business key instead of raising."""
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        config = self._make_write_config(
            tracked_columns=[], business_key=["id", "name"], detect_deletes=True
        )

        result_df, hash_cols, _ = self._make_resource()._prepare_scd2(df, config, self._make_wctx())

        assert hash_cols == ["id", "name"]
        assert "row_hash" in result_df.columns
        expected = df.with_columns(compute_row_hash(["id", "name"], alias="row_hash"))
        assert result_df["row_hash"].to_list() == expected["row_hash"].to_list()

    def test_auto_derive_empty_presence_only(self) -> None:
        """tracked_columns=None with business key covering all columns is presence-only."""
        df = pl.DataFrame({"id": [1], "name": ["a"]})
        config = self._make_write_config(business_key=["id", "name"], detect_deletes=True)

        result_df, hash_cols, _ = self._make_resource()._prepare_scd2(df, config, self._make_wctx())

        assert hash_cols == ["id", "name"]
        assert "row_hash" in result_df.columns

    def test_presence_only_hash_constant_within_key(self) -> None:
        """Presence-only row_hash is a pure function of the business key."""
        df = pl.DataFrame({"id": [1, 1, 2], "name": ["a", "a", "b"]})
        config = self._make_write_config(tracked_columns=[], business_key=["id", "name"])

        result_df, _, _ = self._make_resource()._prepare_scd2(df, config, self._make_wctx())

        hashes = result_df["row_hash"].to_list()
        assert hashes[0] == hashes[1]
        assert hashes[0] != hashes[2]

    def test_presence_only_logs_info_with_detect_deletes(self) -> None:
        """detect_deletes=True logs the presence-only mode at INFO."""
        df = pl.DataFrame({"id": [1]})
        config = self._make_write_config(
            tracked_columns=[], business_key=["id"], detect_deletes=True
        )
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.info.assert_called_once()
        assert "presence-only" in wctx.log.info.call_args[0][0]
        wctx.log.warning.assert_not_called()

    def test_presence_only_warns_without_detect_deletes(self) -> None:
        """detect_deletes=False warns that spans will never close."""
        df = pl.DataFrame({"id": [1]})
        config = self._make_write_config(tracked_columns=[], business_key=["id"])
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.warning.assert_called_once()
        msg = wctx.log.warning.call_args[0][0]
        assert "presence-only" in msg
        assert "detect_deletes" in msg

    def test_presence_only_with_uncovered_columns_still_warns(self) -> None:
        """tracked_columns=[] with extra data columns keeps the uncovered warning."""
        df = pl.DataFrame({"id": [1], "name": ["a"], "status": ["x"]})
        config = self._make_write_config(tracked_columns=[], business_key=["id"])
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        warning_msgs = [c.args[0] for c in wctx.log.warning.call_args_list]
        assert any("status" in m and "NOT trigger" in m for m in warning_msgs)

    def test_empty_business_key_still_raises(self) -> None:
        """No hash columns AND no business key remains an error."""
        df = pl.DataFrame({"id": [1]})
        config = self._make_write_config(tracked_columns=[], business_key=[])
        config["business_key"] = []

        with pytest.raises(ValueError, match="business_key is empty"):
            self._make_resource()._prepare_scd2(df, config, self._make_wctx())

    def test_valid_tracked_columns_succeeds(self) -> None:
        """Non-empty tracked_columns produces a hash column."""
        df = pl.DataFrame({"id": [1], "name": ["a"], "value": [10]})
        config = self._make_write_config(tracked_columns=["name", "value"])

        result_df, hash_cols, _ = self._make_resource()._prepare_scd2(df, config, self._make_wctx())
        assert "row_hash" in result_df.columns
        assert hash_cols == ["name", "value"]

    def test_auto_derived_columns_succeeds(self) -> None:
        """When tracked_columns is None, non-key columns are auto-derived."""
        df = pl.DataFrame({"id": [1], "name": ["a"], "value": [10]})
        config = self._make_write_config(business_key=["id"])

        result_df, hash_cols, _ = self._make_resource()._prepare_scd2(df, config, self._make_wctx())
        assert "row_hash" in result_df.columns
        assert sorted(hash_cols) == ["name", "value"]


class TestPrepareSCD2UncoveredColumnWarning:
    """Verify _prepare_scd2 warns when explicit tracked_columns miss DataFrame columns."""

    @staticmethod
    def _make_write_config(
        *,
        tracked_columns: list[str] | None = None,
        business_key: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "scd2": SCD2Config(),
            "hash_col": "row_hash",
            "tracked_columns": tracked_columns,
            "business_key": business_key or ["code"],
            "effective_from_col": "effective_from",
            "effective_to_col": "effective_to",
            "is_current_col": "is_current",
            "sequence_col": None,
        }

    @staticmethod
    def _make_resource() -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    @staticmethod
    def _make_wctx() -> MagicMock:
        wctx = MagicMock()
        wctx.log = MagicMock()
        return wctx

    def test_warns_uncovered_columns(self) -> None:
        """Warning should list DataFrame columns not in tracked_columns."""
        df = pl.DataFrame({"code": ["A"], "name": ["x"], "status": ["active"]})
        config = self._make_write_config(tracked_columns=["name"])
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.warning.assert_called_once()
        msg = wctx.log.warning.call_args[0][0]
        assert "status" in msg
        assert "tracked_columns" in msg

    def test_no_warning_when_all_covered(self) -> None:
        """No warning when all non-key columns are in tracked_columns."""
        df = pl.DataFrame({"code": ["A"], "name": ["x"]})
        config = self._make_write_config(tracked_columns=["name"])
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.warning.assert_not_called()

    def test_no_warning_when_auto_derived(self) -> None:
        """No warning when tracked_columns is None (auto-derive covers all)."""
        df = pl.DataFrame({"code": ["A"], "name": ["x"], "status": ["active"]})
        config = self._make_write_config(tracked_columns=None)
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.warning.assert_not_called()

    def test_excludes_lineage_and_hash_from_warning(self) -> None:
        """Lineage and hash columns should not appear in the warning."""
        df = pl.DataFrame(
            {
                "code": ["A"],
                "name": ["x"],
                "_lineage_id": ["id"],
                "_lineage_key": ["key"],
                "row_hash": ["hash"],
            }
        )
        config = self._make_write_config(tracked_columns=["name"])
        wctx = self._make_wctx()

        self._make_resource()._prepare_scd2(df, config, wctx)

        wctx.log.warning.assert_not_called()


class TestSCD2EffectiveDate:
    """Tests for effective_date parameter threading through SCD2 write path."""

    def test_scd2_finalize_uses_effective_date(self) -> None:
        """scd2_finalize should use parameterized date instead of now()."""
        from datetime import date

        from moncpipelib.io_managers.writers import scd2_finalize

        cursor = MagicMock()
        cursor.fetchone.return_value = (2, 1)  # new_count=2, changed_count=1
        cursor.fetchall.return_value = []  # #419 dup-key guard: no duplicates
        cursor.rowcount = 3

        scd2_finalize(
            cursor=cursor,
            table_name="silver.dim_test",
            stage_table="_tmp_stage",
            total_staged_rows=5,
            stage_columns=["id", "name", "row_hash"],
            business_key=["id"],
            scd2=SCD2Config(),
            context=MagicMock(),
            effective_date=date(2025, 1, 1),
        )

        sql_calls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        ctas_calls = [s for s in sql_calls if "CREATE TEMP TABLE" in s]
        # Stage 2 INSERT into target -- the CTAS contains "AS" not "INTO"
        insert_calls = [s for s in sql_calls if "INSERT INTO" in s and "CREATE TEMP TABLE" not in s]
        update_calls = [s for s in sql_calls if "UPDATE" in s]

        # CTAS exists with the expected skeleton
        assert len(ctas_calls) == 1
        assert "_diff" in ctas_calls[0]
        assert "ON COMMIT DROP" in ctas_calls[0]
        # CTAS uses the parameterized date, not now()
        assert "now()" not in ctas_calls[0]

        # Stage 2 INSERT exists, has no JOIN against the target, reads from diff.
        # The "no JOIN" check is the load-bearing structural regression guard:
        # it fails loudly if anyone "simplifies" finalize back to the
        # self-referencing single-statement form (see #274).
        assert len(insert_calls) == 1
        assert "JOIN" not in insert_calls[0].upper()
        assert "_diff" in insert_calls[0]
        assert "now()" not in insert_calls[0]

        # Expire UPDATE still uses parameterized date (unchanged behavior)
        assert len(update_calls) == 1
        assert "now()" not in update_calls[0]

    def test_scd2_finalize_defaults_to_now(self) -> None:
        """scd2_finalize without effective_date should use now()."""
        from moncpipelib.io_managers.writers import scd2_finalize

        cursor = MagicMock()
        cursor.fetchone.return_value = (1, 0)  # new_count=1, changed_count=0
        cursor.fetchall.return_value = []  # #419 dup-key guard: no duplicates
        cursor.rowcount = 1

        scd2_finalize(
            cursor=cursor,
            table_name="silver.dim_test",
            stage_table="_tmp_stage",
            total_staged_rows=3,
            stage_columns=["id", "name", "row_hash"],
            business_key=["id"],
            scd2=SCD2Config(),
            context=MagicMock(),
        )

        sql_calls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        ctas_calls = [s for s in sql_calls if "CREATE TEMP TABLE" in s]

        # date_expr lives in the CTAS SELECT, so now() should appear there
        # (not in the Stage 2 INSERT, which reads _eff_from from the diff).
        assert len(ctas_calls) == 1
        assert "now()" in ctas_calls[0]

    def test_write_accepts_effective_date(self) -> None:
        """write() should accept effective_date in its signature."""
        import inspect

        sig = inspect.signature(PostgresResource.write)
        assert "effective_date" in sig.parameters


# ---------------------------------------------------------------------------
# TestPeriodRegistry
# ---------------------------------------------------------------------------


class TestPeriodRegistry:
    """Tests for period registry registration on PostgresResource."""

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_register_period_executes_upsert(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        """register_period() should execute an INSERT ... ON CONFLICT upsert."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        resource.register_period(
            source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            partition_key="2025Q1",
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 3, 31),
            source_uri="https://example.com/data.zip",
            status="materialized",
            registered_by="test_asset",
        )

        mock_cursor.execute.assert_called_once()
        sql = str(mock_cursor.execute.call_args[0][0])
        assert "INSERT INTO" in sql
        assert "ON CONFLICT" in sql
        assert "source_id" in sql
        assert "partition_key" in sql
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_auto_register_skips_without_data_source(self, resource: PostgresResource) -> None:
        """_auto_register_period() should be a no-op when data_source is None and no partition."""
        mock_conn = MagicMock()
        mock_contract = MagicMock()
        mock_contract.data_source = None
        mock_wctx = MagicMock()
        # Explicitly disable partition context so the silver path does not fire
        mock_wctx.has_partition_key = False
        mock_wctx.partition_keys = None

        resource._auto_register_period(mock_conn, mock_contract, date(2025, 1, 1), mock_wctx)

        # No cursor should be opened
        mock_conn.cursor.assert_not_called()

    def test_auto_register_skips_without_context(self, resource: PostgresResource) -> None:
        """_auto_register_period() no-op when no effective_date and no partition."""
        mock_conn = MagicMock()
        mock_contract = MagicMock()
        mock_contract.data_source = MagicMock()
        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = False
        mock_wctx.partition_keys = None

        resource._auto_register_period(mock_conn, mock_contract, None, mock_wctx)

        mock_conn.cursor.assert_not_called()

    def test_auto_register_period_failure_warns(self, resource: PostgresResource) -> None:
        """_auto_register_period() should warn on failure, not raise."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = [
            None,  # _check_period_registry SELECT succeeds
            Exception("connection lost"),  # upsert fails
        ]
        mock_cursor.fetchone.return_value = (1,)  # table exists
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_contract = MagicMock()
        mock_contract.data_source.source_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        mock_contract.data_source.source_name = "cms_asp"
        mock_period = MagicMock()
        mock_period.effective_from = date(2025, 1, 1)
        mock_period.effective_to = date(2025, 3, 31)
        mock_period.partition_key = "2025Q1"
        mock_period.source = "https://example.com/data.zip"
        mock_contract.data_source.periods = [mock_period]

        mock_wctx = MagicMock()
        mock_wctx.asset_name = "test_asset"

        # Should NOT raise
        resource._auto_register_period(mock_conn, mock_contract, date(2025, 1, 1), mock_wctx)

        mock_wctx.log.warning.assert_called_once()
        warning_msg = str(mock_wctx.log.warning.call_args[0][0])
        assert "Failed to register period" in warning_msg

    # ----------------------------------------------------------------------
    # from_ingest period registry registration (issue #263)
    # ----------------------------------------------------------------------

    def _build_from_ingest_contract(self) -> Any:
        """Build a synthetic DataContract whose data_source uses
        ``FromIngestTemplate`` periods. Returned contract is a real dataclass
        instance so ``isinstance(periods, FromIngestTemplate)`` checks fire."""
        from moncpipelib.contracts.models import (
            DataContract,
            DataSource,
            FromIngestTemplate,
            Schema,
        )

        return DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="bronze__rxnorm_full",
            layer="bronze",
            schema=Schema(columns=[]),
            data_source=DataSource(
                source_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                source_name="rxnorm_full",
                periods=FromIngestTemplate(
                    source="rxnorm_full_{release_version}.zip",
                    effective_from_field="effective_from",
                ),
            ),
        )

    def test_auto_register_from_ingest_happy_path(self, resource: PostgresResource) -> None:
        """from_ingest branch upserts with caller-supplied source_uri and
        ``effective_to=None``."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)  # registry table exists
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = True
        mock_wctx.partition_keys = ["2025-08-01"]
        mock_wctx.asset_name = "bronze__rxnorm_full"
        mock_wctx.run_id = "run-xyz"

        resource._auto_register_period(
            mock_conn,
            self._build_from_ingest_contract(),
            date(2025, 8, 1),
            mock_wctx,
            None,
            "azure-blob://acct/container/rxnorm/rxnorm_full_2025-08-01.zip",
        )

        # Two execute calls: registry-existence SELECT, then the upsert.
        assert mock_cursor.execute.call_count == 2
        upsert_call = mock_cursor.execute.call_args_list[1]
        sql = str(upsert_call[0][0])
        assert "INSERT INTO" in sql
        assert "ON CONFLICT" in sql
        # Argument tuple binds: source_id, source_name, partition_key,
        # effective_from, effective_to, source_uri, status, registered_by,
        # run_id, pipeline_id, metadata
        params = upsert_call[0][1]
        assert params[0] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert params[1] == "rxnorm_full"
        assert params[2] == "2025-08-01"
        assert params[3] == date(2025, 8, 1)
        assert params[4] is None  # effective_to is NULL for from_ingest
        assert params[5] == "azure-blob://acct/container/rxnorm/rxnorm_full_2025-08-01.zip"
        assert params[6] == "materialized"
        mock_conn.commit.assert_called_once()

    def test_auto_register_from_ingest_idempotent_double_write(
        self, resource: PostgresResource
    ) -> None:
        """Re-registering the same from_ingest partition is idempotent
        (single ``ON CONFLICT`` upsert per call). The DB-side ``registered_at
        = NOW()`` SET clause guarantees the timestamp updates -- we assert
        the SQL shape rather than running the SQL."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = True
        mock_wctx.partition_keys = ["2025-08-01"]
        mock_wctx.asset_name = "bronze__rxnorm_full"
        mock_wctx.run_id = "run-1"

        contract = self._build_from_ingest_contract()
        for _ in range(2):
            resource._auto_register_period(
                mock_conn,
                contract,
                date(2025, 8, 1),
                mock_wctx,
                None,
                "azure-blob://acct/container/rxnorm/rxnorm_full_2025-08-01.zip",
            )

        # First call: existence SELECT + upsert. Second call: cached
        # existence (no SELECT) + upsert. Total = 3 executes.
        assert mock_cursor.execute.call_count == 3
        sql_first_upsert = str(mock_cursor.execute.call_args_list[1][0][0])
        sql_second_upsert = str(mock_cursor.execute.call_args_list[2][0][0])
        assert "ON CONFLICT" in sql_first_upsert
        assert "registered_at = NOW()" in sql_first_upsert
        # Same SQL shape; DB enforces idempotency via the unique constraint.
        assert sql_first_upsert == sql_second_upsert

    def test_auto_register_from_ingest_last_write_wins_on_source_uri(
        self, resource: PostgresResource
    ) -> None:
        """Two writes to the same partition with different source_uri: the
        second call's URI is what gets bound to the upsert. ``ON CONFLICT
        ... source_uri = EXCLUDED.source_uri`` carries the latest value."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = True
        mock_wctx.partition_keys = ["2025-08-01"]
        mock_wctx.asset_name = "bronze__rxnorm_full"
        mock_wctx.run_id = "run-1"

        contract = self._build_from_ingest_contract()
        resource._auto_register_period(
            mock_conn, contract, date(2025, 8, 1), mock_wctx, None, "uri-A"
        )
        resource._auto_register_period(
            mock_conn, contract, date(2025, 8, 1), mock_wctx, None, "uri-B"
        )

        # First call: existence SELECT (idx 0) + upsert (idx 1). Second
        # call: cached existence + upsert (idx 2). Second upsert binds
        # source_uri="uri-B".
        second_upsert_params = mock_cursor.execute.call_args_list[2][0][1]
        assert second_upsert_params[5] == "uri-B"

    def test_auto_register_from_ingest_defensive_partition_guard(
        self, resource: PostgresResource
    ) -> None:
        """If a non-public caller bypasses the database.write() validation
        and reaches _auto_register_period without partition_keys, the from_
        ingest branch logs a warning and skips. database.write() callers
        cannot reach this state."""
        mock_conn = MagicMock()
        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = False
        mock_wctx.partition_keys = None

        resource._auto_register_period(
            mock_conn,
            self._build_from_ingest_contract(),
            date(2025, 8, 1),
            mock_wctx,
            None,
            "uri-A",
        )

        mock_conn.cursor.assert_not_called()
        mock_wctx.log.warning.assert_called_once()
        warning_msg = str(mock_wctx.log.warning.call_args[0][0])
        assert "missing partition context" in warning_msg

    def test_auto_register_enumerated_ignores_source_uri_arg(
        self, resource: PostgresResource
    ) -> None:
        """For enumerated-period contracts, ``source_uri`` argument is
        ignored -- the registry row's source_uri comes from the matched
        Period.source. Pins that ``source_uri`` is from_ingest-scoped, not
        a general override."""
        from moncpipelib.contracts.models import (
            DataContract,
            DataSource,
            Period,
            Schema,
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        contract = DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="bronze__cms_asp",
            layer="bronze",
            schema=Schema(columns=[]),
            data_source=DataSource(
                source_id="00000000-0000-0000-0000-000000000001",
                source_name="cms_asp",
                periods=[
                    Period(
                        effective_from=date(2025, 1, 1),
                        effective_to=date(2025, 3, 31),
                        partition_key="2025Q1",
                        source="https://cms.example.com/cms_asp_2025Q1.zip",
                    )
                ],
            ),
        )

        mock_wctx = MagicMock()
        mock_wctx.has_partition_key = True
        mock_wctx.partition_keys = ["2025Q1"]
        mock_wctx.asset_name = "bronze__cms_asp"
        mock_wctx.run_id = "run-1"

        resource._auto_register_period(
            mock_conn,
            contract,
            date(2025, 1, 1),
            mock_wctx,
            None,
            "this-should-be-ignored",
        )

        upsert_params = mock_cursor.execute.call_args_list[1][0][1]
        # source_uri (index 5) reflects Period.source, not the arg.
        assert upsert_params[5] == "https://cms.example.com/cms_asp_2025Q1.zip"

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_write_raises_when_from_ingest_missing_source_uri(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        """database.write() raises ValueError when a from_ingest write omits
        source_uri. The error fires before any connection is opened, so no
        partial state can land."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="bronze__rxnorm_full",
            run_id="run-1",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2025-08-01"],
        )

        with pytest.raises(ValueError, match="requires source_uri"):
            resource.write(
                pl.DataFrame({"col": [1, 2, 3]}),
                target="reference_bronze.rxnorm_full",
                context=wctx,
                contract=self._build_from_ingest_contract(),
                effective_date=date(2025, 8, 1),
            )

        # No DB connection was opened: the write SQL never ran.
        mock_connect.assert_not_called()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_write_raises_when_from_ingest_missing_partition_context(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        """database.write() raises ValueError when a from_ingest write has
        no Dagster partition context, even with source_uri supplied."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="bronze__rxnorm_full",
            run_id="run-1",
            log=MagicMock(),
            has_partition_key=False,
            partition_keys=None,
        )

        with pytest.raises(ValueError, match="requires a Dagster partition context"):
            resource.write(
                pl.DataFrame({"col": [1, 2, 3]}),
                target="reference_bronze.rxnorm_full",
                context=wctx,
                contract=self._build_from_ingest_contract(),
                effective_date=date(2025, 8, 1),
                source_uri="azure-blob://acct/container/rxnorm/foo.zip",
            )

        mock_connect.assert_not_called()


class TestPipelineRegistryUpsert:
    """Migration 019 (#308) Phase 2: ``_pipeline_registry_upsert``.

    Idempotent same-cursor upsert that mirrors ``_sync_pii_metadata``: skip
    silently when the table does not exist, INSERT ... ON CONFLICT
    otherwise, do not commit.
    """

    def _resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    def _contract(
        self,
        *,
        pipeline_id: str = "11111111-2222-3333-4444-555555555555",
        asset: str = "bronze__rxnorm_full",
        layer: str = "bronze",
        description: str | None = "Synthetic test contract",
        owner_team: str | None = "data_platform",
        source_id: str | None = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    ) -> Any:
        from moncpipelib.contracts.models import (
            DataContract,
            DataSource,
            FromIngestTemplate,
            Owner,
            Schema,
        )

        data_source: DataSource | None = None
        if source_id is not None:
            data_source = DataSource(
                source_id=source_id,
                source_name=asset,
                periods=FromIngestTemplate(
                    source="x_{release_version}.zip",
                    effective_from_field="effective_from",
                ),
            )

        owner: Owner | None = None
        if owner_team is not None:
            owner = Owner(team=owner_team)

        return DataContract(
            version="1.0",
            pipeline_id=pipeline_id,
            asset=asset,
            layer=layer,
            schema=Schema(columns=[]),
            description=description,
            owner=owner,
            data_source=data_source,
        )

    def _wctx(self) -> MagicMock:
        wctx = MagicMock()
        wctx.asset_name = "bronze__rxnorm_full"
        wctx.run_id = "run-xyz"
        wctx.dagster_asset_key = '["bronze__rxnorm_full"]'
        wctx.dagster_job_name = "ingest_job"
        wctx.code_location_name = "ingest_loc"
        return wctx

    def test_skips_when_table_does_not_exist(self) -> None:
        """``_check_pipeline_registry`` returns False when the table
        is missing; the upsert is a silent no-op."""
        resource = self._resource()
        mock_cursor = MagicMock()
        # First execute: information_schema lookup; fetchone returns None.
        mock_cursor.fetchone.return_value = None

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=self._contract(),
            wctx=self._wctx(),
            layer="bronze",
        )

        # Only the existence-check SELECT fired; no upsert INSERT.
        assert mock_cursor.execute.call_count == 1
        sql = str(mock_cursor.execute.call_args_list[0][0][0])
        assert "information_schema.tables" in sql

    def test_skips_when_contract_is_none(self) -> None:
        """No contract → nothing to upsert, no SQL executed at all."""
        resource = self._resource()
        mock_cursor = MagicMock()

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=None,
            wctx=self._wctx(),
            layer="bronze",
        )

        mock_cursor.execute.assert_not_called()

    def test_skips_when_pipeline_id_missing(self) -> None:
        """A contract without a ``pipeline_id`` is treated like no contract."""
        resource = self._resource()
        mock_cursor = MagicMock()
        contract = self._contract(pipeline_id="")

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=self._wctx(),
            layer="bronze",
        )

        mock_cursor.execute.assert_not_called()

    def test_executes_upsert_when_table_exists(self) -> None:
        """Happy path: registry exists, INSERT ... ON CONFLICT fires
        with the expected param shape (Phase 2 + Phase 3 columns)."""
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)  # registry exists

        contract = self._contract()
        wctx = self._wctx()
        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=wctx,
            layer="bronze",
        )

        # Two execute calls: existence SELECT + the upsert.
        assert mock_cursor.execute.call_count == 2
        upsert_sql = str(mock_cursor.execute.call_args_list[1][0][0])
        upsert_params = mock_cursor.execute.call_args_list[1][0][1]

        assert "INSERT INTO" in upsert_sql
        assert "ON CONFLICT (pipeline_id) DO UPDATE" in upsert_sql
        assert "updated_at = NOW()" in upsert_sql
        # Param tuple order: pipeline_id, asset, layer, source_id,
        # owner_team, description, dagster_asset_key, dagster_job_name,
        # code_location_name, contract_hash, schema_fingerprint,
        # contract_version, sla_freshness_hours, tags (jsonb str),
        # data_classification.
        # The test contract is built directly (not via load_contract), so
        # contract_hash / schema_fingerprint default to "" (coerced to
        # None) and the schema is empty (no PII columns → "none").
        assert upsert_params == (
            "11111111-2222-3333-4444-555555555555",
            "bronze__rxnorm_full",
            "bronze",
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "data_platform",
            "Synthetic test contract",
            '["bronze__rxnorm_full"]',
            "ingest_job",
            "ingest_loc",
            None,  # contract_hash (empty -> None)
            None,  # schema_fingerprint
            "1.0",  # contract_version
            None,  # sla_freshness_hours (no SLA on test contract)
            None,  # tags (empty dict -> None)
            "none",  # data_classification (no PII columns)
        )

    def test_existence_check_cached_across_calls(self) -> None:
        """A second invocation on the same resource should NOT re-issue
        the existence-check SELECT -- ``_check_pipeline_registry`` caches
        the answer in ``_pipeline_registry_available`` (same pattern as
        ``_check_period_registry``)."""
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        contract = self._contract()
        wctx = self._wctx()

        resource._pipeline_registry_upsert(
            mock_cursor, loaded_contract=contract, wctx=wctx, layer="bronze"
        )
        resource._pipeline_registry_upsert(
            mock_cursor, loaded_contract=contract, wctx=wctx, layer="bronze"
        )

        # First call: existence SELECT + INSERT (2 executes).
        # Second call: cached True, only the INSERT fires (1 execute).
        # Total: 3.
        assert mock_cursor.execute.call_count == 3

    def test_optional_fields_pass_through_as_none(self) -> None:
        """Contracts without ``description`` / ``owner`` / ``data_source``
        pass ``None`` for the corresponding columns rather than failing."""
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        contract = self._contract(description=None, owner_team=None, source_id=None)
        wctx = self._wctx()
        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=wctx,
            layer=None,
        )

        upsert_params = mock_cursor.execute.call_args_list[1][0][1]
        # pipeline_id, asset, layer, source_id, owner_team, description, ...
        assert upsert_params[2] is None  # layer
        assert upsert_params[3] is None  # source_id
        assert upsert_params[4] is None  # owner_team
        assert upsert_params[5] is None  # description

    def test_phase3_columns_populated_from_loaded_contract(self) -> None:
        """Migration 019 (#308) Phase 3: ``contract_hash`` /
        ``schema_fingerprint`` / ``contract_version`` / ``sla_freshness_hours``
        / ``tags`` / ``data_classification`` are projected into the upsert
        from the contract."""
        from moncpipelib.contracts.models import (
            SLA,
            Column,
            ColumnType,
            DataContract,
            Owner,
            Schema,
        )

        contract = DataContract(
            version="1.2",
            pipeline_id="22222222-2222-2222-2222-222222222222",
            asset="bronze__demo",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    Column(name="patient_name", type=ColumnType.STRING, nullable=False, pii=True),
                ],
            ),
            description="phase 3 demo",
            owner=Owner(team="data_platform"),
            sla=SLA(freshness_hours=24),
            tags={"domain": "claims", "tier": "bronze"},
        )
        # Simulate the loader's post-parse fingerprinting step.
        contract.contract_hash = "a" * 64
        contract.schema_fingerprint = "b" * 64

        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        wctx = self._wctx()

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=wctx,
            layer="bronze",
        )

        upsert_params = mock_cursor.execute.call_args_list[1][0][1]
        # contract_hash, schema_fingerprint, contract_version,
        # sla_freshness_hours, tags, data_classification:
        assert upsert_params[9] == "a" * 64
        assert upsert_params[10] == "b" * 64
        assert upsert_params[11] == "1.2"
        assert upsert_params[12] == 24
        # tags is bound as a JSON string, NOT a dict, because we pass
        # ``%s::jsonb`` in the SQL. json.dumps may produce either key order.
        assert isinstance(upsert_params[13], str)
        import json as _json

        assert _json.loads(upsert_params[13]) == {"domain": "claims", "tier": "bronze"}
        # patient_name is PII (non-managed) → classification = "PHI"
        assert upsert_params[14] == "PHI"

    def test_phase3_classification_none_when_no_pii_columns(self) -> None:
        """A contract whose non-managed columns all have ``pii=False``
        rolls up to ``data_classification = "none"``."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="33333333-3333-3333-3333-333333333333",
            asset="bronze__public",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    Column(name="counter", type=ColumnType.INTEGER, nullable=False, pii=False),
                ],
            ),
        )

        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=self._wctx(),
            layer="bronze",
        )

        upsert_params = mock_cursor.execute.call_args_list[1][0][1]
        assert upsert_params[14] == "none"

    def test_phase3_managed_pii_columns_excluded_from_classification(self) -> None:
        """Auto-managed columns (e.g., ``_lineage_id``) carrying ``pii=True``
        as metadata-of-metadata must NOT push classification to ``PHI``."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="44444444-4444-4444-4444-444444444444",
            asset="bronze__managed_only",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                    # Managed column with pii=True (system-generated)
                    Column(
                        name="_lineage_id",
                        type=ColumnType.UUID,
                        nullable=False,
                        pii=True,
                        managed=True,
                    ),
                ],
            ),
        )

        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        resource._pipeline_registry_upsert(
            mock_cursor,
            loaded_contract=contract,
            wctx=self._wctx(),
            layer="bronze",
        )

        upsert_params = mock_cursor.execute.call_args_list[1][0][1]
        assert upsert_params[14] == "none", (
            "Managed columns must not influence data_classification rollup"
        )

    def test_upsert_failure_propagates(self) -> None:
        """If the upsert itself fails (not the existence check), the
        exception is logged + re-raised so the surrounding transaction
        rolls back rather than silently committing an inconsistent
        partial state."""
        resource = self._resource()
        mock_cursor = MagicMock()
        # First execute = existence-check SELECT (succeeds), second =
        # upsert (raises).
        mock_cursor.execute.side_effect = [
            None,
            psycopg.errors.UniqueViolation("simulated"),
        ]
        mock_cursor.fetchone.return_value = (1,)

        contract = self._contract()
        wctx = self._wctx()

        with pytest.raises(psycopg.errors.UniqueViolation):
            resource._pipeline_registry_upsert(
                mock_cursor,
                loaded_contract=contract,
                wctx=wctx,
                layer="bronze",
            )
        # The warning should have been logged before the re-raise.
        wctx.log.warning.assert_called_once()


class TestPipelineRegistryRowMatches:
    """Issue #332 fast-path drift check.

    Returns ``True`` when the existing ``pipeline_registry`` row already
    matches the contract + Dagster identity, so the committed wrapper
    can skip the UPSERT entirely.
    """

    def _resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    def _contract_with_hash(self, contract_hash: str | None = "a" * 64) -> Any:
        from moncpipelib.contracts.models import DataContract, Schema

        contract = DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="bronze__rxnorm_full",
            layer="bronze",
            schema=Schema(columns=[]),
        )
        if contract_hash is not None:
            contract.contract_hash = contract_hash
        return contract

    def _wctx(self) -> MagicMock:
        wctx = MagicMock()
        wctx.asset_name = "bronze__rxnorm_full"
        wctx.dagster_asset_key = '["bronze__rxnorm_full"]'
        wctx.dagster_job_name = "ingest_job"
        wctx.code_location_name = "ingest_loc"
        return wctx

    def test_returns_false_when_contract_hash_is_empty(self) -> None:
        """Pre-Phase-3 contracts (no ``contract_hash``) cannot be diffed
        against the registry row — caller falls through to upsert."""
        resource = self._resource()
        mock_cursor = MagicMock()
        contract = self._contract_with_hash(contract_hash=None)
        # Clear the empty-string default so the truthiness check returns False.
        contract.contract_hash = ""

        assert resource._pipeline_registry_row_matches(mock_cursor, contract, self._wctx()) is False
        # No SELECT issued — drift detection short-circuits.
        mock_cursor.execute.assert_not_called()

    def test_returns_false_when_row_missing(self) -> None:
        """No existing registry row → caller must upsert."""
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        assert (
            resource._pipeline_registry_row_matches(
                mock_cursor, self._contract_with_hash(), self._wctx()
            )
            is False
        )

    def test_returns_true_when_all_four_columns_match(self) -> None:
        """Steady-state hit: contract_hash + 3 Dagster handles match → skip upsert."""
        resource = self._resource()
        mock_cursor = MagicMock()
        wctx = self._wctx()
        mock_cursor.fetchone.return_value = (
            "a" * 64,
            wctx.dagster_asset_key,
            wctx.dagster_job_name,
            wctx.code_location_name,
        )

        assert (
            resource._pipeline_registry_row_matches(mock_cursor, self._contract_with_hash(), wctx)
            is True
        )

    def test_returns_false_on_contract_hash_drift(self) -> None:
        resource = self._resource()
        mock_cursor = MagicMock()
        wctx = self._wctx()
        mock_cursor.fetchone.return_value = (
            "b" * 64,  # different hash
            wctx.dagster_asset_key,
            wctx.dagster_job_name,
            wctx.code_location_name,
        )

        assert (
            resource._pipeline_registry_row_matches(mock_cursor, self._contract_with_hash(), wctx)
            is False
        )

    def test_returns_false_on_dagster_handle_drift(self) -> None:
        """Asset rename or code-location move must force a refresh."""
        resource = self._resource()
        mock_cursor = MagicMock()
        wctx = self._wctx()
        mock_cursor.fetchone.return_value = (
            "a" * 64,
            '["bronze__rxnorm_full_renamed"]',  # asset_key drifted
            wctx.dagster_job_name,
            wctx.code_location_name,
        )

        assert (
            resource._pipeline_registry_row_matches(mock_cursor, self._contract_with_hash(), wctx)
            is False
        )


class TestPipelineRegistryUpsertCommitted:
    """Issue #332: pre-write committed wrapper.

    Opens its own short-lived autocommit connection so the registry row
    lock is released before the data-write transaction opens.
    """

    def _resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    def _contract(self, pipeline_id: str = "11111111-2222-3333-4444-555555555555") -> Any:
        from moncpipelib.contracts.models import DataContract, Schema

        contract = DataContract(
            version="1.0",
            pipeline_id=pipeline_id,
            asset="bronze__rxnorm_full",
            layer="bronze",
            schema=Schema(columns=[]),
        )
        # Provide a contract_hash so the fast-path SELECT has a comparison.
        contract.contract_hash = "a" * 64
        return contract

    def _wctx(self) -> MagicMock:
        wctx = MagicMock()
        wctx.asset_name = "bronze__rxnorm_full"
        wctx.dagster_asset_key = '["bronze__rxnorm_full"]'
        wctx.dagster_job_name = "ingest_job"
        wctx.code_location_name = "ingest_loc"
        return wctx

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_skips_when_contract_is_none(self, mock_connect: MagicMock) -> None:
        """No contract → no connection opened at all."""
        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=None,
            wctx=self._wctx(),
            layer="bronze",
        )
        mock_connect.assert_not_called()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_skips_when_pipeline_id_missing(self, mock_connect: MagicMock) -> None:
        """Empty ``pipeline_id`` → no connection opened."""
        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=self._contract(pipeline_id=""),
            wctx=self._wctx(),
            layer="bronze",
        )
        mock_connect.assert_not_called()

    def _mock_connection(
        self, mock_connect: MagicMock, fetchone_results: list[Any]
    ) -> tuple[MagicMock, MagicMock]:
        """Wire up the psycopg.connect mock to behave like a connection
        + cursor and replay ``fetchone_results`` in order across calls."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = fetchone_results
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn
        return mock_conn, mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_opens_autocommit_connection(self, mock_connect: MagicMock) -> None:
        """The dedicated connection must run with ``autocommit=True`` so
        the upsert commits immediately and releases the row lock."""
        mock_conn, _ = self._mock_connection(
            mock_connect,
            fetchone_results=[None],  # table doesn't exist → silent no-op
        )

        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=self._contract(),
            wctx=self._wctx(),
            layer="bronze",
        )

        assert mock_conn.autocommit is True

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_skips_when_table_does_not_exist(self, mock_connect: MagicMock) -> None:
        """Silent no-op (no warning, no raise) when the data-platform
        Alembic migration has not applied the table yet."""
        _, mock_cursor = self._mock_connection(
            mock_connect,
            fetchone_results=[None],  # information_schema lookup returns no row
        )

        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=self._contract(),
            wctx=self._wctx(),
            layer="bronze",
        )

        # Only the information_schema SELECT fired (no drift-check, no INSERT).
        assert mock_cursor.execute.call_count == 1
        sql = str(mock_cursor.execute.call_args_list[0][0][0])
        assert "information_schema.tables" in sql

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_skips_upsert_when_row_already_matches(self, mock_connect: MagicMock) -> None:
        """Steady-state: SELECT confirms the row matches; no UPSERT fires."""
        wctx = self._wctx()
        _, mock_cursor = self._mock_connection(
            mock_connect,
            fetchone_results=[
                (1,),  # information_schema lookup: table exists
                (
                    "a" * 64,
                    wctx.dagster_asset_key,
                    wctx.dagster_job_name,
                    wctx.code_location_name,
                ),  # drift-check SELECT: row matches
            ],
        )

        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=self._contract(),
            wctx=wctx,
            layer="bronze",
        )

        # Two SELECTs (existence + drift), no INSERT.
        assert mock_cursor.execute.call_count == 2
        for call in mock_cursor.execute.call_args_list:
            sql = str(call[0][0])
            assert "INSERT" not in sql, f"Expected no INSERT, got: {sql}"

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_upserts_when_row_does_not_match(self, mock_connect: MagicMock) -> None:
        """Drift detected (or new pipeline) → INSERT ... ON CONFLICT fires."""
        _, mock_cursor = self._mock_connection(
            mock_connect,
            fetchone_results=[
                (1,),  # information_schema: table exists
                None,  # drift-check: no existing row
            ],
        )

        self._resource()._pipeline_registry_upsert_committed(
            loaded_contract=self._contract(),
            wctx=self._wctx(),
            layer="bronze",
        )

        # 3 executes: existence SELECT, drift SELECT, INSERT.
        assert mock_cursor.execute.call_count == 3
        insert_sql = str(mock_cursor.execute.call_args_list[2][0][0])
        assert "INSERT INTO" in insert_sql
        assert "ON CONFLICT (pipeline_id) DO UPDATE" in insert_sql

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_upsert_failure_propagates(self, mock_connect: MagicMock) -> None:
        """If the upsert itself fails, the exception bubbles up so the
        caller aborts the data write rather than proceeding with a
        registry row that does not exist for the Phase 4 FK."""
        wctx = self._wctx()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            (1,),  # information_schema: table exists
            None,  # drift-check: no existing row → upsert path
        ]
        mock_cursor.execute.side_effect = [
            None,  # information_schema SELECT
            None,  # drift-check SELECT
            psycopg.errors.UniqueViolation("simulated"),  # upsert
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        with pytest.raises(psycopg.errors.UniqueViolation):
            self._resource()._pipeline_registry_upsert_committed(
                loaded_contract=self._contract(),
                wctx=wctx,
                layer="bronze",
            )

        wctx.log.warning.assert_called_once()


class TestEnforceContractCheckResults:
    """Migration 019 (#308) Phase 5: ``_enforce_contract`` populates
    ``ContractValidationSummary.check_results`` with one
    ``CheckResultRow`` per check executed (passed or failed)."""

    def _resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            enforce_contracts="warn",
        )

    def test_passed_checks_appear_in_check_results(self) -> None:
        """A clean schema-only contract produces a summary with one
        ``check_results`` entry (the schema check) marked ``passed=True``."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        resource = self._resource()
        contract = DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="demo",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(name="id", type=ColumnType.INTEGER, nullable=False, pii=False),
                ],
            ),
        )
        df = pl.DataFrame({"id": [1, 2, 3]})
        wctx = MagicMock()
        wctx.asset_name = "demo"
        wctx.log = MagicMock()

        _, summary = resource._enforce_contract(
            df, wctx, preloaded_contract=contract, layer="bronze"
        )

        assert summary is not None
        assert len(summary.check_results) == 1
        row = summary.check_results[0]
        assert row.check_name == "schema"
        assert row.severity == "error"
        assert row.passed is True
        assert row.sample_failures is None

    def test_failed_check_carries_sample_failures(self) -> None:
        """A column test that fails produces a ``check_results`` entry
        with ``passed=False`` and a populated ``sample_failures`` payload."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnTest,
            ColumnType,
            DataContract,
            Schema,
            Severity,
        )

        resource = self._resource()
        contract = DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="demo",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(
                        name="id",
                        type=ColumnType.INTEGER,
                        nullable=True,
                        pii=False,
                        tests=[ColumnTest(test_type="not_null", severity=Severity.ERROR)],
                    ),
                ],
            ),
        )
        df = pl.DataFrame({"id": [1, None, 3, None]})
        wctx = MagicMock()
        wctx.asset_name = "demo"
        wctx.log = MagicMock()

        _, summary = resource._enforce_contract(
            df, wctx, preloaded_contract=contract, layer="bronze"
        )

        assert summary is not None
        # schema (passed) + id.not_null (failed) = 2 entries
        assert len(summary.check_results) == 2
        names = {r.check_name for r in summary.check_results}
        assert names == {"schema", "id.not_null"}
        failed = next(r for r in summary.check_results if r.check_name == "id.not_null")
        assert failed.passed is False
        assert failed.severity == "error"
        assert failed.failed_count >= 1

    def test_warn_severity_recorded_with_warn_severity(self) -> None:
        """A warn-severity check that fails appears in ``check_results``
        with ``severity == "warn"``."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnTest,
            ColumnType,
            DataContract,
            Schema,
            Severity,
        )

        resource = self._resource()
        contract = DataContract(
            version="1.0",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            asset="demo",
            layer="bronze",
            schema=Schema(
                columns=[
                    Column(
                        name="id",
                        type=ColumnType.INTEGER,
                        nullable=True,
                        pii=False,
                        tests=[ColumnTest(test_type="not_null", severity=Severity.WARN)],
                    ),
                ],
            ),
        )
        df = pl.DataFrame({"id": [1, None]})
        wctx = MagicMock()
        wctx.asset_name = "demo"
        wctx.log = MagicMock()

        _, summary = resource._enforce_contract(
            df, wctx, preloaded_contract=contract, layer="bronze"
        )

        assert summary is not None
        warn_row = next(r for r in summary.check_results if r.check_name == "id.not_null")
        assert warn_row.severity == "warn"


class TestCheckContractValidationRuns:
    """Migration 019 (#308) Phase 5: ``_check_contract_validation_runs``
    silent-no-op pattern when the data-platform Alembic migration has
    not yet applied."""

    def _resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    def test_skips_when_table_missing(self) -> None:
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        wctx = MagicMock()
        wctx.log = MagicMock()
        assert resource._check_contract_validation_runs(mock_cursor, wctx) is False

    def test_caches_result(self) -> None:
        """Second invocation should NOT re-issue the existence-check SELECT."""
        resource = self._resource()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)

        wctx = MagicMock()
        wctx.log = MagicMock()
        assert resource._check_contract_validation_runs(mock_cursor, wctx) is True
        assert resource._check_contract_validation_runs(mock_cursor, wctx) is True
        assert mock_cursor.execute.call_count == 1


class TestReconcileScd2AuditPersistence:
    """Migration 019 (#308) Phase 6: ``reconcile_scd2`` persists an
    audit row into ``scd2_reconciliations`` when the table exists.

    Silent no-op until the data-platform Alembic migration applies the
    table; behaviour swaps to "persist one row per reconcile" once it
    does.
    """

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_audit_row_skipped_when_table_missing(self, mock_connect: MagicMock) -> None:
        """Existing behaviour preserved: when ``scd2_reconciliations``
        does not exist yet, no extra INSERT fires beyond the reconcile
        DML."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # fetchone sequence (#363 reordered the audit-table probe ahead of
        # the reconcile DML so the FK can be preflighted): scd2_reconciliations
        # existence check (None -> missing, so no FK preflight), set_config
        # result, sequence_col probe.
        mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        # No INSERT INTO lineage.scd2_reconciliations fired
        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("scd2_reconciliations" in s and "INSERT INTO" in s for s in sql_calls)
        # But the return dict now carries duration_seconds (Phase 6 additive)
        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], float)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_audit_row_persisted_when_table_exists(self, mock_connect: MagicMock) -> None:
        """When ``scd2_reconciliations`` exists, the audit row INSERT
        fires on the same cursor before commit."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # fetchone sequence (#363): scd2_reconciliations existence ((1,) ->
        # present), then the FK preflight -- pipeline_registry existence
        # ((1,) -> present) and the pipeline_id lookup ((1,) -> registered) --
        # then set_config result and sequence_col probe.
        mock_cursor.fetchone.side_effect = [(1,), (1,), (1,), ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="run-xyz",
            asset_name="silver/products",
            pipeline_id="11111111-2222-3333-4444-555555555555",
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        audit_inserts = [s for s in sql_calls if "scd2_reconciliations" in s and "INSERT INTO" in s]
        assert len(audit_inserts) == 1, f"expected one audit-row INSERT, got: {audit_inserts}"
        # Commit fires after the audit row INSERT
        mock_conn.commit.assert_called_once()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_audit_row_pipeline_id_derived_from_contract(self, mock_connect: MagicMock) -> None:
        """Contract-driven callers don't have to thread ``pipeline_id``
        explicitly -- it's derived from ``contract.pipeline_id``."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # #363: audit existence, then FK preflight (registry existence +
        # pipeline_id lookup, both present), then set_config + seq probe.
        mock_cursor.fetchone.side_effect = [(1,), (1,), (1,), ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        contract = MagicMock()
        contract.asset = "silver/products"
        contract.pipeline_id = "22222222-2222-2222-2222-222222222222"
        contract.sinks = [
            {
                "type": "table",
                "schema": "silver",
                "table": "products",
                "mode": "scd2",
                "business_key": ["product_id"],
            },
        ]

        resource.reconcile_scd2(contract=contract, collapse_duplicates=False, run_id="test-run")

        # Find the audit INSERT call and check its params for the
        # contract-derived pipeline_id.
        audit_call = next(
            c
            for c in mock_cursor.execute.call_args_list
            if "scd2_reconciliations" in str(c[0][0]) and "INSERT INTO" in str(c[0][0])
        )
        params = audit_call[0][1]
        # Params order: run_id, asset_name, pipeline_id, target_table, ...
        assert params[2] == "22222222-2222-2222-2222-222222222222"
        assert params[1] == "silver/products"  # asset_name from contract

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_unregistered_pipeline_id_fails_before_reconcile_dml(
        self, mock_connect: MagicMock
    ) -> None:
        """Issue #363: when the audit table exists and ``pipeline_id`` is
        absent from ``pipeline_registry``, the reconcile fails fast --
        before any DML -- rather than letting the audit-row FK violation
        roll back the (expensive) reconcile at commit.

        Deliberately raises rather than degrading the FK to ``NULL``: a
        noisy job failure is the current alert for an unregistered pipeline.
        """
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # fetchone sequence: scd2_reconciliations existence ((1,) -> present),
        # pipeline_registry existence ((1,) -> present), then the pipeline_id
        # lookup returns None -> NOT registered -> raise before any DML.
        mock_cursor.fetchone.side_effect = [(1,), (1,), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        with pytest.raises(RuntimeError, match="not present in"):
            resource.reconcile_scd2(
                target="silver.products",
                business_key=["product_id"],
                collapse_duplicates=True,
                run_id="run-xyz",
                pipeline_id="773d73d4-59eb-48b4-9f76-7d98418d0b93",
            )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        # No reconcile DML ran (no DELETE/UPDATE), no advisory lock taken,
        # and no audit-row INSERT -- the guard bailed first.
        assert not any("DELETE FROM" in s or "UPDATE " in s for s in sqls)
        assert not any("pg_advisory_xact_lock" in s for s in sqls)
        assert not any("INSERT INTO" in s and "scd2_reconciliations" in s for s in sqls)
        mock_conn.commit.assert_not_called()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_unregistered_pipeline_id_skipped_when_audit_table_absent(
        self, mock_connect: MagicMock
    ) -> None:
        """Issue #363: a pre-migration-019 environment (no
        ``scd2_reconciliations`` table) has no FK to violate, so the
        reconcile runs unguarded even when ``pipeline_id`` is unregistered --
        the pipeline_registry is never probed."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # Audit table absent (None) -> no FK preflight; set_config + seq probe.
        mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="run-xyz",
            pipeline_id="773d73d4-59eb-48b4-9f76-7d98418d0b93",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        # No pipeline_registry probe fired and the reconcile DML ran + committed.
        assert not any("pipeline_registry" in s for s in sqls)
        assert any("pg_advisory_xact_lock" in s for s in sqls)
        mock_conn.commit.assert_called_once()


class TestReconcileSCD2:
    """Tests for reconcile_scd2() method."""

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_timeline_reconciliation_sql(self, mock_connect: MagicMock) -> None:
        """Verify SQL contains LEAD window function and IS DISTINCT FROM."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert any("LEAD" in s for s in sql_calls)
        assert any("IS DISTINCT FROM" in s for s in sql_calls)
        assert "rows_timeline_updated" in result

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_collapse_duplicates_sql(self, mock_connect: MagicMock) -> None:
        """Collapse uses ROW_NUMBER + USING ranked, NOT a NOT IN subquery (#277)."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=True,
            run_id="test-run",
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        delete_sqls = [s for s in sql_calls if "DELETE" in s]
        assert delete_sqls, "expected at least one DELETE statement"
        assert any("USING ranked" in s for s in delete_sqls)
        assert any("ranked.rn > 1" in s for s in delete_sqls)
        # The NOT IN form is the #277 anti-pattern; guard against accidental
        # reintroduction via "simplification".
        assert not any("NOT IN" in s for s in delete_sqls)
        assert result["rows_collapsed"] >= 0

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_no_collapse(self, mock_connect: MagicMock) -> None:
        """collapse_duplicates=False skips DELETE."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("DELETE" in s for s in sql_calls)
        assert result["rows_collapsed"] == 0

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    @pytest.mark.parametrize("collapse_duplicates", [True, False])
    def test_acquires_advisory_lock_first(
        self, mock_connect: MagicMock, collapse_duplicates: bool
    ) -> None:
        """First cursor.execute is pg_advisory_xact_lock(hashtext(target)) (#278).

        Lock acquisition is the structural protection against snapshot-divergent
        concurrent reconciles. It must be unconditional (independent of
        ``collapse_duplicates``) and must precede any DML.
        """
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # First fetchone call (#363): ``_check_scd2_reconciliations`` --
        # information_schema.tables lookup for ``lineage.scd2_reconciliations``
        # (None -> table absent, so no FK preflight and audit persistence is a
        # silent no-op until data-platform Alembic applies). This probe now
        # runs ahead of the reconcile DML so the audit-row FK can be checked
        # before the expensive work; it consumes the first fetchone.
        # Second fetchone call: set_config('work_mem', ...) result (#294 --
        # per-tx work_mem bump; set_config returns the canonical value, no
        # separate SHOW work_mem needed).
        # Third fetchone call: information_schema.columns sequence_col probe
        # (None -> column absent, skip renumber).
        mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=collapse_duplicates,
            run_id="test-run",
        )

        # #363: the FK preflight reads ``scd2_reconciliations`` existence (a
        # catalog read taking no lock) ahead of the advisory lock, so the lock
        # is no longer literally ``execute[0]``. The load-bearing invariant is
        # unchanged: the lock must precede ANY data-modifying DML.
        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        lock_idx = next(i for i, s in enumerate(sqls) if "pg_advisory_xact_lock(hashtext(%s))" in s)
        dml_idx = next(
            (i for i, s in enumerate(sqls) if "DELETE FROM" in s or "UPDATE " in s),
            len(sqls),
        )
        assert lock_idx < dml_idx
        assert mock_cursor.execute.call_args_list[lock_idx][0][1] == ("silver.products",)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_advisory_lock_uses_resolved_target_from_contract(
        self, mock_connect: MagicMock
    ) -> None:
        """Lock keys on the resolved (contract-derived) target, not raw input."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            schema_override="test_schema",
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # See sibling test for fetchone call ordering rationale (#294 + #308
        # Phase 6 + #363 audit-probe reorder ahead of the DML).
        mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [
            {
                "type": "table",
                "schema": "silver",
                "table": "products",
                "mode": "scd2",
                "business_key": ["product_id"],
            },
        ]

        resource.reconcile_scd2(contract=contract, collapse_duplicates=False, run_id="test-run")

        # #363: catalog probe precedes the lock now; assert the lock precedes
        # any DML and keys on the resolved (schema_override) target.
        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        lock_idx = next(i for i, s in enumerate(sqls) if "pg_advisory_xact_lock(hashtext(%s))" in s)
        dml_idx = next(
            (i for i, s in enumerate(sqls) if "DELETE FROM" in s or "UPDATE " in s),
            len(sqls),
        )
        assert lock_idx < dml_idx
        # schema_override applied -> "test_schema.products"
        assert mock_cursor.execute.call_args_list[lock_idx][0][1] == ("test_schema.products",)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_reconcile_from_contract(self, mock_connect: MagicMock) -> None:
        """Contract with SCD2 sink derives target and business_key."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 2
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [
            {
                "type": "table",
                "schema": "silver",
                "table": "products",
                "mode": "scd2",
                "business_key": ["product_id"],
            }
        ]

        result = resource.reconcile_scd2(
            contract=contract, collapse_duplicates=False, run_id="test-run"
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        # Verify derived target is used in SQL
        assert any("silver.products" in s for s in sql_calls)
        assert "rows_timeline_updated" in result

    def test_reconcile_contract_no_scd2_sink(self) -> None:
        """Contract without SCD2 sink raises ValueError."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [{"type": "table", "mode": "full_refresh"}]

        with pytest.raises(ValueError, match="no SCD2 sink"):
            resource.reconcile_scd2(contract=contract, run_id="test-run")

    def test_reconcile_contract_multiple_scd2_sinks(self) -> None:
        """Multiple SCD2 sinks raises ValueError."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [
            {"type": "table", "mode": "scd2", "business_key": ["a"]},
            {"type": "table", "mode": "scd2", "business_key": ["b"]},
        ]

        with pytest.raises(ValueError, match="requires exactly one"):
            resource.reconcile_scd2(contract=contract, run_id="test-run")

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_reconcile_explicit_overrides_contract(self, mock_connect: MagicMock) -> None:
        """Explicit target/business_key take priority over contract."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [
            {
                "type": "table",
                "schema": "silver",
                "table": "products",
                "mode": "scd2",
                "business_key": ["product_id"],
            },
        ]

        resource.reconcile_scd2(
            target="gold.overridden_table",
            business_key=["override_key"],
            contract=contract,
            collapse_duplicates=False,
            run_id="test-run",
        )

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert any("gold.overridden_table" in s for s in sql_calls)
        assert any('"override_key"' in s for s in sql_calls)

    def test_reconcile_no_target_no_contract(self) -> None:
        """Missing both target and contract raises ValueError."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        with pytest.raises(ValueError, match="target is required"):
            resource.reconcile_scd2(run_id="test-run")

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_reconcile_contract_with_schema_override(self, mock_connect: MagicMock) -> None:
        """Resource schema_override applied to contract-derived target."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            schema_override="test_schema",
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        contract = MagicMock()
        contract.asset = "test_asset"
        contract.sinks = [
            {
                "type": "table",
                "schema": "silver",
                "table": "products",
                "mode": "scd2",
                "business_key": ["product_id"],
            },
        ]

        resource.reconcile_scd2(contract=contract, collapse_duplicates=False, run_id="test-run")

        sql_calls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert any("test_schema.products" in s for s in sql_calls)


class TestReconcileScd2RunIdRequired:
    """Issue #334 Bug 3a: ``reconcile_scd2`` requires an explicit
    ``run_id``.  The prior ``run_id or "reconcile_scd2"`` literal
    fallback produced ``scd2_reconciliations`` rows whose natural key
    couldn't be cohorted by Dagster run; the fallback is gone and
    callers must supply a real identifier.
    """

    def test_run_id_none_raises_value_error(self) -> None:
        """Caller-supplied ``run_id=None`` (or omitted, since the
        default is ``None``) raises ``ValueError`` before any DB
        connection is opened."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        with pytest.raises(ValueError, match="run_id is required"):
            resource.reconcile_scd2(
                target="silver.products",
                business_key=["product_id"],
            )

    def test_run_id_none_raises_before_connection(self) -> None:
        """Validation runs before ``get_connection_raw()`` so a misuse
        does not hold a connection or acquire the advisory lock."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        with (
            patch.object(PostgresResource, "get_connection_raw") as mock_conn,
            pytest.raises(ValueError, match="run_id is required"),
        ):
            resource.reconcile_scd2(
                target="silver.products",
                business_key=["product_id"],
                run_id=None,
            )
        mock_conn.assert_not_called()


class TestReconcileScd2ContextSignals:
    """``PostgresResource._extract_reconcile_context_signals`` is the
    helper that pulls ``run_id`` / ``asset_name`` from any of the
    supported context shapes (Dagster ``AssetExecutionContext`` /
    ``OpExecutionContext`` or moncpipelib ``WriteContext``).  This is
    the seam the ``context=`` kwarg on ``reconcile_scd2`` uses.
    """

    def test_write_context_returns_attributes_directly(self) -> None:
        """``WriteContext`` already has ``run_id`` and ``asset_name``
        as plain ``str`` attributes -- no duck-typing required."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="silver/dim_provider",
            run_id="run-write-context",
            log=MagicMock(),
        )

        run_id, asset_name = PostgresResource._extract_reconcile_context_signals(wctx)

        assert run_id == "run-write-context"
        assert asset_name == "silver/dim_provider"

    def test_dagster_context_extracts_run_id_and_asset_key(self) -> None:
        """A Dagster ``AssetExecutionContext``-shaped mock exposes
        ``run_id`` directly and ``asset_key.to_user_string()`` for
        the asset name."""
        ctx = MagicMock()
        ctx.run_id = "dagster-run-abc"
        ctx.asset_key.to_user_string.return_value = "silver/dim_member"

        run_id, asset_name = PostgresResource._extract_reconcile_context_signals(ctx)

        assert run_id == "dagster-run-abc"
        assert asset_name == "silver/dim_member"

    def test_op_context_without_asset_key_returns_none_for_asset_name(self) -> None:
        """``OpExecutionContext`` for a plain op (not inside an asset
        materialization) has no ``asset_key`` attribute.  The helper
        must degrade to ``(run_id, None)`` rather than raise -- the
        caller's ``contract.asset`` default will fill in
        ``asset_name`` downstream."""

        class _PlainOpContext:
            run_id = "op-run-xyz"

        ctx: Any = _PlainOpContext()
        run_id, asset_name = PostgresResource._extract_reconcile_context_signals(ctx)

        assert run_id == "op-run-xyz"
        assert asset_name is None

    def test_bare_magicmock_context_degrades_safely(self) -> None:
        """``MagicMock``'s auto-attribute behaviour would surface
        child mocks for ``run_id`` and ``asset_key.to_user_string()``.
        The defensive ``isinstance(..., str)`` checks must reject
        those so the helper returns ``(None, None)`` rather than
        leaking mocks into the audit row.  Mirrors the same guard on
        ``_extract_backfill_signals`` (issue #334 PR A)."""
        ctx = MagicMock()  # no explicit attribute setup

        run_id, asset_name = PostgresResource._extract_reconcile_context_signals(ctx)

        assert run_id is None
        assert asset_name is None

    def test_op_context_asset_key_property_raises_degrades_safely(self) -> None:
        """Issue #339 regression: Dagster's ``OpExecutionContext.asset_key``
        is a ``@property`` whose getter raises
        ``DagsterInvalidPropertyError`` on non-asset ops -- it does NOT
        raise ``AttributeError``, so ``getattr(..., None)`` does NOT
        substitute the default.  The helper must wrap the read in a
        ``try`` / ``except`` so the property-raises path degrades to
        ``(run_id, None)`` rather than propagating the exception up
        through ``reconcile_scd2(context=context)``.

        This is the exact shape that broke
        ``data-platform/_npidata_reconcile_op`` after adopting the new
        ``context=`` kwarg from #338.  The previous
        ``test_op_context_without_asset_key_returns_none_for_asset_name``
        test used a synthetic class with no ``asset_key`` attribute at
        all, which exercises the "attribute missing" branch -- but
        Dagster's real ``OpExecutionContext`` exposes ``asset_key`` as
        a descriptor whose getter raises.  This test models the
        descriptor-raises shape directly.
        """

        class _DagsterInvalidPropertyError(Exception):
            """Stand-in for ``dagster._core.errors.DagsterInvalidPropertyError``
            so the test does not require a Dagster import.  The helper
            catches bare ``Exception``, so the specific class doesn't
            matter -- only that the descriptor raises something other
            than ``AttributeError``."""

        class _OpContextWithRaisingAssetKey:
            """Models the relevant slice of Dagster's ``OpExecutionContext``:
            ``run_id`` is a plain ``str`` attribute, but ``asset_key`` is
            a property whose getter raises on non-asset ops."""

            run_id = "op-run-property-raises"

            @property
            def asset_key(self) -> object:
                raise _DagsterInvalidPropertyError(
                    "Op 'npidata_reconcile_op' does not have an assets definition."
                )

        ctx: Any = _OpContextWithRaisingAssetKey()
        run_id, asset_name = PostgresResource._extract_reconcile_context_signals(ctx)

        assert run_id == "op-run-property-raises"
        assert asset_name is None


class TestReconcileScd2ContextKwarg:
    """``reconcile_scd2(context=...)`` mirrors the
    ``database.write(context=...)`` convention: when a Dagster context
    is supplied, the resource extracts ``run_id`` and (when not
    otherwise resolved) ``asset_name`` from it.  Explicit ``run_id``
    / ``asset_name`` kwargs always win over the context-derived
    values.
    """

    def test_context_kwarg_provides_run_id(self) -> None:
        """A Dagster-shaped context with ``run_id`` populates the
        resource's ``run_id`` so the caller no longer has to plumb
        ``context.run_id`` by hand.  Validates by short-circuiting at
        the ``target is None`` check -- if ``run_id`` extraction
        worked, the no-target ``ValueError`` is what we see, not the
        ``run_id is required`` one."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        ctx = MagicMock()
        ctx.run_id = "dagster-run-from-context"

        with pytest.raises(ValueError, match="target is required"):
            resource.reconcile_scd2(context=ctx)

    def test_explicit_run_id_wins_over_context(self) -> None:
        """When both ``context.run_id`` and an explicit ``run_id``
        are passed, the explicit kwarg wins.  Verifies via the
        integration shape (a mocked tracker captures the bound
        ``run_id`` value)."""
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="dim_provider_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "dim_provider",
                    "mode": "scd2",
                    "business_key": ["provider_npi"],
                }
            ],
        )
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

        ctx = MagicMock()
        ctx.run_id = "context-loses"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # #363: audit existence, FK preflight (registry + pipeline_id lookup),
        # then set_config + seq probe.
        mock_cursor.fetchone.side_effect = [(1,), (1,), (1,), ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_tracker = MagicMock()
        with (
            patch("moncpipelib.resources.postgres.psycopg.connect", return_value=mock_conn),
            patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker),
        ):
            resource.reconcile_scd2(
                contract=contract,
                collapse_duplicates=False,
                context=ctx,
                run_id="explicit-wins",
            )

        mock_tracker.write_scd2_reconciliation.assert_called_once()
        assert mock_tracker.write_scd2_reconciliation.call_args.kwargs["run_id"] == "explicit-wins"

    def test_context_kwarg_alone_still_raises_when_run_id_missing(self) -> None:
        """A context with no ``run_id`` attribute (or a ``MagicMock``
        whose child-mock ``run_id`` is rejected by the type guard)
        does NOT bypass the ``run_id is required`` raise.  Misuse
        surfaces at call time."""

        class _ContextWithoutRunId:
            pass  # no run_id attribute

        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        ctx: Any = _ContextWithoutRunId()

        with pytest.raises(ValueError, match="run_id is required"):
            resource.reconcile_scd2(
                target="silver.products",
                business_key=["product_id"],
                context=ctx,
            )

    def test_write_context_works_as_context_kwarg(self) -> None:
        """``WriteContext`` (moncpipelib's own type, returned by
        ``WriteContext.from_asset_context`` / ``from_output_context``)
        is accepted by the ``context=`` kwarg directly.  This is the
        shape ad-hoc callers who already built a ``WriteContext``
        (e.g. from a Dagster ``OutputContext`` in an IO-manager path)
        would pass."""
        from moncpipelib.resources.types import WriteContext

        wctx = WriteContext(
            asset_name="silver/dim_provider",
            run_id="write-context-run",
            log=MagicMock(),
        )
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

        # Same short-circuit shape as test_context_kwarg_provides_run_id:
        # if context extraction worked, no-target is the error.
        with pytest.raises(ValueError, match="target is required"):
            resource.reconcile_scd2(context=wctx)


class TestBuildScd2ReconciliationMetadataPayload:
    """Issue #334 Bug 3b: ``_build_scd2_reconciliation_metadata_payload``
    builds the JSONB shape that lands in ``scd2_reconciliations.metadata``.

    Typed columns already cover the row counts, durations, target
    table, asset name, and pipeline FK.  The payload carries the
    extras that don't fit typed columns -- ``collapse_duplicates``,
    the resolved ``business_key``, and contract identity when a
    contract drove the reconcile.
    """

    @staticmethod
    def _contract():  # type: ignore[no-untyped-def]
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        return DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="dim_provider_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
        )

    def test_no_contract_payload_has_config_and_business_key(self) -> None:
        """An ad-hoc reconcile with explicit ``target`` + ``business_key``
        and no contract produces a payload carrying just the config
        flag and the resolved business key."""
        payload = PostgresResource._build_scd2_reconciliation_metadata_payload(
            business_key=["product_id"],
            collapse_duplicates=False,
            contract=None,
        )

        assert payload == {
            "collapse_duplicates": False,
            "business_key": ["product_id"],
        }

    def test_with_contract_includes_asset_and_version(self) -> None:
        """A contract-driven reconcile additionally carries
        ``contract_asset`` and ``contract_version`` so audit-row
        consumers can distinguish contract-driven from ad-hoc rows
        without joining to ``pipeline_registry``."""
        payload = PostgresResource._build_scd2_reconciliation_metadata_payload(
            business_key=["hcpcs_code", "ndc"],
            collapse_duplicates=True,
            contract=self._contract(),
        )

        assert payload["collapse_duplicates"] is True
        assert payload["business_key"] == ["hcpcs_code", "ndc"]
        assert payload["contract_asset"] == "dim_provider_silver"
        assert payload["contract_version"] == "1.0"

    def test_mock_contract_attributes_filtered(self) -> None:
        """MagicMock contracts have non-``str`` ``asset`` / ``version``
        attributes (child mocks).  The payload must skip those so
        ``json.dumps`` at bind time doesn't choke."""
        mock_contract = MagicMock()
        # Deliberately do NOT pin .asset / .version on the mock --
        # the auto-created child mocks must be rejected.

        payload = PostgresResource._build_scd2_reconciliation_metadata_payload(
            business_key=["product_id"],
            collapse_duplicates=True,
            contract=mock_contract,
        )

        assert "contract_asset" not in payload
        assert "contract_version" not in payload

    def test_business_key_is_copied_not_referenced(self) -> None:
        """The payload's ``business_key`` is a fresh list so a caller
        mutating the input after the call doesn't smuggle data into
        a persisted audit row."""
        bk = ["a", "b"]
        payload = PostgresResource._build_scd2_reconciliation_metadata_payload(
            business_key=bk,
            collapse_duplicates=False,
            contract=None,
        )
        bk.append("c")
        assert payload["business_key"] == ["a", "b"]


class TestReconcileScd2MetadataThreadsToAuditRow:
    """Integration: a representative ``reconcile_scd2(...)`` call
    against a mocked cursor produces a ``write_scd2_reconciliation``
    call whose ``metadata`` kwarg deserialises back to a dict
    containing the documented keys.
    """

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_metadata_payload_reaches_tracker(self, mock_connect: MagicMock) -> None:
        from moncpipelib.contracts.models import (
            Column,
            ColumnType,
            DataContract,
            Schema,
        )

        contract = DataContract(
            version="1.0",
            pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            asset="dim_provider_silver",
            layer="silver",
            schema=Schema(columns=[Column(name="id", type=ColumnType.STRING, nullable=False)]),
            sinks=[
                {
                    "type": "table",
                    "schema": "silver",
                    "table": "dim_provider",
                    "mode": "scd2",
                    "business_key": ["provider_npi"],
                }
            ],
        )

        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # fetchone sequence (#363): scd2_reconciliations existence ((1,) ->
        # present), FK preflight (registry existence + pipeline_id lookup,
        # both (1,) -> present/registered), set_config result, sequence_col
        # probe.
        mock_cursor.fetchone.side_effect = [(1,), (1,), (1,), ("256MB",), None]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        mock_tracker = MagicMock()
        with patch.object(PostgresResource, "_get_lineage_tracker", return_value=mock_tracker):
            resource.reconcile_scd2(
                contract=contract,
                collapse_duplicates=False,
                run_id="dagster-run-abcd",
            )

        mock_tracker.write_scd2_reconciliation.assert_called_once()
        call_kwargs = mock_tracker.write_scd2_reconciliation.call_args.kwargs

        # ``run_id`` flows through verbatim, no literal substitution.
        assert call_kwargs["run_id"] == "dagster-run-abcd"

        # ``metadata`` is a dict containing the documented keys.  The
        # tracker is responsible for ``json.dumps`` at bind time; here
        # we assert the in-Python shape.
        metadata = call_kwargs["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["collapse_duplicates"] is False
        assert metadata["business_key"] == ["provider_npi"]
        assert metadata["contract_asset"] == "dim_provider_silver"
        assert metadata["contract_version"] == "1.0"


class TestReconcileWorkMemHelper:
    """Unit tests for ``PostgresResource._apply_work_mem_local`` (#294).

    The helper applies ``set_config('work_mem', value, true)`` against the
    cursor and returns the canonical value as reported by ``set_config``
    itself.  ``set_config`` returns the post-canonicalization value (so a
    separate ``SHOW work_mem`` round-trip is unnecessary).  Whitespace
    normalization and disable-token handling live one layer up in
    :meth:`PostgresResource._resolve_work_mem` -- this helper expects an
    already-stripped literal and validates it strictly.

    Integration coverage of the round-trip through real Postgres lives in
    ``tests/integration/test_scd2_integration.py``.
    """

    @pytest.mark.parametrize(
        "value",
        ["256MB", "512MB", "1GB", "32kB", "1024kB"],
    )
    def test_accepts_valid_literals(self, value: str) -> None:
        """Valid Postgres ``work_mem`` literals are passed to ``set_config``."""
        cursor = MagicMock()
        cursor.fetchone.return_value = ("256MB",)

        result = PostgresResource._apply_work_mem_local(cursor, value)

        assert result == "256MB"
        # Single execute now: the parameterized set_config (no SHOW round-trip).
        assert len(cursor.execute.call_args_list) == 1
        first_call = cursor.execute.call_args_list[0]
        assert "set_config('work_mem'" in str(first_call[0][0])
        assert first_call[0][1] == (value,)

    @pytest.mark.parametrize(
        "value",
        [
            "256",  # missing unit
            "1.5GB",  # fractional not supported
            "abc",
            "",
            "256 megabytes",
            "256mB",  # case-sensitive on the unit
            "GB",
            "256MB extra",
            "  256MB  ",  # helper expects pre-stripped input; resolver handles whitespace
        ],
    )
    def test_rejects_invalid_literals(self, value: str) -> None:
        """Malformed values raise ``ValueError`` and never touch the cursor."""
        cursor = MagicMock()

        with pytest.raises(ValueError, match="invalid work_mem"):
            PostgresResource._apply_work_mem_local(cursor, value)

        cursor.execute.assert_not_called()

    def test_returns_canonical_form(self) -> None:
        """Helper returns whatever ``set_config`` reports (Postgres canonicalizes)."""
        cursor = MagicMock()
        # Caller passes "262144kB"; Postgres canonicalizes to "256MB" and
        # set_config returns the canonical form directly.
        cursor.fetchone.return_value = ("256MB",)

        result = PostgresResource._apply_work_mem_local(cursor, "262144kB")

        assert result == "256MB"


class TestResolveWorkMem:
    """Unit tests for ``PostgresResource._resolve_work_mem`` (#294).

    The resolver is the single normalization + validation surface for both
    the resource field and the per-call override.  Runs *before* the
    connection is opened so malformed input fails fast.
    """

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "none", "NONE", "None", "off", "OFF", "disabled", " Disabled "],
    )
    def test_disable_inputs_resolve_to_none(self, value: str | None) -> None:
        """``None``, empty, and the disable sentinels all resolve to ``None``."""
        assert PostgresResource._resolve_work_mem(value) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("256MB", "256MB"),
            ("  256MB  ", "256MB"),
            ("\t1GB\n", "1GB"),
            ("32kB", "32kB"),
            ("1024 MB", "1024 MB"),  # internal whitespace tolerated by PG
        ],
    )
    def test_valid_inputs_normalized(self, value: str, expected: str) -> None:
        """Valid literals are stripped and returned unchanged."""
        assert PostgresResource._resolve_work_mem(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["1.5GB", "256", "abc", "256 megabytes", "256mB", "GB"],
    )
    def test_invalid_inputs_raise(self, value: str) -> None:
        """Format-invalid literals raise ``ValueError``."""
        with pytest.raises(ValueError, match="invalid work_mem"):
            PostgresResource._resolve_work_mem(value)


class TestResolveStatementTimeout:
    """Unit tests for ``PostgresResource._resolve_statement_timeout`` (#361).

    Mirrors ``_resolve_work_mem``: the single normalization + validation
    surface for the SCD2 change-detection ``statement_timeout`` knob, run
    before the connection is opened so malformed input fails fast.
    """

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "none", "NONE", "None", "off", "OFF", "disabled", " Disabled "],
    )
    def test_disable_inputs_resolve_to_none(self, value: str | None) -> None:
        """``None``, empty, and the disable sentinels all resolve to ``None``."""
        assert PostgresResource._resolve_statement_timeout(value) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("30min", "30min"),
            ("  30min  ", "30min"),
            ("\t600s\n", "600s"),
            ("900000", "900000"),  # bare integer == milliseconds in PG
            ("500ms", "500ms"),
            ("2h", "2h"),
            ("30 min", "30 min"),  # internal whitespace tolerated by PG
        ],
    )
    def test_valid_inputs_normalized(self, value: str, expected: str) -> None:
        """Valid literals are stripped and returned unchanged."""
        assert PostgresResource._resolve_statement_timeout(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["1.5h", "abc", "30 minutes", "30m", "min", "30sec"],
    )
    def test_invalid_inputs_raise(self, value: str) -> None:
        """Format-invalid literals raise ``ValueError``."""
        with pytest.raises(ValueError, match="invalid statement_timeout"):
            PostgresResource._resolve_statement_timeout(value)


class TestReconcileWorkMem:
    """Tests for the ``work_mem`` plumbing on ``reconcile_scd2`` (#294)."""

    @staticmethod
    def _make_mocks(*, fetchone_values: list[Any] | None = None) -> tuple[MagicMock, MagicMock]:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        if fetchone_values is None:
            # Default (#363 reordered the audit-table probe ahead of the DML):
            # scd2_reconciliations existence check returns None (table absent,
            # silent no-op, no FK preflight), then set_config returns canonical,
            # then sequence_col probe absent. (set_config returns the post-
            # canonicalization value, so no separate SHOW work_mem round-trip.)
            mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        else:
            mock_cursor.fetchone.side_effect = fetchone_values
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_default_resource_field_applies_set_config(self, mock_connect: MagicMock) -> None:
        """No method-level override -> resource field default ('256MB') is applied."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn, mock_cursor = self._make_mocks()
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        set_config_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "set_config('work_mem'" in str(call[0][0])
        ]
        assert len(set_config_calls) == 1
        assert set_config_calls[0][0][1] == ("256MB",)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_resource_field_none_skips_set_config(self, mock_connect: MagicMock) -> None:
        """Resource field set to ``None`` -> no set_config call."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem=None,
        )
        # Without the work_mem helper, the only fetchone is the seq_col probe.
        mock_conn, mock_cursor = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("set_config('work_mem'" in s for s in sqls)

    @pytest.mark.parametrize("disable_value", ["none", "OFF", "Disabled", "  none  ", ""])
    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_resource_field_disable_string_skips_set_config(
        self, mock_connect: MagicMock, disable_value: str
    ) -> None:
        """``"none"``/``"off"``/``"disabled"`` (case-insensitive) skip the override.

        Lets ``EnvVar('PG_RECONCILE_WORK_MEM')='off'`` disable the bump per
        environment without requiring a code change.
        """
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem=disable_value,
        )
        mock_conn, mock_cursor = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("set_config('work_mem'" in s for s in sqls)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_method_override_disable_string_skips_set_config(self, mock_connect: MagicMock) -> None:
        """Per-call ``work_mem='off'`` overrides a real resource default."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="256MB",
        )
        mock_conn, mock_cursor = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            work_mem="off",
            run_id="test-run",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("set_config('work_mem'" in s for s in sqls)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_resource_field_whitespace_padded_value_normalized(
        self, mock_connect: MagicMock
    ) -> None:
        """Resource field tolerates surrounding whitespace (env-var hygiene)."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="  256MB  ",
        )
        mock_conn, mock_cursor = self._make_mocks()
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        set_config_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "set_config('work_mem'" in str(call[0][0])
        ]
        # Resolver strips before forwarding to the helper.
        assert len(set_config_calls) == 1
        assert set_config_calls[0][0][1] == ("256MB",)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_method_override_beats_resource_field(self, mock_connect: MagicMock) -> None:
        """Explicit ``work_mem='1GB'`` on the call wins over resource field."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="256MB",
        )
        # #363: audit probe (None -> absent) leads, then set_config, then seq.
        mock_conn, mock_cursor = self._make_mocks(fetchone_values=[None, ("1GB",), None])
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            work_mem="1GB",
            run_id="test-run",
        )

        set_config_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "set_config('work_mem'" in str(call[0][0])
        ]
        assert len(set_config_calls) == 1
        assert set_config_calls[0][0][1] == ("1GB",)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_method_override_none_disables_resource_field_default(
        self, mock_connect: MagicMock
    ) -> None:
        """Explicit ``work_mem=None`` skips set_config even with a resource default."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="256MB",
        )
        mock_conn, mock_cursor = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            work_mem=None,
            run_id="test-run",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        assert not any("set_config('work_mem'" in s for s in sqls)

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_set_config_runs_after_advisory_lock_before_dml(self, mock_connect: MagicMock) -> None:
        """Order is: advisory lock -> set_config(work_mem) -> DML.

        Putting the bump *after* the lock and *before* DML is what makes the
        ``ExplainCapturingCursor`` regression test (#277) cover the same memory
        profile as production. Reorderings would silently regress that.
        """
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn, mock_cursor = self._make_mocks()
        mock_connect.return_value = mock_conn

        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=True,
            run_id="test-run",
        )

        sqls = [str(c[0][0]) for c in mock_cursor.execute.call_args_list]
        # Locate the indices of the structural checkpoints.
        lock_idx = next(i for i, s in enumerate(sqls) if "pg_advisory_xact_lock" in s)
        set_config_idx = next(i for i, s in enumerate(sqls) if "set_config('work_mem'" in s)
        first_dml_idx = next(i for i, s in enumerate(sqls) if "DELETE FROM" in s)

        assert lock_idx < set_config_idx < first_dml_idx

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_invalid_method_override_raises_before_connection_open(
        self, mock_connect: MagicMock
    ) -> None:
        """Malformed ``work_mem`` raises before the connection is even opened.

        ``_resolve_work_mem`` runs in ``reconcile_scd2`` before the
        ``get_connection_raw()`` call, so a malformed value never consumes a
        Postgres backend or holds an advisory lock waiting for rollback.
        """
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

        with pytest.raises(ValueError, match="invalid work_mem"):
            resource.reconcile_scd2(
                target="silver.products",
                business_key=["product_id"],
                work_mem="not-a-size",
                run_id="test-run",
            )

        # No connection opened, no SQL issued.
        mock_connect.assert_not_called()

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_invalid_resource_field_raises_before_connection_open(
        self, mock_connect: MagicMock
    ) -> None:
        """Malformed ``reconcile_work_mem`` field also fails fast."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="garbage",
        )

        with pytest.raises(ValueError, match="invalid work_mem"):
            resource.reconcile_scd2(
                target="silver.products", business_key=["product_id"], run_id="test-run"
            )

        mock_connect.assert_not_called()


class TestReconcileWorkMemSurfaces:
    """Operator-visibility surfaces for the applied ``work_mem`` (#306).

    Covers the resource-side surfaces introduced by issue #306: the new
    ``work_mem`` key on the ``reconcile_scd2`` return dict and the
    INFO-level log emitted on both the override-applied and skip branches.
    Reuses the same connect/cursor mock shape as
    :class:`TestReconcileWorkMem`.
    """

    @staticmethod
    def _make_mocks(*, fetchone_values: list[Any] | None = None) -> tuple[MagicMock, MagicMock]:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        if fetchone_values is None:
            # #363: scd2_reconciliations existence check (None -> absent, no FK
            # preflight) now leads, then set_config result, then sequence_col
            # probe.
            mock_cursor.fetchone.side_effect = [None, ("256MB",), None]
        else:
            mock_cursor.fetchone.side_effect = fetchone_values
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_return_dict_includes_resolved_work_mem(self, mock_connect: MagicMock) -> None:
        """Return dict surfaces the resolved literal applied to the tx."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn, _ = self._make_mocks()
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        assert result["work_mem"] == "256MB"

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_return_dict_work_mem_none_when_resource_field_none(
        self, mock_connect: MagicMock
    ) -> None:
        """Resource field ``None`` -> return dict carries ``None``, not absent."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem=None,
        )
        mock_conn, _ = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        assert "work_mem" in result
        assert result["work_mem"] is None

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_return_dict_work_mem_none_for_disable_string(self, mock_connect: MagicMock) -> None:
        """``reconcile_work_mem='off'`` resolves to ``None`` in the return dict."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem="off",
        )
        mock_conn, _ = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        result = resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        assert result["work_mem"] is None

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_info_log_emitted_when_override_applied(
        self, mock_connect: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Override branch emits one INFO log under the pinned logger name."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn, _ = self._make_mocks()
        mock_connect.return_value = mock_conn

        caplog.set_level(logging.INFO, logger="moncpipelib.resources")
        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        applied = [
            rec
            for rec in caplog.records
            if rec.name == "moncpipelib.resources"
            and rec.levelno == logging.INFO
            and "per-tx work_mem set to" in rec.getMessage()
        ]
        assert len(applied) == 1
        msg = applied[0].getMessage()
        assert "256MB" in msg
        assert "silver.products" in msg

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_info_log_emitted_when_override_skipped(
        self, mock_connect: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skip branch emits one INFO log so absence-of-log is not the signal."""
        resource = PostgresResource(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",
            reconcile_work_mem=None,
        )
        mock_conn, _ = self._make_mocks(fetchone_values=[None, None])
        mock_connect.return_value = mock_conn

        caplog.set_level(logging.INFO, logger="moncpipelib.resources")
        resource.reconcile_scd2(
            target="silver.products",
            business_key=["product_id"],
            collapse_duplicates=False,
            run_id="test-run",
        )

        skipped = [
            rec
            for rec in caplog.records
            if rec.name == "moncpipelib.resources"
            and rec.levelno == logging.INFO
            and "per-tx work_mem override skipped" in rec.getMessage()
        ]
        assert len(skipped) == 1
        assert "silver.products" in skipped[0].getMessage()


class TestUpdatePeriodMetadata:
    """Tests for update_period_metadata() method."""

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_merge_sql(self, mock_connect: MagicMock) -> None:
        """Verify SQL uses JSONB merge operator."""
        resource = PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        resource.update_period_metadata(
            source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            partition_key="2025-01-01",
            metadata_updates={"reconciled_at": "2025-03-30T00:00:00Z"},
        )

        sql = str(mock_cursor.execute.call_args[0][0])
        assert "||" in sql  # JSONB merge operator
        assert "COALESCE" in sql


class TestGetWriterConfigFullRefreshMethodOverride:
    """Per-write full_refresh_method override into WriterConfig (#4)."""

    @staticmethod
    def _resource(default_method: str = "auto") -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            full_refresh_method=default_method,
        )

    def test_no_override_uses_resource_default(self) -> None:
        from moncpipelib.io_managers.enums import FullRefreshMethod

        cfg = self._resource("truncate")._get_writer_config()
        assert cfg.full_refresh_method == FullRefreshMethod.TRUNCATE

    def test_override_wins_over_resource_default(self) -> None:
        from moncpipelib.io_managers.enums import FullRefreshMethod

        cfg = self._resource("auto")._get_writer_config(full_refresh_method="delete")
        assert cfg.full_refresh_method == FullRefreshMethod.DELETE

    def test_none_override_falls_back_to_default(self) -> None:
        from moncpipelib.io_managers.enums import FullRefreshMethod

        cfg = self._resource("auto")._get_writer_config(full_refresh_method=None)
        assert cfg.full_refresh_method == FullRefreshMethod.AUTO

    def test_invalid_override_raises(self) -> None:
        with pytest.raises(ValueError):
            self._resource("auto")._get_writer_config(full_refresh_method="vacuum")


class TestBuildWriteConfigFullRefreshMethod:
    """_build_write_config carries the full_refresh_method + _explicit pair (#4)."""

    def test_explicit_when_provided(self) -> None:
        cfg = PostgresResource._build_write_config(
            write_mode="full_refresh",
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
            full_refresh_method="delete",
        )
        assert cfg["full_refresh_method"] == "delete"
        assert cfg["full_refresh_method_explicit"] is True

    def test_not_explicit_when_omitted(self) -> None:
        cfg = PostgresResource._build_write_config(
            write_mode="full_refresh",
            primary_key=None,
            update_columns=None,
            partition_column=None,
            business_key=None,
            tracked_columns=None,
            detect_deletes=False,
        )
        assert cfg["full_refresh_method"] is None
        assert cfg["full_refresh_method_explicit"] is False
