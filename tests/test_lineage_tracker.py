"""Tests for lineage tracking functionality."""

import uuid
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.config import LineageDefaults
from moncpipelib.lineage import (
    LineageTracker,
    extract_timestamp_from_uuid7,
    generate_lineage_key,
    generate_uuid7,
    parse_lineage_key,
)


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    return MagicMock()


@pytest.fixture
def tracker(mock_engine):
    """Create a LineageTracker with mocked engine."""
    return LineageTracker(mock_engine)


def test_attach_lineage_to_dataframe(tracker):
    """Test attaching lineage columns to dataframe."""
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    lineage_id = str(uuid.uuid4())
    lineage_key = "v1:test:bronze:2024-01-15:abc123"

    result = tracker.attach_lineage_to_dataframe(df, lineage_id, lineage_key)

    assert LineageDefaults.ID_COLUMN in result.columns
    assert LineageDefaults.KEY_COLUMN in result.columns
    assert result[LineageDefaults.ID_COLUMN].unique().to_list() == [lineage_id]
    assert result[LineageDefaults.KEY_COLUMN].unique().to_list() == [lineage_key]
    assert len(result) == 3


def test_get_parent_lineage_ids(tracker):
    """Test extracting parent lineage IDs from dataframe."""
    lineage_ids = [str(uuid.uuid4()) for _ in range(3)]
    df = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5],
            LineageDefaults.ID_COLUMN: lineage_ids + [lineage_ids[0], lineage_ids[1]],  # Duplicates
        }
    )

    result = tracker.get_parent_lineage_ids(df)

    assert len(result) == 3  # Should deduplicate
    assert set(result) == set(lineage_ids)


def test_get_parent_lineage_ids_missing_column(tracker):
    """Test error when dataframe lacks lineage ID column."""
    df = pl.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(ValueError, match=LineageDefaults.ID_COLUMN):
        tracker.get_parent_lineage_ids(df)


def test_create_lineage_record_basic(tracker, mock_engine):
    """Test creating a basic lineage record."""
    # Mock the database connection and execution
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    lineage_id, lineage_key = tracker.create_lineage_record(
        run_id="test-run-123",
        asset_name="test_asset",
        layer="bronze",
        source_file="test.csv",
        row_count=100,
    )

    # Verify UUID format
    assert isinstance(uuid.UUID(lineage_id), uuid.UUID)

    # Verify lineage key format
    assert lineage_key.startswith("v1:")
    assert ":bronze:" in lineage_key

    # Verify SQL was executed
    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args

    # Verify parameters
    params = call_args[0][1]
    assert params["run_id"] == "test-run-123"
    assert params["asset_name"] == "test_asset"
    assert params["layer"] == "bronze"
    assert params["source_file"] == "test.csv"
    assert params["row_count"] == 100
    # Phase 2 defaults: caller did not pass backfill_id; column must still
    # appear in the INSERT params bound to NULL.
    assert params["backfill_id"] is None
    assert params["is_backfill"] is False


def test_create_lineage_record_with_date_range(tracker, mock_engine):
    """Test creating lineage record with date range."""
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 31)

    tracker.create_lineage_record(
        run_id="test-run-123",
        asset_name="test_asset",
        layer="bronze",
        data_date_range=(start_date, end_date),
    )

    params = mock_conn.execute.call_args[0][1]
    assert params["data_date_range"] == "[2024-01-01,2024-01-31]"


def test_create_lineage_record_backfill(tracker, mock_engine):
    """Test creating lineage record for backfill."""
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    replaces_id = str(uuid.uuid4())

    tracker.create_lineage_record(
        run_id="test-run-123",
        asset_name="test_asset",
        layer="silver",
        is_backfill=True,
        backfill_reason="Data quality fix",
        replaces_lineage_id=replaces_id,
    )

    params = mock_conn.execute.call_args[0][1]
    assert params["is_backfill"] is True
    assert params["backfill_reason"] == "Data quality fix"
    assert params["replaces_lineage_id"] == replaces_id


def test_create_lineage_record_with_backfill_id(tracker, mock_engine):
    """Phase 2: ``backfill_id`` is bound to the INSERT params and stored
    alongside ``is_backfill``."""
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    tracker.create_lineage_record(
        run_id="test-run-123",
        asset_name="test_asset",
        layer="silver",
        is_backfill=True,
        backfill_id="bf_2026_05_22_claims",
    )

    params = mock_conn.execute.call_args[0][1]
    assert params["is_backfill"] is True
    assert params["backfill_id"] == "bf_2026_05_22_claims"


def test_create_lineage_record_insert_sql_includes_backfill_id(tracker, mock_engine):
    """The generated INSERT statement must reference the new ``backfill_id``
    column. Pins the SQL surface so an accidental removal of the column from
    the INSERT statement is caught even if no test passes a non-default
    value."""
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    tracker.create_lineage_record(
        run_id="r",
        asset_name="a",
        layer="bronze",
    )

    sql_text = mock_conn.execute.call_args[0][0]
    # ``sa.text(...)`` round-trips to a ``TextClause``; the underlying SQL
    # is reachable via ``str()``.
    sql_str = str(sql_text)
    assert "backfill_id" in sql_str
    assert ":backfill_id" in sql_str


def test_data_lineage_model_has_backfill_id_column():
    """Migration 018 Phase 2 schema check: the ``DataLineage`` SQLAlchemy
    model must expose a nullable ``backfill_id`` column of type ``Text``.

    Pins both the column's presence and its nullability so a fresh
    ``Base.metadata.create_all()`` in dev/test environments produces a
    schema compatible with the production ``ALTER TABLE`` runbook step.
    """
    from sqlalchemy import Text

    from moncpipelib.lineage.models import DataLineage

    columns = DataLineage.__table__.columns
    assert "backfill_id" in columns, "Phase 2 must add a backfill_id column"

    col = columns["backfill_id"]
    assert col.nullable is True, "backfill_id must be nullable (no default)"
    # Compare via isinstance because SQLAlchemy may wrap the type.
    assert isinstance(col.type, Text), f"backfill_id should be Text, got {col.type!r}"


# ---------------------------------------------------------------------------
# Migration 018 Phase 3: split API for same-txn lineage INSERT
# ---------------------------------------------------------------------------


class TestGenerateLineageIds:
    """``LineageTracker.generate_lineage_ids`` is the side-effect-free
    half of the create / write split: it generates the UUID7 + composite
    key client-side so the caller can attach them to the DataFrame and
    issue the INSERT later, inside the same transaction as the data DML.
    """

    def test_returns_uuid_and_v1_key(self):
        """Returns a tuple of ``(lineage_id, lineage_key)`` where the id
        is a valid UUID and the key has the v1 composite format."""
        from moncpipelib.lineage.tracker import LineageTracker

        mock_engine = MagicMock()
        tracker = LineageTracker(mock_engine)

        lineage_id, lineage_key = tracker.generate_lineage_ids(
            asset_name="claims_silver",
            layer="silver",
            run_id="run-abc-123",
            data_date=date(2024, 1, 15),
        )

        assert isinstance(uuid.UUID(lineage_id), uuid.UUID)
        assert lineage_key.startswith("v1:")
        assert ":silver:" in lineage_key
        assert ":2024-01-15:" in lineage_key

    def test_is_pure_does_not_touch_engine(self):
        """The hot invariant for Phase 3: ``generate_lineage_ids`` must
        not call ``engine.begin()`` / ``engine.connect()`` / ``execute``.
        A mock engine with all DB hooks set to raise verifies this."""
        from moncpipelib.lineage.tracker import LineageTracker

        mock_engine = MagicMock()
        mock_engine.begin.side_effect = AssertionError(
            "generate_lineage_ids must not open a transaction"
        )
        mock_engine.connect.side_effect = AssertionError(
            "generate_lineage_ids must not open a connection"
        )
        mock_engine.execute.side_effect = AssertionError(
            "generate_lineage_ids must not execute SQL"
        )

        tracker = LineageTracker(mock_engine)
        tracker.generate_lineage_ids(
            asset_name="a",
            layer="bronze",
            run_id="r",
        )

    def test_generated_ids_are_unique(self):
        """Successive calls produce distinct UUIDs (UUID7 carries
        timestamp + random)."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        a, _ = tracker.generate_lineage_ids(asset_name="x", layer="bronze", run_id="r")
        b, _ = tracker.generate_lineage_ids(asset_name="x", layer="bronze", run_id="r")
        assert a != b


class TestWriteLineageRecord:
    """``LineageTracker.write_lineage_record`` is the cursor-bound half:
    it executes the INSERT on a supplied psycopg cursor without
    committing, so the caller can sequence it with the data DML inside
    a single transaction.
    """

    def test_executes_on_cursor_without_commit(self):
        """``write_lineage_record`` must call ``cursor.execute(sql, params)``
        exactly once and must NOT call ``cursor.connection.commit()``."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        mock_cursor = MagicMock()

        tracker.write_lineage_record(
            mock_cursor,
            lineage_id="00000000-0000-0000-0000-000000000001",
            lineage_key="v1:claims:bronze:2026-05-15:run-1",
            run_id="run-1",
            asset_name="claims_bronze",
            layer="bronze",
            source_file="claims.csv",
            row_count=100,
            is_backfill=True,
            backfill_id="bf_abc",
        )

        assert mock_cursor.execute.call_count == 1
        # The supplied cursor's connection must NOT be touched -- commit
        # is the caller's responsibility.
        mock_cursor.connection.commit.assert_not_called()
        mock_cursor.connection.rollback.assert_not_called()

    def test_uses_psycopg_placeholder_dialect(self):
        """``write_lineage_record`` targets a psycopg cursor so the SQL
        must use ``%(name)s`` placeholders, not SQLAlchemy ``:name``.
        psycopg cursors with the SA dialect would raise at execute time."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        mock_cursor = MagicMock()

        tracker.write_lineage_record(
            mock_cursor,
            lineage_id="lid",
            lineage_key="lkey",
            run_id="r",
            asset_name="a",
            layer="bronze",
        )

        sql = mock_cursor.execute.call_args[0][0]
        assert "%(lineage_id)s" in sql
        assert "%(backfill_id)s" in sql
        # No SQLAlchemy-style placeholders should leak in.
        assert ":lineage_id" not in sql

    def test_does_not_use_engine(self):
        """``write_lineage_record`` must not touch ``self.engine`` -- it
        works purely on the supplied cursor. A tracker with an engine
        that raises on every attribute access verifies this."""
        from moncpipelib.lineage.tracker import LineageTracker

        engine = MagicMock()
        engine.begin.side_effect = AssertionError(
            "write_lineage_record must not open a tracker-internal txn"
        )
        engine.connect.side_effect = AssertionError(
            "write_lineage_record must not open a tracker-internal connection"
        )

        tracker = LineageTracker(engine)
        mock_cursor = MagicMock()
        tracker.write_lineage_record(
            mock_cursor,
            lineage_id="lid",
            lineage_key="lkey",
            run_id="r",
            asset_name="a",
            layer="bronze",
        )

    def test_metadata_is_dumped_to_json_string(self):
        """Issue #334 Bug 2: ``metadata`` is bound as a JSON string, not
        the raw dict. psycopg3 has no default ``dict → jsonb`` adapter,
        and the SQL casts ``%(metadata)s::jsonb`` -- so the bound value
        must already be a string for Postgres to parse."""
        import json

        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        mock_cursor = MagicMock()

        payload = {
            "write_mode": "full_refresh",
            "contract_enforcement": "error",
            "contract_status": "passed",
        }

        tracker.write_lineage_record(
            mock_cursor,
            lineage_id="lid",
            lineage_key="lkey",
            run_id="r",
            asset_name="a",
            layer="bronze",
            metadata=payload,
        )

        bound = mock_cursor.execute.call_args[0][1]
        assert isinstance(bound["metadata"], str)
        assert json.loads(bound["metadata"]) == payload

    def test_metadata_none_passes_through_as_none(self):
        """When the caller passes ``metadata=None`` (the existing
        default) the bound value must be ``None`` -- not the string
        ``"null"``. ``None`` round-trips to SQL NULL on both dialects;
        ``"null"`` would store the JSON null token in jsonb."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        mock_cursor = MagicMock()

        tracker.write_lineage_record(
            mock_cursor,
            lineage_id="lid",
            lineage_key="lkey",
            run_id="r",
            asset_name="a",
            layer="bronze",
        )

        bound = mock_cursor.execute.call_args[0][1]
        assert bound["metadata"] is None

    def test_lineage_insert_sql_casts_metadata_to_jsonb(self):
        """Regression guard: ``_lineage_insert_sql`` must wrap the
        ``metadata`` placeholder in ``(...)::jsonb`` on both dialects.
        Without the cast the ``jsonb`` column rejects the string at
        execute time; without the wrapping parens SA's named-param
        parser silently truncates ``:metadata`` to ``:metada`` -- see
        ``test_sa_text_binds_metadata_cast_round_trip`` below."""
        from moncpipelib.lineage.tracker import LineageTracker

        psycopg_sql = LineageTracker._lineage_insert_sql(dialect="psycopg")
        sa_sql = LineageTracker._lineage_insert_sql(dialect="sa")

        assert "(%(metadata)s)::jsonb" in psycopg_sql
        assert "(:metadata)::jsonb" in sa_sql

    def test_sa_text_binds_metadata_cast_round_trip(self):
        """Issue #334 Bug 2 regression: SA's ``text()`` named-param
        parser scans ``:name`` greedily and a bare ``:foo::jsonb``
        truncates the placeholder name to ``fo`` (confirmed on
        SQLAlchemy 2.x). The wrapping form ``(:foo)::jsonb`` pins the
        boundary cleanly. This test would fail loudly if a future
        refactor reverts to the bare-suffix form."""
        import json

        import sqlalchemy as sa

        from moncpipelib.lineage.tracker import LineageTracker

        sql = LineageTracker._lineage_insert_sql(dialect="sa")
        clause = sa.text(sql)

        param_names = {p.key for p in clause._bindparams.values()}
        assert "metadata" in param_names
        # Truncated forms ("metada") or absorbed-cast forms
        # ("metadata::jsonb") would both fail this guard.
        assert "metada" not in param_names
        assert not any("jsonb" in name for name in param_names)

        # Round-trip a real bind through to confirm SA accepts it.
        bound = clause.bindparams(
            lineage_id="lid",
            lineage_key="lkey",
            run_id="r",
            asset_name="a",
            pipeline_id=None,
            layer="bronze",
            source_file=None,
            source_system=None,
            data_date=None,
            data_date_range=None,
            row_count=0,
            is_backfill=False,
            backfill_reason=None,
            backfill_id=None,
            replaces_lineage_id=None,
            parent_lineage_ids=None,
            transformation_type=None,
            metadata=json.dumps({"write_mode": "full_refresh"}),
        )
        assert bound is not None


class TestFindPriorLineageId:
    """Migration 018 Phase 4: ``find_prior_lineage_id`` returns the
    most-recent prior ``data_lineage`` row matching the in-flight
    write's ``(asset_name, layer[, partition])`` tuple.
    """

    def test_whole_table_lookup_matches_null_partition_rows(self):
        """No ``data_date`` and no ``data_date_range`` → SQL must filter
        for rows with both columns NULL (whole-table chain)."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()
        cursor.fetchone.return_value = ("00000000-0000-0000-0000-000000000001",)

        result = tracker.find_prior_lineage_id(cursor, asset_name="foo", layer="bronze")

        assert result == "00000000-0000-0000-0000-000000000001"
        sql, params = cursor.execute.call_args[0]
        assert "data_date IS NULL" in sql
        assert "data_date_range IS NULL" in sql
        assert params == {"asset_name": "foo", "layer": "bronze"}

    def test_single_date_lookup_filters_data_date(self):
        """``data_date`` set → SQL filters ``data_date = %s``."""
        from datetime import date

        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()
        cursor.fetchone.return_value = ("00000000-0000-0000-0000-000000000002",)

        result = tracker.find_prior_lineage_id(
            cursor,
            asset_name="foo",
            layer="silver",
            data_date=date(2026, 5, 15),
        )

        assert result == "00000000-0000-0000-0000-000000000002"
        sql, params = cursor.execute.call_args[0]
        assert "data_date = %(data_date)s" in sql
        assert "data_date_range" not in sql
        assert params == {
            "asset_name": "foo",
            "layer": "silver",
            "data_date": date(2026, 5, 15),
        }

    def test_date_range_lookup_filters_data_date_range(self):
        """``data_date_range`` set → SQL filters ``data_date_range = %s``."""
        from datetime import date

        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()
        cursor.fetchone.return_value = ("00000000-0000-0000-0000-000000000003",)

        result = tracker.find_prior_lineage_id(
            cursor,
            asset_name="foo",
            layer="silver",
            data_date_range=(date(2026, 5, 13), date(2026, 5, 15)),
        )

        assert result == "00000000-0000-0000-0000-000000000003"
        sql, params = cursor.execute.call_args[0]
        assert "data_date_range = %(data_date_range)s" in sql
        assert "data_date = " not in sql
        assert params["data_date_range"] == "[2026-05-13,2026-05-15]"

    def test_no_prior_row_returns_none(self):
        """``cursor.fetchone()`` → ``None`` means no prior row matched.
        Helper returns ``None`` and the calling code leaves
        ``replaces_lineage_id = NULL``."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()
        cursor.fetchone.return_value = None

        result = tracker.find_prior_lineage_id(cursor, asset_name="foo", layer="bronze")
        assert result is None

    def test_sql_uses_max_processed_at_via_order_by(self):
        """The lookup must order by ``processed_at DESC LIMIT 1`` so
        the planner can use the new composite index."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        tracker.find_prior_lineage_id(cursor, asset_name="foo", layer="bronze")

        sql, _ = cursor.execute.call_args[0]
        assert "ORDER BY processed_at DESC" in sql
        assert "LIMIT 1" in sql

    def test_uses_supplied_cursor_not_engine(self):
        """``find_prior_lineage_id`` must NOT touch ``self.engine`` -- it
        runs on the supplied cursor inside the caller's transaction."""
        from moncpipelib.lineage.tracker import LineageTracker

        engine = MagicMock()
        engine.begin.side_effect = AssertionError(
            "find_prior_lineage_id must not open a transaction"
        )
        engine.connect.side_effect = AssertionError(
            "find_prior_lineage_id must not open a connection"
        )

        tracker = LineageTracker(engine)
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        tracker.find_prior_lineage_id(cursor, asset_name="foo", layer="bronze")


class TestDataLineageIndexes:
    """Migration 018 Phase 4 schema check: composite index exists."""

    def test_asset_layer_processed_index_present(self):
        """The new composite index must be declared in the SA metadata
        so a fresh ``create_all()`` in dev/test environments produces
        it. Production install uses ``CREATE INDEX CONCURRENTLY`` per
        the Phase 7 runbook."""
        from moncpipelib.lineage.models import DataLineage

        index_names = {idx.name for idx in DataLineage.__table__.indexes}
        assert "ix_lineage_data_lineage_asset_layer_processed" in index_names


class TestUpdateParentLineageIds:
    """Migration 018 Phase 5: ``update_parent_lineage_ids`` is the
    cursor-bound post-DML UPDATE that amends ``parent_lineage_ids`` on
    an already-inserted lineage row. The batched-write path needs this
    because the parent set is only complete after every batch has been
    iterated.
    """

    def test_executes_update_on_cursor_without_commit(self):
        """The UPDATE runs on the supplied cursor and does NOT touch
        ``cursor.connection`` -- the caller owns the transaction."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()

        tracker.update_parent_lineage_ids(
            cursor,
            lineage_id="00000000-0000-0000-0000-000000000001",
            parent_lineage_ids=["aaa", "bbb"],
        )

        assert cursor.execute.call_count == 1
        sql, params = cursor.execute.call_args[0]
        assert "UPDATE lineage.data_lineage" in sql
        assert "SET parent_lineage_ids = %(parent_lineage_ids)s" in sql
        assert "WHERE lineage_id = %(lineage_id)s" in sql
        assert params == {
            "parent_lineage_ids": ["aaa", "bbb"],
            "lineage_id": "00000000-0000-0000-0000-000000000001",
        }
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_empty_list_is_no_op(self):
        """An empty list (or ``None``) is a no-op: the lineage row
        already carries ``NULL``, no UPDATE needed."""
        from moncpipelib.lineage.tracker import LineageTracker

        tracker = LineageTracker(MagicMock())
        cursor = MagicMock()

        tracker.update_parent_lineage_ids(cursor, lineage_id="lid", parent_lineage_ids=[])
        tracker.update_parent_lineage_ids(cursor, lineage_id="lid", parent_lineage_ids=None)
        cursor.execute.assert_not_called()

    def test_does_not_use_engine(self):
        """Like the other cursor-bound helpers, this must not touch the
        tracker's SA engine."""
        from moncpipelib.lineage.tracker import LineageTracker

        engine = MagicMock()
        engine.begin.side_effect = AssertionError(
            "update_parent_lineage_ids must not open a tracker-internal txn"
        )
        engine.connect.side_effect = AssertionError(
            "update_parent_lineage_ids must not open a tracker-internal connection"
        )

        tracker = LineageTracker(engine)
        tracker.update_parent_lineage_ids(MagicMock(), lineage_id="lid", parent_lineage_ids=["a"])


class TestCreateLineageRecordBackwardsCompat:
    """``create_lineage_record`` is preserved as a wrapper for non-resource
    callers (tests, ad-hoc tools). Phase 3 must not change its external
    behaviour: it still opens its own SA transaction and returns the
    same tuple."""

    def test_wrapper_opens_sa_transaction_and_commits(self):
        """``create_lineage_record`` must still call ``engine.begin()``
        once, ``conn.execute()`` once, and return ``(lineage_id, lineage_key)``."""
        from moncpipelib.lineage.tracker import LineageTracker

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        tracker = LineageTracker(mock_engine)
        lineage_id, lineage_key = tracker.create_lineage_record(
            run_id="r1",
            asset_name="claims",
            layer="bronze",
        )

        assert mock_engine.begin.call_count == 1
        assert mock_conn.execute.call_count == 1
        # The wrapper still returns proper UUID + v1 key shapes.
        assert isinstance(uuid.UUID(lineage_id), uuid.UUID)
        assert lineage_key.startswith("v1:")


def test_create_lineage_record_with_parents(tracker, mock_engine):
    """Test creating lineage record with parent lineage IDs."""
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    parent_ids = [str(uuid.uuid4()) for _ in range(3)]

    tracker.create_lineage_record(
        run_id="test-run-123",
        asset_name="test_asset",
        layer="gold",
        parent_lineage_ids=parent_ids,
        transformation_type="aggregate",
    )

    params = mock_conn.execute.call_args[0][1]
    assert params["parent_lineage_ids"] == parent_ids
    assert params["transformation_type"] == "aggregate"


class TestGenerateUUID7:
    """Tests for generate_uuid7 function."""

    def test_generate_uuid7_returns_valid_uuid(self):
        """Test that generate_uuid7 returns a valid UUID."""
        result = generate_uuid7()

        assert isinstance(result, uuid.UUID)
        assert len(str(result)) == 36  # Standard UUID string length

    def test_generate_uuid7_version(self):
        """Test that generate_uuid7 returns version 7."""
        result = generate_uuid7()

        # Version should be 7
        assert result.version == 7

    def test_generate_uuid7_chronologically_ordered(self):
        """Test that UUID7s are chronologically ordered."""
        import time

        uuid1 = generate_uuid7()
        time.sleep(0.01)  # Small delay
        uuid2 = generate_uuid7()

        # UUIDs should be sortable by time
        assert str(uuid1) < str(uuid2)

    def test_generate_uuid7_unique(self):
        """Test that generate_uuid7 produces unique values."""
        uuids = [generate_uuid7() for _ in range(100)]
        unique_uuids = {str(u) for u in uuids}

        assert len(unique_uuids) == 100


class TestExtractTimestampFromUUID7:
    """Tests for extract_timestamp_from_uuid7 function."""

    def test_extract_timestamp_reasonable_value(self):
        """Test that extracted timestamp is reasonable."""
        uuid7 = generate_uuid7()
        now = datetime.now(UTC)

        timestamp = extract_timestamp_from_uuid7(uuid7)

        # Timestamp should be within a second of now
        delta = abs((timestamp - now).total_seconds())
        assert delta < 1.0

    def test_extract_timestamp_utc(self):
        """Test that extracted timestamp is UTC."""
        uuid7 = generate_uuid7()

        timestamp = extract_timestamp_from_uuid7(uuid7)

        assert timestamp.tzinfo == UTC


class TestGenerateLineageKey:
    """Tests for generate_lineage_key function."""

    def test_generate_lineage_key_with_data_date(self):
        """Test lineage key generation with data date."""
        result = generate_lineage_key(
            asset_name="orders_bronze",
            layer="bronze",
            run_id="abc123-def456-ghi789",
            data_date=date(2024, 1, 15),
        )

        assert result == "v1:orders:bronze:2024-01-15:abc123"

    def test_generate_lineage_key_with_source_file(self):
        """Test lineage key generation with source file hash."""
        result = generate_lineage_key(
            asset_name="orders_bronze",
            layer="bronze",
            run_id="abc123-def456",
            source_file="/path/to/data.csv",
        )

        assert result.startswith("v1:orders:bronze:")
        assert ":abc123" in result
        # Should have an 8-char hash
        parts = result.split(":")
        assert len(parts[3]) == 8

    def test_generate_lineage_key_fallback_timestamp(self):
        """Test lineage key generation with timestamp fallback."""
        result = generate_lineage_key(
            asset_name="orders_bronze",
            layer="bronze",
            run_id="abc123",
        )

        assert result.startswith("v1:orders:bronze:")
        # Should have a timestamp format
        parts = result.split(":")
        assert len(parts[3]) == 14  # YYYYMMDDHHmmss format

    def test_generate_lineage_key_strips_layer_suffix(self):
        """Test that layer suffix is stripped from asset name."""
        result = generate_lineage_key(
            asset_name="orders_silver",
            layer="silver",
            run_id="abc123",
            data_date=date(2024, 1, 15),
        )

        # "orders_silver" should become "orders"
        assert result == "v1:orders:silver:2024-01-15:abc123"

    def test_generate_lineage_key_short_run_id(self):
        """Test with run_id shorter than 6 characters."""
        result = generate_lineage_key(
            asset_name="orders",
            layer="bronze",
            run_id="abc",
            data_date=date(2024, 1, 15),
        )

        assert result == "v1:orders:bronze:2024-01-15:abc"


class TestParseLineageKey:
    """Tests for parse_lineage_key function."""

    def test_parse_lineage_key_valid(self):
        """Test parsing a valid v1 lineage key."""
        result = parse_lineage_key("v1:orders:bronze:2024-01-15:abc123")

        assert result == {
            "version": "1",
            "asset": "orders",
            "layer": "bronze",
            "date_or_hash": "2024-01-15",
            "run_id_prefix": "abc123",
        }

    def test_parse_lineage_key_invalid_format(self):
        """Test parsing an invalid lineage key format."""
        with pytest.raises(ValueError, match="Invalid lineage key format"):
            parse_lineage_key("invalid")

    def test_parse_lineage_key_no_version_prefix(self):
        """Test parsing key without version prefix."""
        with pytest.raises(ValueError, match="must start with version prefix"):
            parse_lineage_key("orders:bronze:2024-01-15:abc123")

    def test_parse_lineage_key_wrong_parts(self):
        """Test parsing key with wrong number of parts."""
        with pytest.raises(ValueError, match="must have 5 parts"):
            parse_lineage_key("v1:orders:bronze")

    def test_parse_lineage_key_unsupported_version(self):
        """Test parsing key with unsupported version."""
        with pytest.raises(ValueError, match="Unsupported lineage key version"):
            parse_lineage_key("v2:orders:bronze:2024-01-15:abc123")


class TestQueryLineageHistory:
    """Tests for LineageTracker.query_lineage_history method."""

    def test_query_lineage_history_no_filters(self, tracker, mock_engine):
        """Test query with no filters."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _self: iter([])
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = tracker.query_lineage_history()

        mock_conn.execute.assert_called_once()
        assert result == []

    def test_query_lineage_history_with_filters(self, tracker, mock_engine):
        """Test query with multiple filters."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _self: iter([])
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        tracker.query_lineage_history(
            asset_name="orders",
            layer="bronze",
            source_file="data.csv",
            data_date=date(2024, 1, 15),
            is_backfill=True,
            limit=50,
        )

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]

        assert params["asset_name"] == "orders"
        assert params["layer"] == "bronze"
        assert params["source_file"] == "data.csv"
        assert params["data_date"] == date(2024, 1, 15)
        assert params["is_backfill"] is True
        assert params["limit"] == 50


class TestWriteValidationRuns:
    """Migration 019 (#308) Phase 5: ``LineageTracker.write_validation_runs``.

    Cursor-bound bulk INSERT, no commit. ``sample_failures`` truncated to
    20 rows per check before persistence.
    """

    def test_no_op_when_empty(self, tracker):
        """Empty ``check_results`` skips ``cursor.executemany`` entirely."""
        cursor = MagicMock()
        n = tracker.write_validation_runs(cursor, lineage_id="lineage-1", check_results=[])
        assert n == 0
        cursor.executemany.assert_not_called()

    def test_executes_bulk_insert_with_expected_rows(self, tracker):
        """One row per ``CheckResultRow``, params shaped to match the
        cursor.executemany contract."""
        from moncpipelib.contracts.models import CheckResultRow

        cursor = MagicMock()
        results = [
            CheckResultRow(
                check_name="schema",
                severity="error",
                passed=True,
                failed_count=0,
                total_count=100,
                sample_failures=None,
            ),
            CheckResultRow(
                check_name="patient_id.unique",
                severity="error",
                passed=False,
                failed_count=5,
                total_count=100,
                sample_failures=[{"patient_id": "dup1"}, {"patient_id": "dup2"}],
            ),
        ]

        n = tracker.write_validation_runs(cursor, lineage_id="lineage-1", check_results=results)

        assert n == 2
        cursor.executemany.assert_called_once()
        sql, rows = cursor.executemany.call_args[0]
        assert "INSERT INTO" in sql
        assert "contract_validation_runs" in sql
        assert "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)" in sql
        assert rows[0] == (
            "lineage-1",
            "schema",
            "error",
            True,
            0,
            100,
            None,
        )
        # Failed row: sample_failures JSON-encoded
        assert rows[1][0:6] == ("lineage-1", "patient_id.unique", "error", False, 5, 100)
        import json

        assert json.loads(rows[1][6]) == [
            {"patient_id": "dup1"},
            {"patient_id": "dup2"},
        ]

    def test_sample_failures_truncated_to_20_rows(self, tracker):
        """Plan-pinned: sample_failures bounded at 20 to keep the JSONB
        payload small. Inputs > 20 are truncated; <= 20 pass through."""
        import json

        from moncpipelib.contracts.models import CheckResultRow

        big_sample = [{"row": i} for i in range(50)]
        small_sample = [{"row": i} for i in range(15)]
        results = [
            CheckResultRow(
                check_name="big_fail",
                severity="error",
                passed=False,
                failed_count=50,
                total_count=100,
                sample_failures=big_sample,
            ),
            CheckResultRow(
                check_name="small_fail",
                severity="warn",
                passed=False,
                failed_count=15,
                total_count=100,
                sample_failures=small_sample,
            ),
        ]

        cursor = MagicMock()
        tracker.write_validation_runs(cursor, lineage_id="lineage-x", check_results=results)

        _, rows = cursor.executemany.call_args[0]
        big_sample_persisted = json.loads(rows[0][6])
        small_sample_persisted = json.loads(rows[1][6])
        assert len(big_sample_persisted) == 20  # truncated
        assert big_sample_persisted == big_sample[:20]
        assert len(small_sample_persisted) == 15  # untouched

    def test_does_not_commit(self, tracker):
        """The caller (PostgresResource write path) owns the transaction
        lifecycle; the tracker must NOT call commit."""
        from moncpipelib.contracts.models import CheckResultRow

        cursor = MagicMock()
        cursor.connection = MagicMock()

        tracker.write_validation_runs(
            cursor,
            lineage_id="lineage-1",
            check_results=[
                CheckResultRow(
                    check_name="schema",
                    severity="error",
                    passed=True,
                )
            ],
        )

        cursor.connection.commit.assert_not_called()


class TestWriteScd2Reconciliation:
    """Migration 019 (#308) Phase 6: ``LineageTracker.write_scd2_reconciliation``.

    Single-row INSERT on the caller's cursor; no commit. Audit row is
    intended to be persisted inside ``reconcile_scd2``'s own transaction
    so it is atomic with the reconcile DML.
    """

    def test_executes_single_insert_with_expected_params(self, tracker):
        cursor = MagicMock()

        tracker.write_scd2_reconciliation(
            cursor,
            run_id="run-xyz",
            asset_name="silver/dim_provider",
            target_table="silver.dim_provider",
            pipeline_id="11111111-2222-3333-4444-555555555555",
            work_mem_applied="256MB",
            rows_collapsed=10,
            rows_timeline_updated=42,
            rows_renumbered=42,
            duration_seconds=1.234,
            metadata={"partition": "2026-05-15"},
        )

        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "INSERT INTO" in sql
        assert "scd2_reconciliations" in sql
        # 10 placeholders: run_id, asset_name, pipeline_id, target_table,
        # work_mem_applied, rows_collapsed, rows_timeline_updated,
        # rows_renumbered, duration_seconds, metadata (jsonb)
        assert "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)" in sql
        assert params[0] == "run-xyz"
        assert params[1] == "silver/dim_provider"
        assert params[2] == "11111111-2222-3333-4444-555555555555"
        assert params[3] == "silver.dim_provider"
        assert params[4] == "256MB"
        assert params[5] == 10
        assert params[6] == 42
        assert params[7] == 42
        assert params[8] == 1.234
        import json

        assert json.loads(params[9]) == {"partition": "2026-05-15"}

    def test_pipeline_id_none_pass_through(self, tracker):
        """Caller without a contract may pass ``pipeline_id=None``; the
        FK is nullable so this must round-trip cleanly."""
        cursor = MagicMock()

        tracker.write_scd2_reconciliation(
            cursor,
            run_id="run-xyz",
            asset_name="silver.x",
            target_table="silver.x",
            pipeline_id=None,
            work_mem_applied=None,
            rows_collapsed=0,
            rows_timeline_updated=0,
            rows_renumbered=0,
            duration_seconds=0.5,
        )

        params = cursor.execute.call_args[0][1]
        assert params[2] is None  # pipeline_id
        assert params[4] is None  # work_mem_applied
        # metadata not passed -> None -> serialized as None
        assert params[9] is None

    def test_does_not_commit(self, tracker):
        cursor = MagicMock()
        cursor.connection = MagicMock()

        tracker.write_scd2_reconciliation(
            cursor,
            run_id="r",
            asset_name="a",
            target_table="s.t",
            pipeline_id=None,
            work_mem_applied=None,
            rows_collapsed=0,
            rows_timeline_updated=0,
            rows_renumbered=0,
            duration_seconds=0.0,
        )

        cursor.connection.commit.assert_not_called()
