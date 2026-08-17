"""Unit tests for :func:`moncpipelib.reference.read_latest_partition`.

These tests cover SQL composition, identifier quoting, the empty-table
fail-fast path, the column projection surface, and ``LoggingContext``
propagation -- all of which can be exercised without a real database
by patching :mod:`psycopg.connect` and
:func:`moncpipelib.resources.postgres.read_batched`.

End-to-end streaming behaviour (multi-batch yields, latest-partition
re-evaluation when a row lands mid-stream) lives in
``tests/integration/test_reference.py`` where a real PostgreSQL
container is available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from moncpipelib.reference import (
    EmptyPartitionedTableError,
    _split_qualified_table,
    read_latest_partition,
)
from moncpipelib.resources.postgres import PostgresResource


@pytest.fixture
def resource() -> MagicMock:
    """Construct a stand-in for :class:`PostgresResource`.

    ``PostgresResource`` is a frozen Pydantic model and rejects attribute
    assignment, so a :class:`MagicMock` with ``spec=PostgresResource``
    is the cleanest way to satisfy the type while letting tests stub out
    ``get_connection()``.  The streaming SELECT is patched separately
    via :func:`moncpipelib.reference.read_batched`, so the resource is
    never actually opened against a database in unit tests.
    """
    return MagicMock(spec=PostgresResource)


def _patch_precheck(resource: MagicMock, max_value: object) -> MagicMock:
    """Wire ``resource.get_connection`` so the precheck cursor returns
    ``(max_value,)`` on ``fetchone()``.

    Returns the mock cursor so tests can assert on the SQL the helper
    passed to ``cur.execute(...)``.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (max_value,)
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    resource.get_connection.return_value = mock_conn
    return mock_cursor


class TestSplitQualifiedTable:
    """``_split_qualified_table`` is an injection-surface guard; cover
    the happy path and the rejection cases explicitly."""

    def test_simple_schema_table(self) -> None:
        assert _split_qualified_table("reference_bronze.icdo3") == (
            "reference_bronze",
            "icdo3",
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "no_dot",
            ".table",
            "schema.",
            ".",
            # Three-part names are rejected on purpose:
            # ``sql.Identifier(schema, table)`` only composes a 2-part
            # qualified name, so silently merging ``catalog.schema``
            # into the schema slot would emit a malformed quoted
            # identifier (``"catalog.schema"."table"``).
            "catalog.schema.table",
        ],
    )
    def test_rejects_unqualified(self, bad: str) -> None:
        with pytest.raises(ValueError, match="dotted 'schema.table'"):
            _split_qualified_table(bad)


class TestReadLatestPartitionPrecheck:
    """Cover the pre-check query + empty-table fail-fast."""

    @patch("moncpipelib.reference.read_batched")
    def test_empty_bronze_raises_with_table_and_column_in_message(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """``MAX(...)`` -> NULL should raise ``EmptyPartitionedTableError``."""
        _patch_precheck(resource, None)

        with pytest.raises(EmptyPartitionedTableError) as excinfo:
            # Iterator must be consumed for the precheck to run -- list().
            list(
                read_latest_partition(
                    resource,
                    source_table="reference_bronze.icdo3_morphology",
                    partition_column="load_period",
                )
            )

        msg = str(excinfo.value)
        assert "reference_bronze.icdo3_morphology" in msg
        assert "load_period" in msg
        # No streaming SELECT should have been issued.
        mock_read_batched.assert_not_called()

    def test_empty_bronze_error_is_lookup_error(
        self,
        resource: MagicMock,
    ) -> None:
        """``EmptyPartitionedTableError`` subclasses ``LookupError`` so
        generic ``except LookupError`` blocks still catch the case."""
        _patch_precheck(resource, None)

        with pytest.raises(LookupError):
            list(
                read_latest_partition(
                    resource,
                    source_table="reference_bronze.icdo3_morphology",
                )
            )

    @patch("moncpipelib.reference.read_batched")
    def test_precheck_sql_is_quoted(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """Pre-check should pass a ``Composed`` SQL object (not a string
        with f-string interpolation).  Render it to verify identifiers
        are double-quoted."""
        from psycopg.sql import Composable

        mock_cursor = _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                partition_column="load_period",
            )
        )

        executed = mock_cursor.execute.call_args[0][0]
        assert isinstance(executed, Composable)
        rendered = executed.as_string(None)
        # All identifiers double-quoted -- not f-string interpolated.
        assert '"load_period"' in rendered
        assert '"reference_bronze"' in rendered
        assert '"icdo3_morphology"' in rendered


class TestReadLatestPartitionQueryShape:
    """Cover the SQL handed to ``read_batched``."""

    @patch("moncpipelib.reference.read_batched")
    def test_default_select_star(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """No ``columns=`` -> ``SELECT *``."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
            )
        )

        query = mock_read_batched.call_args[0][0]
        assert "SELECT *" in query
        # Subquery form -- the spec explicitly requires this so a
        # later-landing partition is picked up at execution time.
        assert (
            '"load_period" = (SELECT MAX("load_period") FROM "reference_bronze"."icdo3_morphology")'
        ) in query

    @patch("moncpipelib.reference.read_batched")
    def test_columns_projection(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """``columns=(...)`` -> projected SELECT with quoted identifiers
        in order."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                columns=("code", "label", "category"),
            )
        )

        query = mock_read_batched.call_args[0][0]
        assert 'SELECT "code", "label", "category" FROM' in query

    @patch("moncpipelib.reference.read_batched")
    def test_reserved_word_partition_column_quoted(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """A reserved-word partition column must be double-quoted so the
        composed SQL parses.  This is the identifier-quoting acceptance
        test from the issue."""
        _patch_precheck(resource, 1)
        mock_read_batched.return_value = iter([])

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                partition_column="order",
            )
        )

        query = mock_read_batched.call_args[0][0]
        # ``order`` is reserved -- it must be quoted in both the WHERE
        # and the inner subquery, never appearing bare.
        assert '"order"' in query
        assert " order " not in query.lower().replace('"order"', "")

    @patch("moncpipelib.reference.read_batched")
    def test_passes_batch_size_through(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """``batch_size`` should be forwarded to :func:`read_batched`."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                batch_size=10_000,
            )
        )

        kwargs = mock_read_batched.call_args[1]
        assert kwargs["batch_size"] == 10_000

    @patch("moncpipelib.reference.read_batched")
    def test_columns_iterable_only_consumed_once(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """A generator of column names should be materialised once so a
        future re-iteration (defensive code, future refactor) cannot
        accidentally drain it twice."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        def _names() -> object:
            yield "code"
            yield "label"

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                columns=_names(),
            )
        )

        query = mock_read_batched.call_args[0][0]
        assert 'SELECT "code", "label" FROM' in query


class TestReadLatestPartitionLogging:
    """Cover the ``context=`` propagation contract."""

    @patch("moncpipelib.reference.read_batched")
    def test_logs_latest_partition_when_context_supplied(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """The "Reading latest partition X from Y" line is emitted at
        INFO after the pre-check, and ``context`` is forwarded to
        :func:`read_batched` for per-batch progress logs."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        context = MagicMock()
        context.log = MagicMock()

        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
                context=context,
            )
        )

        info_calls = [c.args for c in context.log.info.call_args_list]
        assert any(
            "Reading latest partition" in c[0]
            and c[1] == "2026-05-01"
            and c[2] == "reference_bronze.icdo3_morphology"
            for c in info_calls
        )

        # ``read_batched`` is called with the same context.
        assert mock_read_batched.call_args[1]["context"] is context

    @patch("moncpipelib.reference.read_batched")
    def test_does_not_log_when_no_context(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        """``context=None`` (the default) silences the helper-side log."""
        _patch_precheck(resource, "2026-05-01")
        mock_read_batched.return_value = iter([])

        # No context -- just check no exception is raised and
        # ``read_batched`` is invoked with ``context=None``.
        list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
            )
        )
        assert mock_read_batched.call_args[1]["context"] is None


class TestReadLatestPartitionIteratorPassthrough:
    """The helper yields whatever :func:`read_batched` yields, batch for
    batch, without rebuffering."""

    @patch("moncpipelib.reference.read_batched")
    def test_yields_each_batch(
        self,
        mock_read_batched: MagicMock,
        resource: MagicMock,
    ) -> None:
        _patch_precheck(resource, "2026-05-01")
        b1 = pl.DataFrame({"id": [1, 2, 3]})
        b2 = pl.DataFrame({"id": [4]})
        mock_read_batched.return_value = iter([b1, b2])

        batches = list(
            read_latest_partition(
                resource,
                source_table="reference_bronze.icdo3_morphology",
            )
        )
        assert len(batches) == 2
        assert batches[0].equals(b1)
        assert batches[1].equals(b2)
