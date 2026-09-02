"""Integration test: streamed batched writes reach COPY sizing via total_rows_hint.

Individually small batches never crossed ``bulk_insert_threshold`` on their
own -- ``insert_rows`` sized the AUTO COPY-vs-execute_values decision on the
per-call batch alone, so a large load streamed as many small parts stayed on
the per-row executemany path forever (root cause of the #456 pgaudit-volume
incident: a 392k-row streamed load produced ~731k pgaudit lines). This mirrors
the full-refresh clear-method sizing bug fixed by
model-oncology-public/moncpipelib#4: ``insert_rows`` now accepts
``total_rows_hint`` and forwards it to ``should_use_copy``, which sizes on
``max(batch_row_count, total_rows_hint)``.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

import moncpipelib.io_managers.writers as writers_mod
from moncpipelib.io_managers.postgres import PostgresIOManager
from moncpipelib.streaming import BatchedDataFrame

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


def _install_write_path_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[int]]:
    """Wrap ``insert_with_copy`` / ``insert_with_execute_values`` with call-count spies.

    Spies still delegate to the real implementation, so the genuine
    COPY/executemany SQL executes against the real database -- this proves
    which path ran without replacing the assertion with a mock that skips
    execution.

    Returns:
        ``(copy_calls, execute_values_calls)`` -- each a list that gets one
        entry (the batch's row count) appended per call to that path.
    """
    copy_calls: list[int] = []
    execute_values_calls: list[int] = []
    original_copy = writers_mod.insert_with_copy
    original_execute_values = writers_mod.insert_with_execute_values

    def _spy_copy(config: Any, cursor: Any, table_name: str, df: pl.DataFrame, context: Any) -> int:
        copy_calls.append(len(df))
        return original_copy(config, cursor, table_name, df, context)

    def _spy_execute_values(
        config: Any,
        cursor: Any,
        table_name: str,
        columns: list[str],
        df: pl.DataFrame,
        context: Any,
    ) -> int:
        execute_values_calls.append(len(df))
        return original_execute_values(config, cursor, table_name, columns, df, context)

    monkeypatch.setattr(writers_mod, "insert_with_copy", _spy_copy)
    monkeypatch.setattr(writers_mod, "insert_with_execute_values", _spy_execute_values)
    return copy_calls, execute_values_calls


class TestStreamTotalReachesCopyThreshold:
    """#456: individually sub-threshold batches, a stream total above it."""

    TABLE_NAME_PREFIX = "stream_copy"
    BATCH_SIZE = 300
    N_BATCHES = 4
    THRESHOLD = 1_000

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.table_name = f"{self.TABLE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns={"id": "INTEGER NOT NULL", "name": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="auto",
            bulk_insert_threshold=self.THRESHOLD,
        )
        yield
        self.builder.drop(self.fqn)

    def test_batched_append_with_large_hint_uses_copy_not_execute_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every batch (300 rows) is below the 1,000-row threshold, but the
        stream total (1,200) is above it.

        The batched write path exposes no ``insert_method`` metadata (unlike
        the single-DataFrame full_refresh/append paths), so path verification
        here wraps ``insert_with_copy`` / ``insert_with_execute_values`` with
        spies that still delegate to the real implementation -- this proves
        which path executed while still exercising the genuine COPY/executemany
        SQL against the real database, rather than replacing the assertion
        with a mock that skips execution.
        """
        copy_calls, execute_values_calls = _install_write_path_spies(monkeypatch)

        total = self.BATCH_SIZE * self.N_BATCHES
        batches = [
            pl.DataFrame(
                {
                    "id": list(range(i * self.BATCH_SIZE, (i + 1) * self.BATCH_SIZE)),
                    "name": [f"row-{i}-{j}" for j in range(self.BATCH_SIZE)],
                }
            )
            for i in range(self.N_BATCHES)
        ]
        batched = BatchedDataFrame(batches=iter(batches), total_rows_hint=total)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        self.io_mgr.handle_output(ctx, batched)

        # Every batch reached COPY; execute_values was never used, even though
        # every individual batch (300 rows) is below the 1,000-row threshold.
        assert copy_calls == [self.BATCH_SIZE] * self.N_BATCHES
        assert execute_values_calls == []

        assert self.builder.count(self.fqn) == total
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert [r["id"] for r in rows] == list(range(total))

    def test_batched_append_without_hint_stays_on_execute_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control case: without a hint, small batches stay on execute_values.

        Pins that the fix is additive (a hint reaching the threshold), not a
        blanket switch to COPY for every batched write.
        """
        copy_calls, execute_values_calls = _install_write_path_spies(monkeypatch)

        batches = [
            pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            pl.DataFrame({"id": [4, 5], "name": ["d", "e"]}),
        ]
        batched = BatchedDataFrame(batches=iter(batches))  # no total_rows_hint

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        self.io_mgr.handle_output(ctx, batched)

        assert copy_calls == []
        assert execute_values_calls == [3, 2]
        assert self.builder.count(self.fqn) == 5


class TestCopyInsertFidelityEncoding:
    """#456 review Fix 1: insert_with_copy's pre-fix encoding collapses a
    literal string to SQL NULL, diverging from the parameterized path.

    Before the fix, ``insert_with_copy`` serialized CSV with
    ``null_value="\\N"`` and issued ``COPY ... WITH (FORMAT CSV, NULL
    '\\N')`` -- an encoding that cannot distinguish SQL NULL from the
    literal two-character string ``"\\N"``: both collapse to NULL on
    read-back. Verified live during review: a hinted batched append (routed
    to COPY by ``total_rows_hint``) landed ``None`` for a ``"\\N"`` value
    where the parameterized executemany path (routed to the identical
    insert by omitting the hint) landed the literal string ``'\\N'`` for the
    same input.

    Both tests write the exact same four values (SQL NULL, empty string, the
    literal text ``"\\N"``, and a unicode value) through the *same* table and
    batch shape -- only ``total_rows_hint`` differs, routing one to COPY and
    the other to the parameterized path. Fidelity requires both to read back
    identically.

    The COPY-path test MUST fail before the fix (the "\\N" row reads back as
    None instead of the literal string); it passes once ``insert_with_copy``
    switches to ``serialize_for_staging_copy`` / ``COPY_STAGING_OPTIONS``
    (the same encoding upsert staging already uses, #375 D1).
    """

    TABLE_NAME_PREFIX = "copy_fidelity"
    THRESHOLD = 1_000
    VALUES: list[str | None] = ["\\N", "", None, "héllo wörld 世界 🎉"]

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> Any:
        self.table_name = f"{self.TABLE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns={"id": "INTEGER NOT NULL", "val": "TEXT"},
            primary_key=["id"],
        )
        self.builder = table_builder
        self.io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="auto",
            bulk_insert_threshold=self.THRESHOLD,
        )
        yield
        self.builder.drop(self.fqn)

    def _write_and_read(
        self, monkeypatch: pytest.MonkeyPatch, *, total_rows_hint: int | None
    ) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        copy_calls, execute_values_calls = _install_write_path_spies(monkeypatch)

        df = pl.DataFrame({"id": list(range(1, len(self.VALUES) + 1)), "val": self.VALUES})
        batched = BatchedDataFrame(batches=iter([df]), total_rows_hint=total_rows_hint)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        self.io_mgr.handle_output(ctx, batched)

        rows = self.builder.read_all(self.fqn, order_by="id")
        return copy_calls, execute_values_calls, rows

    def test_copy_path_round_trips_null_empty_backslash_n_and_unicode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hint above threshold routes the (sub-threshold) batch to COPY."""
        copy_calls, execute_values_calls, rows = self._write_and_read(
            monkeypatch, total_rows_hint=self.THRESHOLD * 2
        )
        assert copy_calls == [len(self.VALUES)]
        assert execute_values_calls == []
        assert [r["val"] for r in rows] == self.VALUES

    def test_executemany_path_round_trips_identically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: without a hint, the same batch stays on the parameterized
        executemany path -- and must land the exact same values."""
        copy_calls, execute_values_calls, rows = self._write_and_read(
            monkeypatch, total_rows_hint=None
        )
        assert copy_calls == []
        assert execute_values_calls == [len(self.VALUES)]
        assert [r["val"] for r in rows] == self.VALUES


class TestNestedDtypeFallsBackFromCopy:
    """#456 review Fix 2: a nested-dtype column crashes the COPY path.

    ``df.write_csv`` raises ``ComputeError: CSV format does not support
    nested data`` for any List/Array/Struct column. Before the fix,
    ``should_use_copy`` alone decided whether to route a batch to COPY --
    it never sees the DataFrame, so a ``pl.List(pl.Utf8)`` column bound for
    a ``TEXT[]`` target crashed whenever COPY was selected, whether by AUTO
    sizing (boosted by ``total_rows_hint``) or by an explicit
    ``bulk_insert_method="copy"``. The parameterized executemany path
    already handled this fine -- psycopg adapts a Python ``list`` to a
    Postgres array natively.

    The hinted-batch test MUST raise ``ComputeError`` before the fix (a
    sub-threshold batch gets pushed to COPY by the large hint); it passes
    once ``insert_rows`` additionally gates on ``_frame_supports_copy``.
    """

    TABLE_NAME_PREFIX = "nested_dtype"
    THRESHOLD = 1_000

    @pytest.fixture(autouse=True)
    def setup(self, table_builder: TableBuilder) -> Any:
        self.table_name = f"{self.TABLE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns={"id": "INTEGER NOT NULL", "tags": "TEXT[]"},
            primary_key=["id"],
        )
        self.builder = table_builder
        yield
        self.builder.drop(self.fqn)

    def test_batched_append_with_large_hint_falls_back_and_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> None:
        """Every batch is sub-threshold, but the stream-total hint pushes AUTO
        toward COPY -- the fallback must still land correct array values."""
        io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="auto",
            bulk_insert_threshold=self.THRESHOLD,
        )
        copy_calls, execute_values_calls = _install_write_path_spies(monkeypatch)

        df = pl.DataFrame({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
        batched = BatchedDataFrame(batches=iter([df]), total_rows_hint=self.THRESHOLD * 2)

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        io_mgr.handle_output(ctx, batched)

        # Never reached COPY despite the hint -- _frame_supports_copy vetoed it.
        assert copy_calls == []
        assert execute_values_calls == [2]

        assert self.builder.count(self.fqn) == 2
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[0]["tags"] == ["a", "b"]
        assert rows[1]["tags"] == ["c"]

    def test_single_shot_explicit_copy_falls_back_with_warning(
        self,
        io_manager_factory: Callable[..., PostgresIOManager],
    ) -> None:
        """An explicit bulk_insert_method='copy' still succeeds via the
        parameterized fallback, and warns that it did so."""
        io_mgr = io_manager_factory(
            db_schema="test_write",
            enable_row_lineage=False,
            add_metadata_columns=False,
            bulk_insert_method="copy",
        )

        df = pl.DataFrame({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "append"},
        )
        io_mgr.handle_output(ctx, df)

        assert self.builder.count(self.fqn) == 2
        rows = self.builder.read_all(self.fqn, order_by="id")
        assert rows[0]["tags"] == ["a", "b"]
        assert rows[1]["tags"] == ["c"]

        warnings = [str(call.args[0]) for call in ctx.log.warning.call_args_list]
        assert any("nested" in w.lower() for w in warnings)
