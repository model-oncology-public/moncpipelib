"""Integration tests for PodResourceSampler against real system metrics.

Tests the diagnostics sampler end-to-end: reading /proc/self/status on Linux,
collecting samples during database writes, producing Dagster metadata, and
emitting structured JSON log lines.

Requires Docker. Run with: uv run pytest -m integration -v
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import polars as pl
import psycopg
import pytest

from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig, SamplerMode

from .conftest import TableBuilder, make_mock_output_context

pytestmark = pytest.mark.integration


class TestPodResourceSamplerIntegration:
    """Verify PodResourceSampler works against real /proc metrics and database IO."""

    TABLE_COLUMNS: dict[str, str] = {
        "id": "INTEGER",
        "name": "TEXT",
        "value": "DOUBLE PRECISION",
    }

    @pytest.fixture(autouse=True)
    def setup(
        self,
        table_builder: TableBuilder,
        pg_connection: psycopg.Connection,
        io_manager_factory: Any,
    ) -> Any:
        self.suffix = uuid.uuid4().hex[:8]
        self.table_name = f"diag_sampler_{self.suffix}"
        self.fqn = table_builder.create_table(
            self.table_name,
            columns=self.TABLE_COLUMNS,
            primary_key=["id"],
        )
        self.builder = table_builder
        self.conn = pg_connection
        self.io_mgr = io_manager_factory(db_schema="test_write")
        yield
        self.builder.drop(self.fqn)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_sampler_reads_proc_memory(self) -> None:
        """Run PodResourceSampler with default config; verify memory_peak_bytes > 0.

        On Linux (including WSL2 and Docker), /proc/self/status is available
        and VmRSS will be reported. The sampler falls back to /proc when
        cgroup v2 files are absent.
        """
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
        )
        with PodResourceSampler(config) as sampler:
            # Give the daemon thread time to collect at least one sample
            time.sleep(1.2)

        summary = sampler.summary
        assert summary.sample_count > 0, "Expected at least one sample"
        assert summary.memory_peak_bytes > 0, (
            "Expected memory_peak_bytes > 0 on Linux (reads /proc/self/status VmRSS)"
        )

    def test_sampler_during_db_write(self) -> None:
        """Wrap a handle_output call in a PodResourceSampler context.

        Verify samples > 0 and duration > 0 after the write completes.
        """
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
        )

        df = pl.DataFrame(
            {
                "id": list(range(1, 101)),
                "name": [f"row-{i}" for i in range(1, 101)],
                "value": [float(i) * 1.1 for i in range(1, 101)],
            }
        )

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )

        with PodResourceSampler(config) as sampler:
            self.io_mgr.handle_output(ctx, df)
            # Allow at least one sample tick after the write
            time.sleep(0.8)

        summary = sampler.summary
        assert summary.sample_count > 0, "Expected at least one sample during DB write"
        assert summary.duration_seconds > 0, "Expected non-zero duration"

    def test_metadata_attaches_to_dagster_context(self) -> None:
        """Run sampler, call to_metadata(), verify dict has expected keys."""
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
        )
        with PodResourceSampler(config) as sampler:
            time.sleep(1.2)

        metadata = sampler.summary.to_metadata()

        expected_keys = {
            "pod_resource_samples",
            "pod_resource_duration_sec",
            "pod_memory_peak_bytes",
            "pod_memory_avg_bytes",
            "pod_memory_peak_pct",
            "pod_cpu_peak_pct",
            "pod_cpu_avg_pct",
        }
        assert expected_keys.issubset(set(metadata.keys())), (
            f"Missing metadata keys: {expected_keys - set(metadata.keys())}"
        )

    def test_log_mode_emits_during_io(self, caplog: pytest.LogCaptureFixture) -> None:
        """Use caplog to capture log output during a handle_output wrapped in sampler.

        Verify JSON log lines containing "pod_resource_sample" are emitted.
        """
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            logger_name="moncpipelib.diagnostics",
        )

        df = pl.DataFrame(
            {
                "id": list(range(1, 51)),
                "name": [f"item-{i}" for i in range(1, 51)],
                "value": [float(i) for i in range(1, 51)],
            }
        )

        ctx = make_mock_output_context(
            asset_name=self.table_name,
            metadata={"write_mode": "full_refresh"},
        )

        with (
            caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
            PodResourceSampler(config) as _sampler,
        ):
            self.io_mgr.handle_output(ctx, df)
            # Allow time for at least one log-mode sample tick
            time.sleep(1.2)

        # Find JSON log lines emitted by the sampler
        sample_logs: list[dict[str, Any]] = []
        for record in caplog.records:
            if record.name == "moncpipelib.diagnostics":
                try:
                    parsed = json.loads(record.getMessage())
                    if parsed.get("event") == "pod_resource_sample":
                        sample_logs.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    continue

        assert len(sample_logs) > 0, (
            "Expected at least one JSON log line with event=pod_resource_sample"
        )
        # Verify the log line contains memory_bytes (available on Linux via /proc)
        assert "memory_bytes" in sample_logs[0], "Expected memory_bytes in the JSON log payload"
