"""Cookbook tests for the diagnostics module.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers (not included in generated docs)
# ---------------------------------------------------------------------------


def _write_cgroup_files(
    base: Path,
    *,
    memory_current: str | None = None,
    memory_max: str | None = None,
    cpu_stat: str | None = None,
) -> None:
    """Write mock cgroup v2 files into a temporary directory."""
    if memory_current is not None:
        (base / "memory.current").write_text(memory_current)
    if memory_max is not None:
        (base / "memory.max").write_text(memory_max)
    if cpu_stat is not None:
        (base / "cpu.stat").write_text(cpu_stat)


# ---------------------------------------------------------------------------
# Cookbook examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Basic PodResourceSampler Usage",
    description=(
        "Use PodResourceSampler as a context manager to collect CPU and memory "
        "metrics during a workload. Access the summary after the block completes. "
        "The sampler spawns a lightweight daemon thread that reads cgroup v2 metrics "
        "at a configurable interval."
    ),
    category="diagnostics",
)
def test_basic_context_manager(tmp_path: Path) -> None:
    """Demonstrate basic PodResourceSampler context manager usage."""
    _write_cgroup_files(
        tmp_path,
        memory_current="2097152\n",  # 2 MiB
        memory_max="4194304\n",  # 4 MiB
        cpu_stat="usage_usec 500000\n",
    )

    # --- cookbook:start ---
    from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig, SamplerMode

    config = SamplerConfig(
        interval_seconds=0.5,
        mode=SamplerMode.METADATA,
        cgroup_base_path=str(tmp_path),  # default: /sys/fs/cgroup
    )

    with PodResourceSampler(config) as sampler:
        time.sleep(1.0)  # simulate pipeline workload

    # Summary is safely accessible after the context manager exits
    summary = sampler.summary
    print(f"Samples collected: {summary.sample_count}")
    print(f"Peak memory: {summary.memory_peak_bytes} bytes")
    print(f"Peak memory %: {summary.memory_peak_pct:.1f}%")
    print(f"Duration: {summary.duration_seconds:.1f}s")
    # --- cookbook:end ---

    assert summary.sample_count >= 1
    assert summary.memory_peak_bytes == 2_097_152
    assert summary.memory_peak_pct == pytest.approx(50.0, abs=0.1)


@pytest.mark.cookbook(
    title="Structured JSON Log Output",
    description=(
        "Configure PodResourceSampler in LOG mode to emit structured JSON lines "
        "for each sample tick. Each line contains an ``event`` key and whichever "
        "metrics are available (memory, CPU). Useful for shipping resource metrics "
        "to a log aggregator alongside pipeline execution logs."
    ),
    category="diagnostics",
)
def test_log_mode_json_output(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Demonstrate LOG mode with structured JSON output."""
    _write_cgroup_files(
        tmp_path,
        memory_current="1048576\n",  # 1 MiB
        memory_max="2097152\n",  # 2 MiB
        cpu_stat="usage_usec 100000\n",
    )

    # --- cookbook:start ---
    from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig, SamplerMode

    config = SamplerConfig(
        interval_seconds=0.5,
        mode=SamplerMode.LOG,
        cgroup_base_path=str(tmp_path),  # default: /sys/fs/cgroup
    )

    with (
        caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
        PodResourceSampler(config),
    ):
        time.sleep(0.8)  # allow at least one sample tick

    # Each log record is a compact JSON object with resource metrics
    for record in caplog.records:
        if "pod_resource_sample" in record.message:
            sample = json.loads(record.message)
            print(json.dumps(sample, indent=2))
    # --- cookbook:end ---

    json_lines = [r.message for r in caplog.records if "pod_resource_sample" in r.message]
    assert len(json_lines) >= 1
    parsed = json.loads(json_lines[0])
    assert parsed["event"] == "pod_resource_sample"
    assert "memory_bytes" in parsed


@pytest.mark.cookbook(
    title="Dagster Asset Metadata Integration",
    description=(
        "Convert the sampler summary into Dagster MetadataValue entries, suitable "
        "for attaching to asset materializations via ``context.add_output_metadata()``. "
        "Keys include ``pod_resource_samples``, ``pod_memory_peak_bytes``, "
        "``pod_cpu_peak_pct``, and more."
    ),
    category="diagnostics",
)
def test_dagster_metadata(tmp_path: Path) -> None:
    """Demonstrate converting summary to Dagster asset metadata."""
    _write_cgroup_files(
        tmp_path,
        memory_current="4194304\n",  # 4 MiB
        memory_max="8388608\n",  # 8 MiB
        cpu_stat="usage_usec 250000\n",
    )

    # --- cookbook:start ---
    from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig, SamplerMode

    config = SamplerConfig(
        interval_seconds=0.5,
        mode=SamplerMode.METADATA,
        cgroup_base_path=str(tmp_path),  # default: /sys/fs/cgroup
    )

    sampler = PodResourceSampler(config)
    with sampler:
        time.sleep(1.0)  # simulate pipeline workload

    # Convert accumulated stats to Dagster metadata
    metadata = sampler.summary.to_metadata()
    for key, value in sorted(metadata.items()):
        print(f"{key}: {value.value}")

    # In a real Dagster asset you would call:
    # context.add_output_metadata(sampler.summary.to_metadata())
    # --- cookbook:end ---

    assert "pod_resource_samples" in metadata
    assert "pod_memory_peak_bytes" in metadata
    assert metadata["pod_resource_samples"].value >= 1
