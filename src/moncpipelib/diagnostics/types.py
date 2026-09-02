"""Type definitions for the diagnostics subpackage."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Any


class SamplerMode(enum.Flag):
    """Operating modes for PodResourceSampler.

    Can be combined with bitwise OR: ``SamplerMode.METADATA | SamplerMode.LOG``.
    """

    METADATA = enum.auto()
    LOG = enum.auto()


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the PodResourceSampler.

    Attributes:
        interval_seconds: Seconds between metric samples. Must be >= 0.5.
        mode: One or both of METADATA and LOG.
        logger_name: Logger name for LOG mode structured output.
        cgroup_base_path: Base path for cgroup v2 files.
    """

    interval_seconds: float = field(
        default_factory=lambda: float(os.environ.get("MONCPIPELIB_DIAGNOSTICS_INTERVAL", "5.0"))
    )
    mode: SamplerMode = field(default_factory=lambda: SamplerMode.METADATA | SamplerMode.LOG)
    logger_name: str = "moncpipelib.diagnostics"
    cgroup_base_path: str = field(
        default_factory=lambda: os.environ.get(
            "MONCPIPELIB_DIAGNOSTICS_CGROUP_PATH",
            "/sys/fs/cgroup",
        )
    )


@dataclass(frozen=True)
class ResourceSample:
    """A single point-in-time resource measurement.

    Attributes:
        timestamp: Monotonic clock value (time.monotonic()).
        memory_bytes: Total cgroup memory usage in bytes (``memory.current``),
            including process RSS *and* kernel page cache.  None if unavailable.
        memory_limit_bytes: Memory limit in bytes, or None if unavailable.
        memory_rss_bytes: Process RSS (heap, stack, anonymous mmap) from the
            ``anon`` counter in ``memory.stat``.  This is the non-evictable
            portion of memory -- what actually matters for sizing pod limits.
            None if ``memory.stat`` is unavailable.
        memory_cache_bytes: Evictable memory (filesystem page cache +
            reclaimable kernel slab) from ``file + slab_reclaimable`` in
            ``memory.stat``.  The kernel reclaims this under pressure, so it
            does not represent true memory consumption.  None if
            ``memory.stat`` is unavailable.
        cpu_usage_usec: Cumulative CPU usage in microseconds, or None.
    """

    timestamp: float
    memory_bytes: int | None
    memory_limit_bytes: int | None
    memory_rss_bytes: int | None = None
    memory_cache_bytes: int | None = None
    cpu_usage_usec: int | None = None


@dataclass
class SamplerSummary:
    """Accumulated summary statistics from a sampling session.

    Mutable accumulator updated by the sampling thread under lock.
    """

    sample_count: int = 0
    duration_seconds: float = 0.0

    # Total memory stats -- includes RSS + page cache (bytes)
    memory_peak_bytes: int = 0
    memory_min_bytes: int | None = None
    memory_sum_bytes: int = 0

    # Total memory utilization (percentage of limit)
    memory_peak_pct: float = 0.0

    # RSS-only stats -- process heap/stack/anon mmap, non-evictable (bytes)
    memory_rss_peak_bytes: int = 0
    memory_rss_min_bytes: int | None = None
    memory_rss_sum_bytes: int = 0
    memory_rss_peak_pct: float = 0.0

    # Cache stats -- page cache + slab_reclaimable, evictable (bytes)
    memory_cache_peak_bytes: int = 0
    memory_cache_sum_bytes: int = 0

    # CPU stats (percentage, 0-100+ for multi-core)
    cpu_peak_pct: float = 0.0
    cpu_sum_pct: float = 0.0
    cpu_min_pct: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return Dagster MetadataValue dict for asset output metadata.

        Performs a runtime import of ``dagster.MetadataValue`` so the
        diagnostics module does not hard-depend on dagster.

        Raises:
            ImportError: If dagster is not installed.
        """
        from dagster import MetadataValue

        avg_memory = self.memory_sum_bytes // self.sample_count if self.sample_count > 0 else 0
        avg_cpu = self.cpu_sum_pct / self.sample_count if self.sample_count > 0 else 0.0

        metadata: dict[str, Any] = {
            "pod_resource_samples": MetadataValue.int(self.sample_count),
            "pod_resource_duration_sec": MetadataValue.float(round(self.duration_seconds, 2)),
            "pod_memory_peak_bytes": MetadataValue.int(self.memory_peak_bytes),
            "pod_memory_avg_bytes": MetadataValue.int(avg_memory),
            "pod_memory_peak_pct": MetadataValue.float(round(self.memory_peak_pct, 2)),
            "pod_cpu_peak_pct": MetadataValue.float(round(self.cpu_peak_pct, 2)),
            "pod_cpu_avg_pct": MetadataValue.float(round(avg_cpu, 2)),
        }

        if self.memory_min_bytes is not None:
            metadata["pod_memory_min_bytes"] = MetadataValue.int(self.memory_min_bytes)

        # RSS (non-evictable process memory) breakdown
        if self.memory_rss_peak_bytes > 0:
            avg_rss = self.memory_rss_sum_bytes // self.sample_count if self.sample_count > 0 else 0
            metadata["pod_memory_rss_peak_bytes"] = MetadataValue.int(self.memory_rss_peak_bytes)
            metadata["pod_memory_rss_avg_bytes"] = MetadataValue.int(avg_rss)
            metadata["pod_memory_rss_peak_pct"] = MetadataValue.float(
                round(self.memory_rss_peak_pct, 2)
            )
            if self.memory_rss_min_bytes is not None:
                metadata["pod_memory_rss_min_bytes"] = MetadataValue.int(self.memory_rss_min_bytes)

        # Cache (evictable page cache + slab) breakdown
        if self.memory_cache_peak_bytes > 0:
            avg_cache = (
                self.memory_cache_sum_bytes // self.sample_count if self.sample_count > 0 else 0
            )
            metadata["pod_memory_cache_peak_bytes"] = MetadataValue.int(
                self.memory_cache_peak_bytes
            )
            metadata["pod_memory_cache_avg_bytes"] = MetadataValue.int(avg_cache)

        if self.cpu_min_pct is not None:
            metadata["pod_cpu_min_pct"] = MetadataValue.float(round(self.cpu_min_pct, 2))

        return metadata
