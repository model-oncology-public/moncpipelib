"""Background pod resource sampler for Kubernetes workloads.

Spawns a daemon thread to periodically sample CPU and memory metrics from
cgroup v2 (or /proc fallback). Supports two composable modes:

- **METADATA**: Accumulates summary statistics for Dagster asset metadata.
- **LOG**: Emits structured JSON log lines at each sample tick.

Security note: This module emits only infrastructure metrics (memory bytes,
CPU percentages, sample counts, durations). No PHI, application data, or
credentials are included in log output or metadata.

Example::

    from moncpipelib.diagnostics import PodResourceSampler, SamplerConfig

    with PodResourceSampler() as sampler:
        # ... pipeline work ...
        pass
    context.add_output_metadata(sampler.summary.to_metadata())
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from types import TracebackType
from typing import Any

from moncpipelib.diagnostics.cgroup import CgroupReader
from moncpipelib.diagnostics.types import (
    ResourceSample,
    SamplerConfig,
    SamplerMode,
    SamplerSummary,
)

logger = logging.getLogger(__name__)


class PodResourceSampler:
    """Background resource sampler that runs as a context manager.

    Spawns a daemon thread that reads cgroup v2 / proc metrics at a
    configurable interval. The daemon thread is marked ``daemon=True``
    so it dies automatically if the process exits.

    Thread safety: the summary accumulator is protected by a
    ``threading.Lock``. The cgroup reader is used only by the daemon thread.
    """

    def __init__(self, config: SamplerConfig | None = None) -> None:
        self._config = config or SamplerConfig()

        if self._config.interval_seconds < 0.5:
            msg = f"interval_seconds must be >= 0.5, got {self._config.interval_seconds}"
            raise ValueError(msg)

        self._reader = CgroupReader(self._config.cgroup_base_path)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._summary = SamplerSummary()
        self._prev_sample: ResourceSample | None = None
        self._start_time: float = 0.0
        self._log = logging.getLogger(self._config.logger_name)
        self._added_handler: logging.Handler | None = None
        self._original_log_level: int | None = None

    @property
    def config(self) -> SamplerConfig:
        """Return the sampler configuration."""
        return self._config

    @property
    def summary(self) -> SamplerSummary:
        """Return the accumulated summary statistics.

        Safe to read after the context manager has exited. Reading while
        the sampler is running is also safe (lock-protected) but values
        will be partial.
        """
        return self._summary

    def __enter__(self) -> PodResourceSampler:
        """Start the background sampling thread."""
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._summary = SamplerSummary()
        self._prev_sample = None

        # Ensure log output is possible when LOG mode is active.
        # In Dagster K8s pods the root logger typically has no handlers and
        # a default level of WARNING, so INFO-level sample ticks would be
        # silently dropped without intervention.
        if bool(self._config.mode & SamplerMode.LOG):
            # 1) Ensure effective level allows INFO records.  The logger's
            #    own level may be NOTSET (inheriting WARNING from root).
            if self._log.getEffectiveLevel() > logging.INFO:
                self._original_log_level = self._log.level  # save for restore
                self._log.setLevel(logging.INFO)

            # 2) Ensure at least one handler exists.  If parents have
            #    handlers the records will propagate; otherwise we add a
            #    StreamHandler(stderr) so lines appear in pod logs.
            if not self._log.hasHandlers():
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._log.addHandler(handler)
                self._added_handler = handler

        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="moncpipelib-pod-sampler",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "PodResourceSampler started (interval=%.1fs, mode=%s)",
            self._config.interval_seconds,
            self._config.mode,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Signal the sampling thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._config.interval_seconds + 2.0)
            if self._thread.is_alive():
                logger.warning("PodResourceSampler thread did not stop within timeout")
        with self._lock:
            self._summary.duration_seconds = time.monotonic() - self._start_time

        # Restore logging config we changed in __enter__ to avoid side effects.
        if self._added_handler is not None:
            self._log.removeHandler(self._added_handler)
            self._added_handler = None
        if self._original_log_level is not None:
            self._log.setLevel(self._original_log_level)
            self._original_log_level = None

        logger.debug(
            "PodResourceSampler stopped (samples=%d, duration=%.1fs)",
            self._summary.sample_count,
            self._summary.duration_seconds,
        )

    def _sampling_loop(self) -> None:
        """Main loop for the daemon sampling thread."""
        while not self._stop_event.is_set():
            try:
                now = time.monotonic()
                sample = self._reader.sample(now)
                cpu_pct = self._compute_cpu_pct(sample)

                if bool(self._config.mode & SamplerMode.METADATA):
                    self._update_summary(sample, cpu_pct)

                if bool(self._config.mode & SamplerMode.LOG):
                    self._emit_log_line(sample, cpu_pct)

                self._prev_sample = sample

            except Exception:
                logger.debug("PodResourceSampler tick failed", exc_info=True)

            self._stop_event.wait(timeout=self._config.interval_seconds)

    def _compute_cpu_pct(self, sample: ResourceSample) -> float | None:
        """Compute CPU percentage from delta of usage_usec between samples.

        Returns None for the first sample or when CPU metrics are unavailable.
        """
        if sample.cpu_usage_usec is None:
            return None
        if self._prev_sample is None or self._prev_sample.cpu_usage_usec is None:
            return None

        delta_usec = sample.cpu_usage_usec - self._prev_sample.cpu_usage_usec
        delta_time = sample.timestamp - self._prev_sample.timestamp

        if delta_time <= 0:
            return None

        delta_time_usec = delta_time * 1_000_000
        return (delta_usec / delta_time_usec) * 100.0

    def _update_summary(self, sample: ResourceSample, cpu_pct: float | None) -> None:
        """Update the accumulated summary under lock."""
        with self._lock:
            self._summary.sample_count += 1

            if sample.memory_bytes is not None:
                mem = sample.memory_bytes
                self._summary.memory_sum_bytes += mem
                if mem > self._summary.memory_peak_bytes:
                    self._summary.memory_peak_bytes = mem
                if self._summary.memory_min_bytes is None or mem < self._summary.memory_min_bytes:
                    self._summary.memory_min_bytes = mem

                limit = sample.memory_limit_bytes
                if limit is not None and limit > 0:
                    mem_pct = (mem / limit) * 100.0
                    if mem_pct > self._summary.memory_peak_pct:
                        self._summary.memory_peak_pct = mem_pct

            if sample.memory_rss_bytes is not None:
                rss = sample.memory_rss_bytes
                self._summary.memory_rss_sum_bytes += rss
                if rss > self._summary.memory_rss_peak_bytes:
                    self._summary.memory_rss_peak_bytes = rss
                if (
                    self._summary.memory_rss_min_bytes is None
                    or rss < self._summary.memory_rss_min_bytes
                ):
                    self._summary.memory_rss_min_bytes = rss

                limit = sample.memory_limit_bytes
                if limit is not None and limit > 0:
                    rss_pct = (rss / limit) * 100.0
                    if rss_pct > self._summary.memory_rss_peak_pct:
                        self._summary.memory_rss_peak_pct = rss_pct

            if sample.memory_cache_bytes is not None:
                cache = sample.memory_cache_bytes
                self._summary.memory_cache_sum_bytes += cache
                if cache > self._summary.memory_cache_peak_bytes:
                    self._summary.memory_cache_peak_bytes = cache

            if cpu_pct is not None:
                self._summary.cpu_sum_pct += cpu_pct
                if cpu_pct > self._summary.cpu_peak_pct:
                    self._summary.cpu_peak_pct = cpu_pct
                if self._summary.cpu_min_pct is None or cpu_pct < self._summary.cpu_min_pct:
                    self._summary.cpu_min_pct = cpu_pct

    def _emit_log_line(self, sample: ResourceSample, cpu_pct: float | None) -> None:
        """Emit a structured JSON log line for the current sample."""
        record: dict[str, Any] = {"event": "pod_resource_sample"}

        if sample.memory_bytes is not None:
            record["memory_bytes"] = sample.memory_bytes
        if sample.memory_limit_bytes is not None:
            record["memory_limit_bytes"] = sample.memory_limit_bytes
            if sample.memory_bytes is not None:
                record["memory_pct"] = round(
                    (sample.memory_bytes / sample.memory_limit_bytes) * 100.0, 2
                )
        if sample.memory_rss_bytes is not None:
            record["memory_rss_bytes"] = sample.memory_rss_bytes
            if sample.memory_limit_bytes is not None and sample.memory_limit_bytes > 0:
                record["memory_rss_pct"] = round(
                    (sample.memory_rss_bytes / sample.memory_limit_bytes) * 100.0, 2
                )
        if sample.memory_cache_bytes is not None:
            record["memory_cache_bytes"] = sample.memory_cache_bytes
        if cpu_pct is not None:
            record["cpu_pct"] = round(cpu_pct, 2)
        if sample.cpu_usage_usec is not None:
            record["cpu_usage_usec"] = sample.cpu_usage_usec

        self._log.info(json.dumps(record, separators=(",", ":")))
