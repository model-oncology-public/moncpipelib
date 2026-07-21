"""Low-level cgroup v2 metric readers with /proc fallback.

Security note: This module reads only from /sys/fs/cgroup and /proc/self/status.
No PHI or application data is accessed. All file paths are constructed from a
validated base path, not from user input.
"""

from __future__ import annotations

import logging
from pathlib import Path

from moncpipelib.diagnostics.types import ResourceSample

logger = logging.getLogger(__name__)


class CgroupReader:
    """Reads resource metrics from the cgroup v2 filesystem.

    Falls back to /proc/self/status VmRSS when cgroup files are unavailable
    (local development, non-Linux platforms).
    """

    def __init__(self, base_path: str = "/sys/fs/cgroup") -> None:
        self._base = Path(base_path)
        self._memory_current = self._base / "memory.current"
        self._memory_max = self._base / "memory.max"
        self._memory_stat = self._base / "memory.stat"
        self._cpu_stat = self._base / "cpu.stat"
        self._proc_status = Path("/proc/self/status")

        self._has_cgroup_memory = self._memory_current.is_file()
        self._has_memory_stat = self._memory_stat.is_file()
        self._has_cgroup_cpu = self._cpu_stat.is_file()
        self._has_proc_status = self._proc_status.is_file()

        if not self._has_cgroup_memory:
            if self._has_proc_status:
                logger.info(
                    "cgroup memory.current not found; falling back to /proc/self/status VmRSS"
                )
            else:
                logger.info(
                    "Neither cgroup memory files nor /proc/self/status found; "
                    "memory metrics will be unavailable"
                )
        if not self._has_cgroup_cpu:
            logger.info("cgroup cpu.stat not found; CPU metrics will be unavailable")

    def read_memory_bytes(self) -> int | None:
        """Read current memory usage in bytes.

        Tries cgroup v2 ``memory.current`` first, then ``/proc/self/status`` VmRSS.
        """
        if self._has_cgroup_memory:
            return self._read_cgroup_memory()
        if self._has_proc_status:
            return self._read_proc_vmrss()
        return None

    def read_memory_limit_bytes(self) -> int | None:
        """Read the memory limit from cgroup v2.

        Returns None if unavailable or set to ``max`` (no limit).
        """
        if not self._has_cgroup_memory:
            return None
        try:
            text = self._memory_max.read_text().strip()
            if text == "max":
                return None
            return int(text)
        except (OSError, ValueError):
            return None

    def read_memory_stat(self) -> tuple[int | None, int | None]:
        """Read RSS and cache breakdown from cgroup v2 ``memory.stat``.

        Returns:
            A tuple of ``(rss_bytes, cache_bytes)`` where:

            - *rss_bytes* is the ``anon`` counter -- process heap, stack, and
              anonymous mmap allocations (non-evictable under memory pressure).
            - *cache_bytes* is ``file + slab_reclaimable`` -- filesystem page
              cache plus reclaimable kernel slab objects.  Both are evictable
              by the kernel under memory pressure and do not represent true
              memory consumption.

            Either value may be ``None`` if ``memory.stat`` is unavailable.
        """
        if not self._has_memory_stat:
            return None, None
        try:
            text = self._memory_stat.read_text()
            anon: int | None = None
            file_cache: int | None = None
            slab_reclaimable: int | None = None
            for line in text.splitlines():
                if line.startswith("anon "):
                    anon = int(line.split()[1])
                elif line.startswith("file "):
                    file_cache = int(line.split()[1])
                elif line.startswith("slab_reclaimable "):
                    slab_reclaimable = int(line.split()[1])
            rss = anon
            cache = None
            if file_cache is not None:
                cache = file_cache + (slab_reclaimable or 0)
            return rss, cache
        except (OSError, ValueError):
            return None, None

    def read_cpu_usage_usec(self) -> int | None:
        """Read cumulative CPU usage in microseconds from cgroup v2 ``cpu.stat``."""
        if not self._has_cgroup_cpu:
            return None
        try:
            text = self._cpu_stat.read_text()
            for line in text.splitlines():
                if line.startswith("usage_usec"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
            return None
        except (OSError, ValueError):
            return None

    def sample(self, timestamp: float) -> ResourceSample:
        """Collect a single resource sample at the given monotonic timestamp."""
        rss_bytes, cache_bytes = self.read_memory_stat()
        return ResourceSample(
            timestamp=timestamp,
            memory_bytes=self.read_memory_bytes(),
            memory_limit_bytes=self.read_memory_limit_bytes(),
            memory_rss_bytes=rss_bytes,
            memory_cache_bytes=cache_bytes,
            cpu_usage_usec=self.read_cpu_usage_usec(),
        )

    def _read_cgroup_memory(self) -> int | None:
        try:
            return int(self._memory_current.read_text().strip())
        except (OSError, ValueError):
            return None

    def _read_proc_vmrss(self) -> int | None:
        """Read VmRSS from /proc/self/status (reported in kB, converted to bytes)."""
        try:
            text = self._proc_status.read_text()
            for line in text.splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
            return None
        except (OSError, ValueError):
            return None
