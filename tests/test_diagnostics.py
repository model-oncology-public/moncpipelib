"""Tests for the diagnostics subpackage."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from moncpipelib.diagnostics import (
    PodResourceSampler,
    ResourceSample,
    SamplerConfig,
    SamplerMode,
    SamplerSummary,
)
from moncpipelib.diagnostics.cgroup import CgroupReader

# ---------------------------------------------------------------------------
# SamplerMode
# ---------------------------------------------------------------------------


class TestSamplerMode:
    def test_metadata_only(self) -> None:
        mode = SamplerMode.METADATA
        assert bool(mode & SamplerMode.METADATA)
        assert not bool(mode & SamplerMode.LOG)

    def test_log_only(self) -> None:
        mode = SamplerMode.LOG
        assert not bool(mode & SamplerMode.METADATA)
        assert bool(mode & SamplerMode.LOG)

    def test_both(self) -> None:
        mode = SamplerMode.METADATA | SamplerMode.LOG
        assert bool(mode & SamplerMode.METADATA)
        assert bool(mode & SamplerMode.LOG)

    def test_flag_composition(self) -> None:
        combined = SamplerMode.METADATA | SamplerMode.LOG
        assert combined == SamplerMode.METADATA | SamplerMode.LOG


# ---------------------------------------------------------------------------
# SamplerConfig
# ---------------------------------------------------------------------------


class TestSamplerConfig:
    def test_default_values(self) -> None:
        cfg = SamplerConfig()
        assert cfg.interval_seconds == 5.0
        assert cfg.mode == SamplerMode.METADATA | SamplerMode.LOG
        assert cfg.logger_name == "moncpipelib.diagnostics"
        assert cfg.cgroup_base_path == "/sys/fs/cgroup"

    def test_frozen(self) -> None:
        cfg = SamplerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.interval_seconds = 10.0  # type: ignore[misc]

    def test_env_var_override_interval(self) -> None:
        with patch.dict("os.environ", {"MONCPIPELIB_DIAGNOSTICS_INTERVAL": "2.5"}):
            cfg = SamplerConfig()
            assert cfg.interval_seconds == 2.5

    def test_env_var_override_cgroup_path(self) -> None:
        with patch.dict("os.environ", {"MONCPIPELIB_DIAGNOSTICS_CGROUP_PATH": "/custom/cgroup"}):
            cfg = SamplerConfig()
            assert cfg.cgroup_base_path == "/custom/cgroup"

    def test_explicit_values_override_env(self) -> None:
        with patch.dict("os.environ", {"MONCPIPELIB_DIAGNOSTICS_INTERVAL": "99.0"}):
            cfg = SamplerConfig(interval_seconds=1.0)
            assert cfg.interval_seconds == 1.0


# ---------------------------------------------------------------------------
# ResourceSample
# ---------------------------------------------------------------------------


class TestResourceSample:
    def test_frozen(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            memory_bytes=1024,
            memory_limit_bytes=2048,
            cpu_usage_usec=100,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.memory_bytes = 0  # type: ignore[misc]

    def test_none_fields(self) -> None:
        sample = ResourceSample(
            timestamp=0.0,
            memory_bytes=None,
            memory_limit_bytes=None,
        )
        assert sample.memory_bytes is None
        assert sample.memory_limit_bytes is None
        assert sample.memory_rss_bytes is None
        assert sample.memory_cache_bytes is None
        assert sample.cpu_usage_usec is None


# ---------------------------------------------------------------------------
# SamplerSummary
# ---------------------------------------------------------------------------


class TestSamplerSummary:
    def test_default_values(self) -> None:
        s = SamplerSummary()
        assert s.sample_count == 0
        assert s.memory_peak_bytes == 0
        assert s.memory_min_bytes is None
        assert s.cpu_peak_pct == 0.0
        assert s.cpu_min_pct is None

    def test_to_metadata_returns_expected_keys(self) -> None:
        s = SamplerSummary(
            sample_count=10,
            duration_seconds=50.0,
            memory_peak_bytes=1_048_576,
            memory_min_bytes=524_288,
            memory_sum_bytes=7_340_032,
            memory_peak_pct=50.0,
            cpu_peak_pct=75.5,
            cpu_sum_pct=300.0,
            cpu_min_pct=10.2,
        )
        metadata = s.to_metadata()

        assert "pod_resource_samples" in metadata
        assert "pod_resource_duration_sec" in metadata
        assert "pod_memory_peak_bytes" in metadata
        assert "pod_memory_avg_bytes" in metadata
        assert "pod_memory_min_bytes" in metadata
        assert "pod_memory_peak_pct" in metadata
        assert "pod_cpu_peak_pct" in metadata
        assert "pod_cpu_avg_pct" in metadata
        assert "pod_cpu_min_pct" in metadata

    def test_to_metadata_computed_averages(self) -> None:
        s = SamplerSummary(
            sample_count=4,
            duration_seconds=20.0,
            memory_sum_bytes=4096,
            cpu_sum_pct=200.0,
        )
        metadata = s.to_metadata()
        # avg_memory = 4096 // 4 = 1024
        assert metadata["pod_memory_avg_bytes"].value == 1024
        # avg_cpu = 200.0 / 4 = 50.0
        assert metadata["pod_cpu_avg_pct"].value == 50.0

    def test_to_metadata_omits_none_min(self) -> None:
        s = SamplerSummary(sample_count=1, duration_seconds=5.0)
        metadata = s.to_metadata()
        assert "pod_memory_min_bytes" not in metadata
        assert "pod_cpu_min_pct" not in metadata

    def test_to_metadata_zero_samples(self) -> None:
        s = SamplerSummary()
        metadata = s.to_metadata()
        assert metadata["pod_memory_avg_bytes"].value == 0
        assert metadata["pod_cpu_avg_pct"].value == 0.0

    def test_to_metadata_includes_rss_and_cache(self) -> None:
        s = SamplerSummary(
            sample_count=4,
            duration_seconds=20.0,
            memory_rss_peak_bytes=400_000,
            memory_rss_min_bytes=200_000,
            memory_rss_sum_bytes=1_200_000,
            memory_rss_peak_pct=19.2,
            memory_cache_peak_bytes=1_700_000,
            memory_cache_sum_bytes=6_000_000,
        )
        metadata = s.to_metadata()

        assert metadata["pod_memory_rss_peak_bytes"].value == 400_000
        assert metadata["pod_memory_rss_avg_bytes"].value == 300_000
        assert metadata["pod_memory_rss_min_bytes"].value == 200_000
        assert metadata["pod_memory_rss_peak_pct"].value == 19.2
        assert metadata["pod_memory_cache_peak_bytes"].value == 1_700_000
        assert metadata["pod_memory_cache_avg_bytes"].value == 1_500_000

    def test_to_metadata_omits_rss_when_zero(self) -> None:
        s = SamplerSummary(sample_count=1, duration_seconds=5.0)
        metadata = s.to_metadata()
        assert "pod_memory_rss_peak_bytes" not in metadata
        assert "pod_memory_cache_peak_bytes" not in metadata


# ---------------------------------------------------------------------------
# CgroupReader
# ---------------------------------------------------------------------------


def _write_cgroup_files(
    base: Path,
    *,
    memory_current: str | None = None,
    memory_max: str | None = None,
    memory_stat: str | None = None,
    cpu_stat: str | None = None,
) -> None:
    """Helper to write mock cgroup files into a tmp directory."""
    if memory_current is not None:
        (base / "memory.current").write_text(memory_current)
    if memory_max is not None:
        (base / "memory.max").write_text(memory_max)
    if memory_stat is not None:
        (base / "memory.stat").write_text(memory_stat)
    if cpu_stat is not None:
        (base / "cpu.stat").write_text(cpu_stat)


class TestCgroupReader:
    def test_read_memory_bytes_cgroup(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="1073741824\n")
        reader = CgroupReader(str(tmp_path))
        assert reader.read_memory_bytes() == 1_073_741_824

    def test_read_memory_limit_bytes(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="100\n", memory_max="2147483648\n")
        reader = CgroupReader(str(tmp_path))
        assert reader.read_memory_limit_bytes() == 2_147_483_648

    def test_read_memory_limit_max_returns_none(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="100\n", memory_max="max\n")
        reader = CgroupReader(str(tmp_path))
        assert reader.read_memory_limit_bytes() is None

    def test_read_cpu_usage_usec(self, tmp_path: Path) -> None:
        cpu_content = "usage_usec 123456789\nuser_usec 100000000\nsystem_usec 23456789\n"
        _write_cgroup_files(tmp_path, cpu_stat=cpu_content)
        reader = CgroupReader(str(tmp_path))
        assert reader.read_cpu_usage_usec() == 123_456_789

    def test_no_cgroup_files_returns_none(self, tmp_path: Path) -> None:
        # tmp_path exists but has no cgroup files and no /proc
        reader = CgroupReader(str(tmp_path))
        # Override the proc fallback check too
        reader._has_proc_status = False
        assert reader.read_memory_bytes() is None
        assert reader.read_memory_limit_bytes() is None
        assert reader.read_cpu_usage_usec() is None

    def test_sample_returns_resource_sample(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="1048576\n",
            memory_max="2097152\n",
            cpu_stat="usage_usec 500000\n",
        )
        reader = CgroupReader(str(tmp_path))
        sample = reader.sample(42.0)
        assert isinstance(sample, ResourceSample)
        assert sample.timestamp == 42.0
        assert sample.memory_bytes == 1_048_576
        assert sample.memory_limit_bytes == 2_097_152
        assert sample.cpu_usage_usec == 500_000

    def test_read_memory_stat(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="2147106816\n",
            memory_stat=(
                "anon 412483584\nfile 1734901760\nkernel 8192000\nslab_reclaimable 5242880\n"
            ),
        )
        reader = CgroupReader(str(tmp_path))
        rss, cache = reader.read_memory_stat()
        assert rss == 412_483_584
        assert cache == 1_734_901_760 + 5_242_880

    def test_read_memory_stat_no_slab_reclaimable(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="100\n",
            memory_stat="anon 1000\nfile 2000\n",
        )
        reader = CgroupReader(str(tmp_path))
        rss, cache = reader.read_memory_stat()
        assert rss == 1000
        assert cache == 2000

    def test_read_memory_stat_unavailable(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="100\n")
        reader = CgroupReader(str(tmp_path))
        rss, cache = reader.read_memory_stat()
        assert rss is None
        assert cache is None

    def test_sample_includes_rss_and_cache(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="1048576\n",
            memory_max="2097152\n",
            memory_stat="anon 400000\nfile 600000\nslab_reclaimable 48576\n",
            cpu_stat="usage_usec 500000\n",
        )
        reader = CgroupReader(str(tmp_path))
        sample = reader.sample(42.0)
        assert sample.memory_rss_bytes == 400_000
        assert sample.memory_cache_bytes == 648_576

    def test_corrupted_file_returns_none(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="not_a_number\n")
        reader = CgroupReader(str(tmp_path))
        assert reader.read_memory_bytes() is None


# ---------------------------------------------------------------------------
# PodResourceSampler
# ---------------------------------------------------------------------------


class TestPodResourceSampler:
    def test_interval_validation(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be >= 0.5"):
            PodResourceSampler(SamplerConfig(interval_seconds=0.1))

    def test_default_config(self) -> None:
        sampler = PodResourceSampler()
        assert sampler.config.interval_seconds == 5.0

    def test_context_manager_starts_and_stops_thread(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="1048576\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        with PodResourceSampler(config) as sampler:
            assert sampler._thread is not None
            assert sampler._thread.is_alive()
            time.sleep(0.8)

        assert sampler._thread is not None
        assert not sampler._thread.is_alive()
        assert sampler.summary.sample_count >= 1

    def test_metadata_accumulates_memory(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="2097152\n",
            memory_max="4194304\n",
        )
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        with PodResourceSampler(config):
            time.sleep(1.2)

        # Can't read sampler from here -- use a reference
        # Re-test with explicit reference:
        sampler = PodResourceSampler(config)
        with sampler:
            time.sleep(1.2)

        assert sampler.summary.memory_peak_bytes == 2_097_152
        assert sampler.summary.memory_peak_pct == pytest.approx(50.0, abs=0.1)

    def test_metadata_accumulates_rss_and_cache(self, tmp_path: Path) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="2097152\n",
            memory_max="4194304\n",
            memory_stat="anon 400000\nfile 1600000\nslab_reclaimable 97152\n",
        )
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        with sampler:
            time.sleep(1.2)

        assert sampler.summary.memory_rss_peak_bytes == 400_000
        assert sampler.summary.memory_cache_peak_bytes == 1_697_152
        assert sampler.summary.memory_rss_peak_pct == pytest.approx(
            400_000 / 4_194_304 * 100.0, abs=0.1
        )

    def test_log_mode_includes_rss_fields(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="2097152\n",
            memory_max="4194304\n",
            memory_stat="anon 400000\nfile 1600000\nslab_reclaimable 97152\n",
        )
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        with (
            caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
            PodResourceSampler(config),
        ):
            time.sleep(0.8)

        json_lines = [r.message for r in caplog.records if "pod_resource_sample" in r.message]
        assert len(json_lines) >= 1
        parsed = json.loads(json_lines[0])
        assert parsed["memory_rss_bytes"] == 400_000
        assert parsed["memory_cache_bytes"] == 1_697_152
        assert "memory_rss_pct" in parsed

    def test_log_mode_emits_json(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_cgroup_files(
            tmp_path,
            memory_current="1048576\n",
            memory_max="2097152\n",
        )
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        with (
            caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
            PodResourceSampler(config),
        ):
            time.sleep(0.8)

        json_lines = [r.message for r in caplog.records if "pod_resource_sample" in r.message]
        assert len(json_lines) >= 1

        parsed = json.loads(json_lines[0])
        assert parsed["event"] == "pod_resource_sample"
        assert "memory_bytes" in parsed
        assert "memory_limit_bytes" in parsed

    def test_log_output_contains_only_allowed_keys(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        allowed_keys = {
            "event",
            "memory_bytes",
            "memory_limit_bytes",
            "memory_pct",
            "memory_rss_bytes",
            "memory_rss_pct",
            "memory_cache_bytes",
            "cpu_pct",
            "cpu_usage_usec",
        }
        _write_cgroup_files(
            tmp_path,
            memory_current="1048576\n",
            memory_max="2097152\n",
            cpu_stat="usage_usec 100000\n",
        )
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        with (
            caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
            PodResourceSampler(config),
        ):
            time.sleep(0.8)

        json_lines = [r.message for r in caplog.records if "pod_resource_sample" in r.message]
        for line in json_lines:
            parsed = json.loads(line)
            assert set(parsed.keys()).issubset(allowed_keys)

    def test_metadata_mode_does_not_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_cgroup_files(tmp_path, memory_current="1048576\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        with (
            caplog.at_level(logging.INFO, logger="moncpipelib.diagnostics"),
            PodResourceSampler(config),
        ):
            time.sleep(0.8)

        json_lines = [r.message for r in caplog.records if "pod_resource_sample" in r.message]
        assert len(json_lines) == 0

    def test_cpu_pct_computation(self) -> None:
        config = SamplerConfig(interval_seconds=1.0, mode=SamplerMode.METADATA)
        sampler = PodResourceSampler(config)

        prev = ResourceSample(
            timestamp=100.0,
            memory_bytes=None,
            memory_limit_bytes=None,
            cpu_usage_usec=1_000_000,
        )
        curr = ResourceSample(
            timestamp=101.0,
            memory_bytes=None,
            memory_limit_bytes=None,
            cpu_usage_usec=1_500_000,
        )

        sampler._prev_sample = prev
        result = sampler._compute_cpu_pct(curr)

        # 500_000 usec over 1_000_000 usec (1 sec) = 50%
        assert result is not None
        assert result == pytest.approx(50.0, abs=0.01)

    def test_cpu_pct_none_on_first_sample(self) -> None:
        config = SamplerConfig(interval_seconds=1.0, mode=SamplerMode.METADATA)
        sampler = PodResourceSampler(config)

        sample = ResourceSample(
            timestamp=100.0,
            memory_bytes=None,
            memory_limit_bytes=None,
            cpu_usage_usec=1_000_000,
        )
        assert sampler._compute_cpu_pct(sample) is None

    def test_cpu_pct_none_when_unavailable(self) -> None:
        config = SamplerConfig(interval_seconds=1.0, mode=SamplerMode.METADATA)
        sampler = PodResourceSampler(config)

        sample = ResourceSample(
            timestamp=100.0,
            memory_bytes=None,
            memory_limit_bytes=None,
            cpu_usage_usec=None,
        )
        assert sampler._compute_cpu_pct(sample) is None

    def test_exception_in_reader_does_not_crash(self, tmp_path: Path) -> None:
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        # Patch the reader to raise on every sample
        sampler._reader = MagicMock(spec=CgroupReader)
        sampler._reader.sample.side_effect = OSError("disk on fire")

        with sampler:
            time.sleep(0.8)

        # Thread should have survived; summary should be empty
        assert sampler._thread is not None
        assert not sampler._thread.is_alive()
        assert sampler.summary.sample_count == 0

    def test_summary_accessible_after_exit(self, tmp_path: Path) -> None:
        _write_cgroup_files(tmp_path, memory_current="4096\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        with sampler:
            time.sleep(0.8)

        assert sampler.summary.duration_seconds > 0
        assert sampler.summary.sample_count >= 1
        assert sampler.summary.memory_peak_bytes == 4096

    def test_log_mode_sets_level_when_effective_too_high(self, tmp_path: Path) -> None:
        """LOG mode lowers effective level to INFO so sample ticks are emitted."""
        _write_cgroup_files(tmp_path, memory_current="1024\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        log = logging.getLogger(config.logger_name)

        # Save and clear any explicit level so it inherits WARNING from root
        original_level = log.level
        log.setLevel(logging.NOTSET)
        try:
            assert log.getEffectiveLevel() >= logging.WARNING

            with sampler:
                # Level should now allow INFO
                assert log.getEffectiveLevel() <= logging.INFO
                time.sleep(0.6)

            # After exit, level should be restored
            assert log.level == logging.NOTSET
        finally:
            log.setLevel(original_level)

    def test_log_mode_adds_handler_when_none_exists(self, tmp_path: Path) -> None:
        """LOG mode auto-adds a StreamHandler when no handlers exist in hierarchy."""
        _write_cgroup_files(tmp_path, memory_current="1024\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        log = logging.getLogger(config.logger_name)

        # Isolate from parent handlers so hasHandlers() returns False
        original_propagate = log.propagate
        log.propagate = False
        original_handlers = list(log.handlers)
        for h in original_handlers:
            log.removeHandler(h)
        try:
            assert not log.hasHandlers()

            with sampler:
                assert sampler._added_handler is not None
                assert log.hasHandlers()
                time.sleep(0.6)

            # After exit, our handler should be removed
            assert sampler._added_handler is None
        finally:
            for h in original_handlers:
                log.addHandler(h)
            log.propagate = original_propagate

    def test_log_mode_skips_handler_when_already_configured(self, tmp_path: Path) -> None:
        """LOG mode does not add a handler when one already exists."""
        _write_cgroup_files(tmp_path, memory_current="1024\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.LOG,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        log = logging.getLogger(config.logger_name)

        existing = logging.NullHandler()
        log.addHandler(existing)
        try:
            with sampler:
                assert sampler._added_handler is None
                time.sleep(0.6)
        finally:
            log.removeHandler(existing)

    def test_metadata_only_mode_does_not_modify_logging(self, tmp_path: Path) -> None:
        """METADATA-only mode never adds handlers or changes level."""
        _write_cgroup_files(tmp_path, memory_current="1024\n")
        config = SamplerConfig(
            interval_seconds=0.5,
            mode=SamplerMode.METADATA,
            cgroup_base_path=str(tmp_path),
        )
        sampler = PodResourceSampler(config)
        with sampler:
            assert sampler._added_handler is None
            assert sampler._original_log_level is None
            time.sleep(0.6)
