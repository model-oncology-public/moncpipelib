"""Tests for central configuration module."""

import os
from unittest.mock import patch

import pytest


class TestMoncpipelibConfig:
    """Tests for MoncpipelibConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from moncpipelib.config import MoncpipelibConfig

        config = MoncpipelibConfig()

        assert config.openlineage.namespace == "moncpipelib"
        assert config.lineage.table_name == "data_lineage"
        assert config.lineage.schema_name == "lineage"

    def test_openlineage_schema_url_default(self):
        """Test default OpenLineage schema URL points to repo schemas."""
        from moncpipelib.config import MoncpipelibConfig

        config = MoncpipelibConfig()

        # Custom facet schemas are hosted in the moncpipelib repository
        assert "model-oncology-public/moncpipelib" in config.openlineage.schema_url_base
        assert "schemas/openlineage" in config.openlineage.schema_url_base

    def test_config_singleton_import(self):
        """Test that config singleton is accessible from main package."""
        from moncpipelib import config

        assert config is not None
        assert hasattr(config, "openlineage")
        assert hasattr(config, "lineage")


class TestEnvironmentVariableOverrides:
    """Tests for environment variable configuration overrides."""

    def test_openlineage_namespace_from_env(self):
        """Test OpenLineage namespace can be overridden via env var."""
        with patch.dict(os.environ, {"MONCPIPELIB_OPENLINEAGE_NAMESPACE": "custom-namespace"}):
            from moncpipelib.config import OpenLineageDefaults

            defaults = OpenLineageDefaults()
            assert defaults.namespace == "custom-namespace"

    def test_openlineage_schema_url_from_env(self):
        """Test OpenLineage schema URL can be overridden via env var."""
        with patch.dict(
            os.environ,
            {"MONCPIPELIB_OPENLINEAGE_SCHEMA_URL": "https://custom.example.com/schemas/"},
        ):
            from moncpipelib.config import OpenLineageDefaults

            defaults = OpenLineageDefaults()
            assert defaults.schema_url_base == "https://custom.example.com/schemas/"

    def test_lineage_table_from_env(self):
        """Test lineage table name can be overridden via env var."""
        with patch.dict(os.environ, {"MONCPIPELIB_LINEAGE_TABLE": "custom_lineage"}):
            from moncpipelib.config import LineageDefaults

            defaults = LineageDefaults()
            assert defaults.table_name == "custom_lineage"

    def test_lineage_schema_from_env(self):
        """Test lineage schema can be overridden via env var."""
        with patch.dict(os.environ, {"MONCPIPELIB_LINEAGE_SCHEMA": "custom_schema"}):
            from moncpipelib.config import LineageDefaults

            defaults = LineageDefaults()
            assert defaults.schema_name == "custom_schema"


class TestConfigImmutability:
    """Tests for configuration immutability."""

    def test_config_is_frozen(self):
        """Test that config dataclasses are frozen."""
        from moncpipelib.config import MoncpipelibConfig

        config = MoncpipelibConfig()

        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            config.openlineage = None  # type: ignore


class TestConfigModuleExports:
    """Tests for config module exports."""

    def test_exports_from_main_package(self):
        """Test config and MoncpipelibConfig are exported from main package."""
        from moncpipelib import MoncpipelibConfig, config

        assert MoncpipelibConfig is not None
        assert config is not None
        assert isinstance(config, MoncpipelibConfig)


class TestVerboseMetadataToggle:
    """Tests for the runtime ``VERBOSE_METADATA`` toggle helpers (#260).

    These cover the ergonomic surface intended for in-pipeline use:
    ``set_verbose_metadata(True/False)`` for process-wide flips and
    ``with verbose_metadata():`` for block-scoped diagnostics.  The
    integration with ``PostgresResource._write_batched`` is tested in
    ``test_postgres_resource.py``.
    """

    @pytest.fixture(autouse=True)
    def _restore_flag(self):
        """Snapshot and restore ``VERBOSE_METADATA`` around each test
        so test order does not leak state.  ``set_verbose_metadata``
        and the context manager mutate module-level state, so we have
        to be careful not to leave the flag flipped on for downstream
        tests."""
        import sys

        cfg = sys.modules["moncpipelib.config"]
        original = cfg.VERBOSE_METADATA
        try:
            yield
        finally:
            cfg.VERBOSE_METADATA = original

    def test_set_verbose_metadata_flips_flag(self):
        import sys

        from moncpipelib.config import set_verbose_metadata

        cfg = sys.modules["moncpipelib.config"]

        set_verbose_metadata(True)
        assert cfg.VERBOSE_METADATA is True

        set_verbose_metadata(False)
        assert cfg.VERBOSE_METADATA is False

    def test_set_verbose_metadata_default_arg_is_true(self):
        """Calling ``set_verbose_metadata()`` with no args turns it ON,
        matching the common ``set_verbose_metadata()`` ergonomics."""
        import sys

        from moncpipelib.config import set_verbose_metadata

        cfg = sys.modules["moncpipelib.config"]
        cfg.VERBOSE_METADATA = False

        set_verbose_metadata()
        assert cfg.VERBOSE_METADATA is True

    def test_verbose_metadata_context_manager_scopes_change(self):
        import sys

        from moncpipelib.config import verbose_metadata

        cfg = sys.modules["moncpipelib.config"]
        cfg.VERBOSE_METADATA = False

        with verbose_metadata():
            assert cfg.VERBOSE_METADATA is True

        assert cfg.VERBOSE_METADATA is False, (
            "context manager must restore the previous value on exit"
        )

    def test_verbose_metadata_context_manager_restores_on_exception(self):
        """The flag must be restored even if the block raises -- this is
        the test that justifies using ``try/finally`` over a plain
        assignment in ``verbose_metadata()``."""
        import sys

        from moncpipelib.config import verbose_metadata

        cfg = sys.modules["moncpipelib.config"]
        cfg.VERBOSE_METADATA = False

        with pytest.raises(RuntimeError, match="boom"), verbose_metadata():
            assert cfg.VERBOSE_METADATA is True
            raise RuntimeError("boom")

        assert cfg.VERBOSE_METADATA is False

    def test_verbose_metadata_context_manager_can_disable(self):
        """``verbose_metadata(False)`` should scope the flag OFF -- not
        common but completes the API surface."""
        import sys

        from moncpipelib.config import verbose_metadata

        cfg = sys.modules["moncpipelib.config"]
        cfg.VERBOSE_METADATA = True

        with verbose_metadata(False):
            assert cfg.VERBOSE_METADATA is False

        assert cfg.VERBOSE_METADATA is True

    def test_helpers_are_re_exported_from_main_package(self):
        """Pipeline code should be able to ``from moncpipelib import
        set_verbose_metadata`` without reaching into submodules.  This
        is the obvious-import promise (#260 follow-up)."""
        from moncpipelib import set_verbose_metadata, verbose_metadata
        from moncpipelib.config import (
            set_verbose_metadata as _set_from_module,
        )
        from moncpipelib.config import (
            verbose_metadata as _ctx_from_module,
        )

        assert set_verbose_metadata is _set_from_module
        assert verbose_metadata is _ctx_from_module
