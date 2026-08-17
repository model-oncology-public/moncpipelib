"""Tests for the ingest template engine.

Covers ``render_prefix`` (existing behavior) and
``render_payload_filename`` (added per #270).  Both share the bounded
placeholder set in :data:`moncpipelib.ingest.prefix._ALLOWED_PLACEHOLDERS`
-- the tests exercise both rendering and the unknown-placeholder error
path so reviewers can verify the same vocabulary applies to both.
"""

from __future__ import annotations

import pytest

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.prefix import render_payload_filename, render_prefix


def _contract(source_name: str = "demo") -> IngestContract:
    """Minimal IngestContract suitable for template rendering."""
    return IngestContract(
        source_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        source_name=source_name,
        sensitivity="public",
        pattern="http_urls",
        prefix_template="ignored",
        extract=(),
        strip_extensions=(),
        pattern_config={},
    )


class TestRenderPrefix:
    """Existing renderer; regression-pinned after the template engine refactor."""

    def test_renders_partition_key(self) -> None:
        out = render_prefix("demo/{partition_key}", "2026_q1", _contract())
        assert out == "demo/2026_q1"

    def test_renders_source_name(self) -> None:
        out = render_prefix("{source_name}/{partition_key}", "v1", _contract(source_name="alpha"))
        assert out == "alpha/v1"

    def test_unknown_placeholder_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown placeholder"):
            render_prefix("demo/{nope}", "v1", _contract())


class TestRenderPayloadFilename:
    """New renderer (#270) for non-archive payload filename templates."""

    def test_renders_partition_key(self) -> None:
        out = render_payload_filename("feed_{partition_key}.csv", "2026-04-19", _contract())
        assert out == "feed_2026-04-19.csv"

    def test_renders_source_name(self) -> None:
        out = render_payload_filename(
            "{source_name}.csv", "ignored", _contract(source_name="seer_cpc_smvl")
        )
        assert out == "seer_cpc_smvl.csv"

    def test_renders_both_placeholders(self) -> None:
        out = render_payload_filename(
            "{source_name}_{partition_key}.csv",
            "V2024B",
            _contract(source_name="seer_cpc_smvl"),
        )
        assert out == "seer_cpc_smvl_V2024B.csv"

    def test_unknown_placeholder_raises(self) -> None:
        """Symmetric with prefix template: extending the placeholder
        vocabulary is a conscious choice -- typos surface as ValueError."""
        with pytest.raises(ValueError, match="Unknown placeholder"):
            render_payload_filename("{release_date}.csv", "v1", _contract())

    def test_unknown_placeholder_error_names_field(self) -> None:
        """Error message cites 'payload filename template' so authors
        can tell which field rejected the placeholder."""
        with pytest.raises(ValueError, match="payload filename template"):
            render_payload_filename("{nope}.csv", "v1", _contract())
