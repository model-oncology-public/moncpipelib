"""Tests for ``moncpipelib.resources.types``.

Covers the shared ``WriteContext`` factory classmethods. Migration 018
(#309) adds backfill signals and Dagster asset-graph capture; the bulk of
the tests below pin that behaviour, since downstream phases (lineage row
auto-population) consume those fields.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from moncpipelib.resources.types import (
    WriteContext,
    _extract_backfill_signals,
    _extract_dagster_handles,
    _normalize_asset_deps,
)

# ---------------------------------------------------------------------------
# Backfill-signal extraction
# ---------------------------------------------------------------------------


class TestExtractBackfillSignals:
    """Tests for ``_extract_backfill_signals``."""

    def test_backfill_run_returns_id_and_true(self) -> None:
        """A run with the ``dagster/backfill`` tag and a ``backfill_id``
        attribute should produce ``(backfill_id, True)``."""
        ctx = MagicMock()
        ctx.run.backfill_id = "bf_2026_05_22_claims"
        ctx.run.tags = {"dagster/backfill": "bf_2026_05_22_claims"}

        assert _extract_backfill_signals(ctx) == ("bf_2026_05_22_claims", True)

    def test_normal_run_returns_none_and_false(self) -> None:
        """A run without the backfill tag and without a backfill id should
        produce ``(None, False)``."""
        ctx = MagicMock()
        ctx.run.backfill_id = None
        ctx.run.tags = {"dagster/job_name": "claims_pipeline"}

        assert _extract_backfill_signals(ctx) == (None, False)

    def test_missing_backfill_id_attribute_does_not_raise(self) -> None:
        """Older Dagster versions may not expose ``run.backfill_id``; the
        helper must degrade to ``None`` instead of raising ``AttributeError``."""

        class _OldRun:
            tags: dict[str, str] = {}

        class _OldContext:
            run = _OldRun()

        ctx: Any = _OldContext()
        backfill_id, is_backfill = _extract_backfill_signals(ctx)
        assert backfill_id is None
        assert is_backfill is False

    def test_missing_run_attribute_does_not_raise(self) -> None:
        """Some unit-test doubles do not expose ``.run`` at all. The
        helper must return ``(None, False)`` cleanly."""

        class _NoRun:
            pass

        ctx: Any = _NoRun()
        assert _extract_backfill_signals(ctx) == (None, False)

    def test_tag_presence_not_truth_is_the_signal(self) -> None:
        """``is_backfill`` reflects *presence* of the ``dagster/backfill``
        tag, not the truthiness of its value. An empty-string tag value
        still means the run is part of a backfill."""
        ctx = MagicMock()
        ctx.run.backfill_id = "bf_abc"
        ctx.run.tags = {"dagster/backfill": ""}  # empty value, but tag is set

        _, is_backfill = _extract_backfill_signals(ctx)
        assert is_backfill is True

    def test_bare_magicmock_context_degrades_safely(self) -> None:
        """Regression: a ``MagicMock`` context whose ``run.backfill_id`` /
        ``run.tags`` are NOT explicitly pinned must degrade to
        ``(None, False)``. Without this guard ``MagicMock``'s
        auto-attribute behaviour returns child mocks, and Phase 2 then
        passes those mocks into ``MetadataValue.text``, which Dagster
        rejects at runtime. Caught 69 integration test failures post-
        merge of Phase 2."""
        ctx = MagicMock()  # no explicit attribute setup on .run.*

        backfill_id, is_backfill = _extract_backfill_signals(ctx)

        assert backfill_id is None, "MagicMock backfill_id must NOT leak through as a value"
        assert is_backfill is False, "MagicMock tags must NOT be treated as a real Mapping"

    def test_non_str_backfill_id_rejected(self) -> None:
        """Defensive: a non-``str`` ``backfill_id`` (int, list, etc.) is
        rejected even when explicitly set, since downstream Dagster
        metadata requires a string."""
        ctx = MagicMock()
        ctx.run.backfill_id = 12345  # type: ignore[assignment]
        ctx.run.tags = {}

        backfill_id, _ = _extract_backfill_signals(ctx)
        assert backfill_id is None

    def test_non_mapping_tags_rejected(self) -> None:
        """Defensive: a non-``Mapping`` ``tags`` value never triggers
        ``is_backfill=True``."""
        ctx = MagicMock()
        ctx.run.backfill_id = None
        ctx.run.tags = ["dagster/backfill"]  # list, not Mapping

        _, is_backfill = _extract_backfill_signals(ctx)
        assert is_backfill is False

    def test_asset_rematerialization_falls_back_to_tag_value(self) -> None:
        """Issue #334 Bug 1: asset-rematerialization-flavoured backfills
        leave ``run.backfill_id`` as ``None`` and put the canonical
        backfill id in the ``dagster/backfill`` *tag value*. The helper
        must surface the tag value as the id."""
        ctx = MagicMock()
        ctx.run.backfill_id = None
        ctx.run.tags = {"dagster/backfill": "bf_remat_2026_05"}

        assert _extract_backfill_signals(ctx) == ("bf_remat_2026_05", True)

    def test_explicit_backfill_id_wins_over_tag_value(self) -> None:
        """Precedence: when both ``run.backfill_id`` and the tag value
        are populated, the attribute takes precedence so older Dagster
        shapes that set both stay stable."""
        ctx = MagicMock()
        ctx.run.backfill_id = "bf_attr"
        ctx.run.tags = {"dagster/backfill": "bf_tag"}

        assert _extract_backfill_signals(ctx) == ("bf_attr", True)

    def test_empty_tag_value_does_not_become_id(self) -> None:
        """An empty-string tag value still makes ``is_backfill=True`` (the
        existing ``test_tag_presence_not_truth_is_the_signal`` invariant)
        but must NOT be promoted to ``backfill_id``. Dagster uses the
        empty value as a "this run is part of a backfill but the id is
        elsewhere" marker, not as the id itself."""
        ctx = MagicMock()
        ctx.run.backfill_id = None
        ctx.run.tags = {"dagster/backfill": ""}

        assert _extract_backfill_signals(ctx) == (None, True)

    def test_non_str_tag_value_rejected(self) -> None:
        """Defensive: an integer-valued tag does not leak through as a
        ``backfill_id``. The downstream surface
        (``data_lineage.backfill_id`` text column / Dagster
        ``MetadataValue.text``) requires a string."""
        ctx = MagicMock()
        ctx.run.backfill_id = None
        ctx.run.tags = {"dagster/backfill": 42}  # type: ignore[dict-item]

        assert _extract_backfill_signals(ctx) == (None, True)

    def test_run_property_raising_degrades_to_none(self) -> None:
        """Issue #341 audit: ``OpExecutionContext.run`` /
        ``AssetExecutionContext.run`` are ``@property`` descriptors that
        delegate to ``self._step_execution_context.dagster_run``; on an
        ephemeral / partially-constructed context that delegation can
        raise something other than ``AttributeError``. ``getattr(..., None)``
        only substitutes the default on ``AttributeError``, so the helper
        must wrap the read in ``try`` / ``except`` and degrade to
        ``(None, False)``.

        A ``MagicMock`` fixture cannot exercise this -- auto-attribute
        access on a mock never raises -- so this uses a hand-rolled class
        whose ``run`` property raises.
        """

        class _CheckError(Exception):
            """Stand-in for the Dagster/check error raised when a context's
            step-execution-context is not available. The helper catches
            bare ``Exception``, so the exact class is irrelevant."""

        class _CtxWithRaisingRun:
            @property
            def run(self) -> object:
                raise _CheckError("No step execution context available")

        ctx: Any = _CtxWithRaisingRun()
        assert _extract_backfill_signals(ctx) == (None, False)


# ---------------------------------------------------------------------------
# Asset-deps normalisation
# ---------------------------------------------------------------------------


class _FakeAssetKey:
    """Minimal AssetKey stand-in exposing ``to_user_string()``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def to_user_string(self) -> str:
        return self._name


class TestNormalizeAssetDeps:
    """Tests for ``_normalize_asset_deps``."""

    def test_normalises_asset_key_dict(self) -> None:
        """``dict[AssetKey, list[AssetKey]]`` should round-trip to
        ``dict[str, list[str]]`` using ``AssetKey.to_user_string()``."""
        ctx = MagicMock()
        ctx.asset_deps = {
            _FakeAssetKey("silver/dim_provider"): [
                _FakeAssetKey("bronze/provider_raw"),
                _FakeAssetKey("bronze/npi_raw"),
            ],
            _FakeAssetKey("gold/fact_claims"): [],
        }

        result = _normalize_asset_deps(ctx)

        assert result == {
            "silver/dim_provider": ["bronze/provider_raw", "bronze/npi_raw"],
            "gold/fact_claims": [],
        }

    def test_missing_attribute_returns_none(self) -> None:
        """If ``asset_deps`` is unavailable, return ``None`` rather than
        a partial dict."""

        class _NoDeps:
            pass

        ctx: Any = _NoDeps()
        assert _normalize_asset_deps(ctx) is None

    def test_none_attribute_returns_none(self) -> None:
        """A literal ``None`` on ``asset_deps`` is propagated."""
        ctx = MagicMock()
        ctx.asset_deps = None
        assert _normalize_asset_deps(ctx) is None

    def test_malformed_deps_returns_none(self) -> None:
        """If ``asset_deps`` is not iterable in the expected shape, the
        helper returns ``None`` rather than raising."""
        ctx = MagicMock()
        ctx.asset_deps = "not-a-dict"  # malformed
        assert _normalize_asset_deps(ctx) is None

    def test_bare_magicmock_asset_deps_rejected(self) -> None:
        """Regression: a ``MagicMock`` context whose ``asset_deps`` is
        NOT explicitly pinned must degrade to ``None``. Without the
        ``Mapping`` type-check, ``MagicMock``'s auto-iterator behaviour
        produces an empty dict, masking the intended "asset graph
        unavailable" signal."""
        ctx = MagicMock()  # no .asset_deps setup
        assert _normalize_asset_deps(ctx) is None

    def test_asset_deps_property_raising_degrades_to_none(self) -> None:
        """Issue #341 audit: in the pinned Dagster (1.13.x) ``asset_deps``
        is absent (covered by ``test_missing_attribute_returns_none``), but
        other Dagster versions expose it as a ``@property`` whose getter can
        raise on an op context with no assets definition. ``getattr(..., None)``
        only swallows ``AttributeError``, so the read is guarded and a
        raising getter degrades to ``None``.

        ``MagicMock`` cannot model a raising descriptor, so this uses a
        hand-rolled class whose ``asset_deps`` property raises.
        """

        class _DagsterInvalidPropertyError(Exception):
            """Stand-in for the Dagster property error; the helper catches
            bare ``Exception`` so the exact class is irrelevant."""

        class _CtxWithRaisingAssetDeps:
            @property
            def asset_deps(self) -> object:
                raise _DagsterInvalidPropertyError(
                    "Op 'plain_op' does not have an assets definition."
                )

        ctx: Any = _CtxWithRaisingAssetDeps()
        assert _normalize_asset_deps(ctx) is None


# ---------------------------------------------------------------------------
# WriteContext.from_output_context
# ---------------------------------------------------------------------------


def _build_output_context(
    *,
    asset_name: str = "silver/dim_provider",
    run_id: str = "run-001",
    has_partition_key: bool = False,
    partition_keys: list[str] | None = None,
    partition_key: str | None = None,
    backfill_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> MagicMock:
    """Build a Dagster-``OutputContext``-like ``MagicMock``."""
    ctx = MagicMock()
    ctx.asset_key.to_user_string.return_value = asset_name
    ctx.run_id = run_id
    ctx.has_partition_key = has_partition_key
    if partition_keys is not None:
        ctx.asset_partition_keys = partition_keys
    if partition_key is not None:
        ctx.partition_key = partition_key
    ctx.run.backfill_id = backfill_id
    ctx.run.tags = tags if tags is not None else {}
    return ctx


class TestFromOutputContext:
    """Tests for ``WriteContext.from_output_context``."""

    def test_backfill_run_populates_signals(self) -> None:
        """A backfill OutputContext produces ``is_backfill=True`` and a
        non-None ``backfill_id``."""
        ctx = _build_output_context(
            backfill_id="bf_2026_05_22",
            tags={"dagster/backfill": "bf_2026_05_22"},
        )

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.is_backfill is True
        assert wctx.backfill_id == "bf_2026_05_22"

    def test_normal_run_leaves_signals_unset(self) -> None:
        """A non-backfill OutputContext produces ``is_backfill=False`` and
        ``backfill_id=None``."""
        ctx = _build_output_context()  # no tags, no backfill_id

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.is_backfill is False
        assert wctx.backfill_id is None

    def test_dagster_asset_deps_is_none_on_io_manager_path(self) -> None:
        """The IO-manager surface does not expose the asset graph in the
        same shape as ``AssetExecutionContext``; ``dagster_asset_deps``
        must be ``None`` here even if the underlying mock would otherwise
        return a value."""
        ctx = _build_output_context()
        ctx.asset_deps = {_FakeAssetKey("foo"): [_FakeAssetKey("bar")]}

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.dagster_asset_deps is None

    def test_pre_existing_fields_unchanged(self) -> None:
        """Phase 1 is additive; previously-existing fields should still
        be populated from the same OutputContext attributes."""
        ctx = _build_output_context(
            asset_name="gold/dim_member",
            run_id="run-42",
            has_partition_key=True,
            partition_keys=["2026-05-14"],
        )

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.asset_name == "gold/dim_member"
        assert wctx.run_id == "run-42"
        assert wctx.has_partition_key is True
        assert wctx.partition_keys == ["2026-05-14"]

    def test_asset_rematerialization_populates_backfill_id_from_tag(self) -> None:
        """Issue #334 Bug 1, end-to-end: asset-rematerialization backfills
        on the IO-manager path leave ``run.backfill_id`` as ``None`` and
        put the canonical id in the ``dagster/backfill`` tag value.
        ``WriteContext.from_output_context`` must surface the tag value
        as ``backfill_id`` so the downstream lineage INSERT records it."""
        ctx = _build_output_context(
            backfill_id=None,
            tags={"dagster/backfill": "bf_remat_io"},
        )

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.is_backfill is True
        assert wctx.backfill_id == "bf_remat_io"


# ---------------------------------------------------------------------------
# WriteContext.from_asset_context
# ---------------------------------------------------------------------------


def _build_asset_context(
    *,
    asset_name: str = "silver/dim_provider",
    run_id: str = "run-001",
    has_partition_key: bool = False,
    partition_keys: list[str] | None = None,
    partition_key: str | None = None,
    backfill_id: str | None = None,
    tags: dict[str, str] | None = None,
    asset_deps: dict[Any, list[Any]] | None = None,
) -> MagicMock:
    """Build a Dagster-``AssetExecutionContext``-like ``MagicMock``."""
    ctx = MagicMock()
    ctx.asset_key.to_user_string.return_value = asset_name
    ctx.run_id = run_id
    ctx.has_partition_key = has_partition_key
    if partition_keys is not None:
        ctx.partition_keys = partition_keys
    if partition_key is not None:
        ctx.partition_key = partition_key
    ctx.run.backfill_id = backfill_id
    ctx.run.tags = tags if tags is not None else {}
    # MagicMock would silently auto-create ``asset_deps``; only attach
    # it when the test explicitly opts in so we can also exercise the
    # "attribute absent" branch via ``del``.
    if asset_deps is None:
        del ctx.asset_deps
    else:
        ctx.asset_deps = asset_deps
    return ctx


class TestFromAssetContext:
    """Tests for ``WriteContext.from_asset_context``."""

    def test_backfill_run_populates_signals(self) -> None:
        """A backfill AssetExecutionContext produces both signals."""
        ctx = _build_asset_context(
            backfill_id="bf_xyz",
            tags={"dagster/backfill": "bf_xyz"},
        )

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.is_backfill is True
        assert wctx.backfill_id == "bf_xyz"

    def test_normal_run_leaves_signals_unset(self) -> None:
        """A non-backfill AssetExecutionContext produces ``False`` /
        ``None``."""
        ctx = _build_asset_context()

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.is_backfill is False
        assert wctx.backfill_id is None

    def test_missing_backfill_id_attribute_does_not_raise(self) -> None:
        """Older Dagster versions may not expose ``run.backfill_id``.
        The factory must degrade rather than raising.

        Issue #334 Bug 1: when the attribute is absent but the tag
        value carries the id (as with asset-rematerialization
        backfills), the tag value is surfaced as ``backfill_id``."""

        class _OldRun:
            tags: dict[str, str] = {"dagster/backfill": "bf_z"}

        class _OldContext:
            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.to_user_string.return_value = "asset"
                self.run_id = "r"
                self.log = MagicMock()
                self.has_partition_key = False
                self.run = _OldRun()

        ctx: Any = _OldContext()
        wctx = WriteContext.from_asset_context(ctx)

        # tag present -> is_backfill True; backfill_id attribute missing
        # but tag value is a non-empty str -> fallback surfaces the value.
        assert wctx.is_backfill is True
        assert wctx.backfill_id == "bf_z"

    def test_dagster_asset_deps_normalised(self) -> None:
        """``asset_deps`` is captured and normalised to
        ``dict[str, list[str]]``."""
        ctx = _build_asset_context(
            asset_deps={
                _FakeAssetKey("silver/dim_provider"): [
                    _FakeAssetKey("bronze/provider_raw"),
                ],
            },
        )

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.dagster_asset_deps == {
            "silver/dim_provider": ["bronze/provider_raw"],
        }

    def test_dagster_asset_deps_none_when_absent(self) -> None:
        """When the underlying context exposes no ``asset_deps``,
        ``WriteContext.dagster_asset_deps`` is ``None`` (not an empty dict)."""
        ctx = _build_asset_context()  # asset_deps absent via ``del`` above

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.dagster_asset_deps is None

    def test_pre_existing_fields_unchanged(self) -> None:
        """Phase 1 is additive; previously-existing fields are unchanged."""
        ctx = _build_asset_context(
            asset_name="gold/fact_claims",
            run_id="run-99",
            has_partition_key=True,
            partition_keys=["2026-05-13", "2026-05-14"],
        )

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.asset_name == "gold/fact_claims"
        assert wctx.run_id == "run-99"
        assert wctx.has_partition_key is True
        assert wctx.partition_keys == ["2026-05-13", "2026-05-14"]

    def test_asset_rematerialization_populates_backfill_id_from_tag(self) -> None:
        """Issue #334 Bug 1, end-to-end on the direct-resource path:
        ``from_asset_context`` must surface a tag-value backfill id
        when ``run.backfill_id`` is absent. Reference-silver asset-
        rematerialization runs that produced the empty ``backfill_id``
        column in production took this code path."""
        ctx = _build_asset_context(
            backfill_id=None,
            tags={"dagster/backfill": "bf_remat_asset"},
        )

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.is_backfill is True
        assert wctx.backfill_id == "bf_remat_asset"


# ---------------------------------------------------------------------------
# Defaults on direct construction
# ---------------------------------------------------------------------------


class TestWriteContextDefaults:
    """Tests for ``WriteContext`` default values.

    Pinned so callers that construct ``WriteContext`` directly (existing
    test helpers, ad-hoc tools) continue to work without setting the new
    Phase 1 fields explicitly.
    """

    def test_new_fields_have_safe_defaults(self) -> None:
        """``WriteContext`` constructed without the new fields should
        have ``backfill_id=None``, ``is_backfill=False``,
        ``dagster_asset_deps=None``."""
        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())
        assert wctx.backfill_id is None
        assert wctx.is_backfill is False
        assert wctx.dagster_asset_deps is None

    def test_phase2_dagster_handles_have_safe_defaults(self) -> None:
        """Migration 019 (#308) Phase 2: ``dagster_asset_key``,
        ``dagster_job_name``, ``code_location_name`` default to ``None``
        so callers that construct ``WriteContext`` directly continue to
        work without setting them."""
        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())
        assert wctx.dagster_asset_key is None
        assert wctx.dagster_job_name is None
        assert wctx.code_location_name is None


# ---------------------------------------------------------------------------
# Migration 019 (#308) Phase 2: Dagster join handle extraction
# ---------------------------------------------------------------------------


class TestExtractDagsterHandles:
    """Tests for ``_extract_dagster_handles``.

    Same defensive shape as ``_extract_backfill_signals`` /
    ``_normalize_asset_deps``: must (a) return useful values when the
    context provides them in real Dagster shapes, and (b) degrade to
    ``None`` rather than leaking a ``MagicMock`` auto-attribute downstream.
    """

    def test_full_context_returns_all_three_handles(self) -> None:
        """A context with ``asset_key.path`` (list of str), ``job_def.name``,
        and ``repository_def.name`` populates all three handles."""
        ctx = MagicMock()
        ctx.asset_key.path = ["fda_ndc_package_bronze"]
        ctx.job_def.name = "fda_ndc_job"
        ctx.repository_def.name = "ingest_code_location"

        asset_key_json, job_name, code_location = _extract_dagster_handles(ctx)

        assert asset_key_json == '["fda_ndc_package_bronze"]'
        assert job_name == "fda_ndc_job"
        assert code_location == "ingest_code_location"

    def test_multi_component_asset_key_path(self) -> None:
        """Multi-component asset keys (prefix + name) round-trip through
        the JSON-array encoder so a join to ``dagster.public.asset_keys``
        works for prefixed asset keys."""
        ctx = MagicMock()
        ctx.asset_key.path = ["bronze", "fda_ndc_package"]
        ctx.job_def.name = "j"
        ctx.repository_def.name = "loc"

        asset_key_json, _, _ = _extract_dagster_handles(ctx)
        assert asset_key_json == '["bronze", "fda_ndc_package"]'

    def test_falls_back_to_job_name_when_job_def_absent(self) -> None:
        """``OutputContext`` exposes ``job_name`` directly; the helper
        falls back to it when ``job_def`` is missing."""

        class _OutputCtx:
            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.path = ["asset_x"]
                self.job_name = "j_via_output_context"
                self.repository_def = MagicMock()
                self.repository_def.name = "loc"

        ctx: Any = _OutputCtx()
        _, job_name, _ = _extract_dagster_handles(ctx)
        assert job_name == "j_via_output_context"

    def test_missing_attributes_degrade_to_none(self) -> None:
        """All three handles are best-effort; absent attributes degrade
        to ``None`` so older Dagster versions don't break the write."""

        class _MinimalCtx:
            pass

        ctx: Any = _MinimalCtx()
        result = _extract_dagster_handles(ctx)
        assert result == (None, None, None)

    def test_bare_magicmock_returns_all_none(self) -> None:
        """Regression: a bare ``MagicMock`` produces child mocks for
        ``asset_key.path`` / ``job_def.name`` / ``repository_def.name``.
        The type-strict guards must reject those rather than producing
        MagicMock-stringified payloads that ``MetadataValue.text`` would
        reject downstream."""
        ctx = MagicMock()  # no explicit attribute pinning

        result = _extract_dagster_handles(ctx)
        assert result == (None, None, None)

    def test_non_string_path_components_rejected(self) -> None:
        """Defensive: if ``asset_key.path`` is a list containing non-str
        components, reject rather than ``json.dumps``-ing the mess."""
        ctx = MagicMock()
        ctx.asset_key.path = ["valid", 12345, "also_valid"]  # type: ignore[list-item]
        ctx.job_def.name = "j"
        ctx.repository_def.name = "loc"

        asset_key_json, _, _ = _extract_dagster_handles(ctx)
        assert asset_key_json is None

    def test_non_list_path_rejected(self) -> None:
        """A scalar ``asset_key.path`` (older Dagster shape, or test fixture
        misconfiguration) must not become ``json.dumps('scalar')``."""
        ctx = MagicMock()
        ctx.asset_key.path = "fda_ndc_package_bronze"  # not a list
        ctx.job_def.name = "j"
        ctx.repository_def.name = "loc"

        asset_key_json, _, _ = _extract_dagster_handles(ctx)
        assert asset_key_json is None

    def test_non_string_job_name_rejected(self) -> None:
        """A non-string ``job_def.name`` degrades to ``None`` (defensive
        against MagicMock leakage when the test does not pin it)."""
        ctx = MagicMock()
        ctx.asset_key.path = ["a"]
        ctx.job_def.name = 12345  # type: ignore[assignment]
        ctx.repository_def.name = "loc"

        _, job_name, _ = _extract_dagster_handles(ctx)
        assert job_name is None

    def test_repository_def_property_raising_degrades_to_none(self) -> None:
        """Real ``AssetExecutionContext.repository_def`` is a defined property
        that raises ``dagster_shared.check.CheckError`` ("No repository
        definition was set on the step context") on an ephemeral
        ``materialize()`` call without a ``Definitions`` wrapper. ``getattr``
        does not swallow that -- it only swallows ``AttributeError`` -- so
        the access must be wrapped in a try/except.

        Regression: data-platform integration_test_runner.py materializes
        assets directly via ``dagster.materialize(group_assets, ...)`` for
        speed. Before the fix this leaked the CheckError up through
        ``WriteContext.from_asset_context`` and failed every silver asset
        that calls ``database.write()`` in CI.
        """

        class _RaisingCtx:
            """Mirrors the real ``AssetExecutionContext.repository_def``
            shape: the attribute exists as a property, accessing it raises."""

            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.path = ["silver", "rxnorm_concepts"]
                self.job_def = MagicMock()
                self.job_def.name = "__ephemeral_asset_job__"

            @property
            def repository_def(self) -> Any:
                msg = "No repository definition was set on the step context"
                raise RuntimeError(msg)

        ctx: Any = _RaisingCtx()
        asset_key_json, job_name, code_location = _extract_dagster_handles(ctx)

        assert asset_key_json == '["silver", "rxnorm_concepts"]'
        assert job_name == "__ephemeral_asset_job__"
        assert code_location is None

    def test_asset_key_property_raising_degrades_to_none(self) -> None:
        """Issue #341 audit: ``asset_key`` is a ``@property`` on the real
        Dagster contexts whose getter raises -- ``DagsterInvariantViolationError``
        inside a ``multi_asset`` (>1 output) and ``DagsterInvalidPropertyError``
        on a non-asset op (same root cause as #339, but reached via the
        write path's ``_extract_dagster_handles`` rather than the reconcile
        path). ``getattr(..., None)`` only swallows ``AttributeError``, so
        the read must be wrapped; the asset-key handle degrades to ``None``
        while ``job_name`` still resolves.
        """

        class _DagsterInvariantViolationError(Exception):
            """Stand-in for the Dagster error; the helper catches bare
            ``Exception`` so the exact class is irrelevant."""

        class _CtxWithRaisingAssetKey:
            def __init__(self) -> None:
                self.job_def = MagicMock()
                self.job_def.name = "multi_asset_job"
                self.repository_def = MagicMock()
                self.repository_def.name = "loc"

            @property
            def asset_key(self) -> object:
                raise _DagsterInvariantViolationError(
                    "Cannot call `context.asset_key` in a multi_asset with more "
                    "than one asset. Use `context.asset_key_for_output` instead."
                )

        ctx: Any = _CtxWithRaisingAssetKey()
        asset_key_json, job_name, code_location = _extract_dagster_handles(ctx)

        assert asset_key_json is None
        assert job_name == "multi_asset_job"
        assert code_location == "loc"

    def test_job_name_property_raising_degrades_to_none(self) -> None:
        """Issue #341 audit: ``OutputContext.job_name`` is a ``@property``
        whose getter raises ``DagsterInvariantViolationError`` when the
        job name was not provided at construction (and ``job_def`` is absent
        on ``OutputContext``, so the fallback is the only path). The read
        must be guarded so the job-name handle degrades to ``None`` while
        the asset-key handle still resolves.
        """

        class _DagsterInvariantViolationError(Exception):
            pass

        class _OutputCtxWithRaisingJobName:
            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.path = ["bronze", "fda_ndc_package"]
                # job_def absent -> helper falls back to job_name property

            @property
            def job_name(self) -> object:
                raise _DagsterInvariantViolationError(
                    "Attempting to access pipeline_name, but it was not provided "
                    "when constructing the OutputContext"
                )

        ctx: Any = _OutputCtxWithRaisingJobName()
        asset_key_json, job_name, code_location = _extract_dagster_handles(ctx)

        assert asset_key_json == '["bronze", "fda_ndc_package"]'
        assert job_name is None
        assert code_location is None

    def test_job_def_property_raising_falls_back_to_job_name(self) -> None:
        """Issue #341 audit: ``job_def`` is a ``@property`` on op/asset
        contexts that delegates to ``self._step_execution_context`` and can
        raise on ephemeral runs. The guarded read must degrade past the
        raising ``job_def`` and still pick up a plain ``job_name`` fallback.
        """

        class _CheckError(Exception):
            pass

        class _CtxWithRaisingJobDef:
            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.path = ["asset_x"]
                self.job_name = "j_via_fallback"

            @property
            def job_def(self) -> object:
                raise _CheckError("No step execution context available")

        ctx: Any = _CtxWithRaisingJobDef()
        asset_key_json, job_name, _ = _extract_dagster_handles(ctx)

        assert asset_key_json == '["asset_x"]'
        assert job_name == "j_via_fallback"

    def test_job_def_and_job_name_both_raising_degrade_to_none(self) -> None:
        """Issue #341 audit: combinatorial case where *both* the primary
        ``job_def`` read and the ``job_name`` fallback raise (e.g. an
        ephemeral context whose ``_step_execution_context`` is unavailable
        for both). Both ``except`` branches assign ``None``, so the
        job-name handle degrades to ``None`` while the asset-key handle
        still resolves -- no exception escapes the helper.
        """

        class _CheckError(Exception):
            pass

        class _CtxWithBothJobPropsRaising:
            def __init__(self) -> None:
                self.asset_key = MagicMock()
                self.asset_key.path = ["asset_y"]

            @property
            def job_def(self) -> object:
                raise _CheckError("No step execution context available")

            @property
            def job_name(self) -> object:
                raise _CheckError("No step execution context available")

        ctx: Any = _CtxWithBothJobPropsRaising()
        asset_key_json, job_name, code_location = _extract_dagster_handles(ctx)

        assert asset_key_json == '["asset_y"]'
        assert job_name is None
        assert code_location is None


# ---------------------------------------------------------------------------
# Migration 019 (#308) Phase 2: WriteContext factories populate handles
# ---------------------------------------------------------------------------


class TestPhase2FactoryWiring:
    """``WriteContext.from_output_context`` and ``from_asset_context``
    populate the Phase 2 handles from the context attributes."""

    def test_from_output_context_populates_handles(self) -> None:
        """A populated ``OutputContext`` produces populated handles on
        the resulting ``WriteContext``."""
        ctx = _build_output_context(asset_name="silver/dim_provider")
        ctx.asset_key.path = ["silver", "dim_provider"]
        ctx.job_def.name = "silver_job"
        ctx.repository_def.name = "silver_loc"

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.dagster_asset_key == '["silver", "dim_provider"]'
        assert wctx.dagster_job_name == "silver_job"
        assert wctx.code_location_name == "silver_loc"

    def test_from_asset_context_populates_handles(self) -> None:
        """A populated ``AssetExecutionContext`` produces populated handles."""
        ctx = _build_asset_context(asset_name="gold/fact_claims")
        ctx.asset_key.path = ["gold", "fact_claims"]
        ctx.job_def.name = "gold_job"
        ctx.repository_def.name = "gold_loc"

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.dagster_asset_key == '["gold", "fact_claims"]'
        assert wctx.dagster_job_name == "gold_job"
        assert wctx.code_location_name == "gold_loc"

    def test_from_output_context_bare_mock_defaults_handles_to_none(self) -> None:
        """A minimally-configured ``MagicMock`` OutputContext (with only
        the previously-existing attributes pinned) leaves the new Phase 2
        handles ``None`` because the type-strict extractor rejects bare
        MagicMock children."""
        ctx = _build_output_context()  # nothing pinned for asset_key.path / job_def / repo_def

        wctx = WriteContext.from_output_context(ctx)

        assert wctx.dagster_asset_key is None
        assert wctx.dagster_job_name is None
        assert wctx.code_location_name is None

    def test_from_asset_context_bare_mock_defaults_handles_to_none(self) -> None:
        """Same defensive contract on the asset-context path."""
        ctx = _build_asset_context()

        wctx = WriteContext.from_asset_context(ctx)

        assert wctx.dagster_asset_key is None
        assert wctx.dagster_job_name is None
        assert wctx.code_location_name is None


# ---------------------------------------------------------------------------
# Migration 018 Phase 4: resolve_partition_dates
# ---------------------------------------------------------------------------


class TestResolvePartitionDates:
    """``WriteContext.resolve_partition_dates`` derives ``data_date`` /
    ``data_date_range`` from the partition keys so the lineage row's
    partition columns populate (and the partition-scoped
    ``replaces_lineage_id`` lookup actually keys on something)."""

    def test_non_partitioned_returns_none_pair(self) -> None:
        """Without ``partition_column``, both outputs are ``None``."""
        wctx = WriteContext(asset_name="a", run_id="r", log=MagicMock())
        assert wctx.resolve_partition_dates({"partition_column": None}) == (None, None)

    def test_partition_column_set_but_no_keys_returns_none_pair(self) -> None:
        """``partition_column`` set but ``partition_keys`` empty: ``(None, None)``."""
        wctx = WriteContext(
            asset_name="a",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=[],
        )
        result = wctx.resolve_partition_dates({"partition_column": "date_col"})
        assert result == (None, None)

    def test_single_iso_date_returns_data_date(self) -> None:
        """One ISO-format partition key → ``(parsed_date, None)``."""
        from datetime import date

        wctx = WriteContext(
            asset_name="a",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2026-05-15"],
        )
        result = wctx.resolve_partition_dates({"partition_column": "date_col"})
        assert result == (date(2026, 5, 15), None)

    def test_multiple_iso_dates_return_data_date_range(self) -> None:
        """Multiple partition keys → ``(None, (min, max))``."""
        from datetime import date

        wctx = WriteContext(
            asset_name="a",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            # Out of order to verify min/max is correct, not "first/last".
            partition_keys=["2026-05-15", "2026-05-13", "2026-05-17"],
        )
        result = wctx.resolve_partition_dates({"partition_column": "date_col"})
        assert result == (None, (date(2026, 5, 13), date(2026, 5, 17)))

    def test_non_iso_partition_key_bails_to_none(self) -> None:
        """A non-ISO partition shape (e.g. ``2026-05-15-00``) bails to
        ``(None, None)`` and logs at DEBUG -- the helper does not guess
        a date format."""
        log = MagicMock()
        wctx = WriteContext(
            asset_name="a",
            run_id="r",
            log=log,
            has_partition_key=True,
            partition_keys=["2026-05-15-00"],
        )
        result = wctx.resolve_partition_dates({"partition_column": "date_col"})
        assert result == (None, None)
        log.debug.assert_called()

    def test_partition_column_missing_from_config(self) -> None:
        """``write_config`` without a ``partition_column`` key (just like
        a non-partitioned write) returns ``(None, None)``."""
        wctx = WriteContext(
            asset_name="a",
            run_id="r",
            log=MagicMock(),
            has_partition_key=True,
            partition_keys=["2026-05-15"],
        )
        # Dict literally lacks the key.
        result = wctx.resolve_partition_dates({})
        assert result == (None, None)


# ---------------------------------------------------------------------------
# Migration 018 Phase 4: WriteResult.replaces_lineage_id metadata surface
# ---------------------------------------------------------------------------


class TestReplacesLineageIdMetadata:
    """``WriteResult.replaces_lineage_id`` surfaces on
    ``to_dagster_metadata()`` when non-``None`` so the materialization
    event view exposes the chain link."""

    def test_omitted_when_none(self) -> None:
        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={},
            row_count=0,
        )
        assert "replaces_lineage_id" not in result.to_dagster_metadata()

    def test_emitted_when_present(self) -> None:
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.FULL_REFRESH,
            stats={},
            row_count=0,
            replaces_lineage_id="00000000-0000-0000-0000-000000000099",
        )
        metadata = result.to_dagster_metadata()
        assert metadata["replaces_lineage_id"] == MetadataValue.text(
            "00000000-0000-0000-0000-000000000099"
        )


class TestParentLineageCountMetadata:
    """Migration 018 Phase 5: ``parent_lineage_count`` always surfaces
    on ``to_dagster_metadata()`` so the materialization-event view shows
    the cardinality of the upstream lineage set (the full UUID list lives
    on ``data_lineage.parent_lineage_ids`` only)."""

    def test_zero_count_still_emitted(self) -> None:
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
        )
        # ``0`` is informative -- distinguishes "no upstream lineage"
        # from "lineage disabled" (which would never construct a
        # ``WriteResult`` in the first place).
        assert result.to_dagster_metadata()["parent_lineage_count"] == MetadataValue.int(0)

    def test_nonzero_count_emitted(self) -> None:
        from dagster import MetadataValue

        from moncpipelib.io_managers.enums import WriteMode
        from moncpipelib.resources.types import WriteResult

        result = WriteResult(
            table_name="silver.x",
            schema="silver",
            layer="silver",
            write_mode=WriteMode.APPEND,
            stats={},
            row_count=0,
            parent_lineage_count=3,
        )
        assert result.to_dagster_metadata()["parent_lineage_count"] == MetadataValue.int(3)
