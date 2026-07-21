"""Shared types for PostgresResource write operations.

Defines the context protocol, write context bridge, and write result dataclass
used by both ``PostgresResource.write()`` (direct usage) and
``PostgresIOManager.handle_output()`` (delegation path).

Testing the context-shape helpers (``_extract_backfill_signals``,
``_extract_dagster_handles``, ``_normalize_asset_deps``): several of the
Dagster context attributes they read (``run``, ``asset_key``, ``job_def``,
``job_name``, ``repository_def``, ``asset_deps``) are ``@property``
descriptors whose getters raise something *other* than ``AttributeError``
(e.g. ``DagsterInvalidPropertyError`` / ``DagsterInvariantViolationError``)
on non-asset ops, multi-assets, or ephemeral runs. ``MagicMock`` fixtures
hide this failure mode -- auto-attribute access on a mock returns a child
mock and never raises -- so a green test against a ``MagicMock`` context
proves nothing about the descriptor-raises path. Always include a
hand-rolled fixture whose property *raises* (see
``TestExtractDagsterHandles.test_*_property_raising_degrades_to_none``)
for any helper that reads a Dagster ``@property``. See the
``test_*_property_raising_*`` tests across ``TestExtractBackfillSignals``,
``TestNormalizeAssetDeps``, and ``TestExtractDagsterHandles``. See #339 / #341.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dagster import AssetExecutionContext, MetadataValue, OutputContext

    from moncpipelib.contracts.models import ContractValidationSummary, DataContract
    from moncpipelib.io_managers.enums import WriteMode

COLUMN_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "int",
    "decimal": "float",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "uuid": "uuid",
    "json": "json",
    "jsonb": "jsonb",
}


class _Sentinel:
    """Sentinel value for contract auto-discovery default.

    Used to distinguish between ``contract=None`` (explicitly skip contract
    loading) and ``contract`` not passed (auto-discover).
    """


SENTINEL = _Sentinel()
"""Module-level sentinel instance for ``write(contract=...)`` default."""

_DAGSTER_BACKFILL_TAG = "dagster/backfill"


def _extract_backfill_signals(context: Any) -> tuple[str | None, bool]:
    """Return ``(backfill_id, is_backfill)`` for a Dagster context.

    Both ``OutputContext`` and ``AssetExecutionContext`` expose ``.run``,
    so the extraction is symmetric. ``is_backfill`` reflects *presence* of
    the ``dagster/backfill`` run tag (not its boolean truth) — Dagster
    sets the tag on every backfill-participant run.

    ``backfill_id`` is read from ``run.backfill_id`` when present.
    Asset-rematerialization backfills are the exception: Dagster leaves
    ``run.backfill_id`` as ``None`` on those and puts the canonical id
    in the ``dagster/backfill`` tag *value*. When the attribute is
    absent but the tag value is a non-empty string we surface that as
    the id. Empty-string tag values are rejected — Dagster uses an
    empty value as a "this run is part of a backfill but the id is
    elsewhere" marker, not as the id itself.

    Type-strict on the extracted values: only ``str`` is accepted as a
    ``backfill_id`` (anything else, including a ``MagicMock`` produced
    by a test harness that does not explicitly pin ``.run.backfill_id``,
    degrades to ``None``); only ``Mapping`` is accepted as ``tags``.
    Without this guard, ``MagicMock``'s auto-attribute behaviour returns
    child mocks for any access, which would surface as non-string
    metadata payloads downstream (rejected by ``MetadataValue.text``).

    Resilient to API drift: missing ``run``, ``backfill_id``, or ``tags``
    attributes degrade silently to ``(None, False)``.

    Descriptor-raises guard (#341): ``OpExecutionContext.run`` /
    ``AssetExecutionContext.run`` are ``@property`` descriptors that
    delegate to ``self._step_execution_context.dagster_run``; on an
    ephemeral / partially-constructed context that delegation can raise
    something other than ``AttributeError`` (mirroring ``repository_def``
    in ``_extract_dagster_handles``). ``getattr(..., None)`` only swallows
    ``AttributeError``, so the read is wrapped in ``try`` / ``except``.
    ``OutputContext`` does not expose ``run`` at all -- that path stays a
    clean ``AttributeError`` and degrades to ``(None, False)``.
    ``run.backfill_id`` and ``run.tags`` are plain attributes on
    ``DagsterRun`` (not descriptors), so they need no further guard.
    """
    try:
        run = getattr(context, "run", None)
    except Exception:  # noqa: BLE001 -- run is a property; getter can raise on ephemeral contexts
        run = None
    if run is None:
        return None, False

    raw_backfill_id = getattr(run, "backfill_id", None)
    backfill_id = raw_backfill_id if isinstance(raw_backfill_id, str) else None

    raw_tags = getattr(run, "tags", None)
    is_backfill = isinstance(raw_tags, Mapping) and _DAGSTER_BACKFILL_TAG in raw_tags

    if backfill_id is None and is_backfill:
        tag_value = raw_tags.get(_DAGSTER_BACKFILL_TAG)  # type: ignore[union-attr]
        if isinstance(tag_value, str) and tag_value:
            backfill_id = tag_value

    return backfill_id, is_backfill


def _extract_dagster_handles(context: Any) -> tuple[str | None, str | None, str | None]:
    """Return ``(dagster_asset_key, dagster_job_name, code_location_name)``.

    Migration 019 (#308) Phase 2: ``pipeline_registry`` carries these as
    join handles so downstream queries can correlate moncpipelib's lineage
    with Dagster's own ``asset_keys`` / runs tables without JSON-array
    gymnastics.

    - ``dagster_asset_key`` is encoded in Dagster's native JSON-array form
      (``["fda_ndc_package_bronze"]``) -- not the slash-joined user
      string -- because that is the form ``dagster.public.asset_keys``
      stores. A direct text comparison joins without ``regexp_replace``.
    - ``dagster_job_name`` is sourced from ``context.job_def.name`` on
      ``AssetExecutionContext`` or ``context.job_name`` on
      ``OutputContext``. Either path degrades to ``None`` silently.
    - ``code_location_name`` is sourced from ``context.repository_def.name``
      when reachable. Not always available on the IO-manager path; the
      Phase 2 plan documents this asymmetry.

    Type-strict (mirroring ``_extract_backfill_signals``): a ``MagicMock``-
    style auto-generated attribute returns a child mock, which we reject
    so non-string payloads never reach the database.

    Descriptor-raises guard (#341): ``asset_key``, ``job_def`` and
    ``job_name`` are all ``@property`` descriptors on the real Dagster
    contexts, not plain attributes -- and their getters raise something
    other than ``AttributeError`` in ordinary situations:

    - ``OpExecutionContext.asset_key`` raises ``DagsterInvariantViolationError``
      inside a ``multi_asset`` (>1 output) and ``DagsterInvalidPropertyError``
      on a non-asset op (same root cause as #339 / #340).
    - ``OutputContext.asset_key`` / ``OutputContext.job_name`` raise
      ``DagsterInvariantViolationError`` when the underlying value was not
      provided when the context was constructed.
    - ``job_def`` / ``job_name`` on op/asset contexts delegate to
      ``self._step_execution_context``, which can raise on ephemeral runs.

    ``getattr(..., None)`` only substitutes the default on ``AttributeError``,
    so each property read is wrapped in ``try`` / ``except`` -- mirroring the
    long-standing guard on ``repository_def`` below. Each handle is
    best-effort; a raising getter degrades that handle to ``None``.
    """
    import json as _json

    asset_key_json: str | None = None
    try:
        raw_asset_key = getattr(context, "asset_key", None)
    except Exception:  # noqa: BLE001 -- asset_key is a property; getter raises on multi_asset / non-asset op
        raw_asset_key = None
    if raw_asset_key is not None:
        path = getattr(raw_asset_key, "path", None)
        if isinstance(path, (list, tuple)) and all(isinstance(p, str) for p in path):
            asset_key_json = _json.dumps(list(path))

    job_name: str | None = None
    try:
        job_def = getattr(context, "job_def", None)
    except Exception:  # noqa: BLE001 -- job_def is a property; getter can raise on ephemeral contexts
        job_def = None
    if job_def is not None:
        raw_name = getattr(job_def, "name", None)
        if isinstance(raw_name, str):
            job_name = raw_name
    if job_name is None:
        try:
            raw_job_name = getattr(context, "job_name", None)
        except Exception:  # noqa: BLE001 -- job_name is a property; OutputContext getter raises when unset
            raw_job_name = None
        if isinstance(raw_job_name, str):
            job_name = raw_job_name

    code_location: str | None = None
    # ``AssetExecutionContext.repository_def`` is a defined property, not a
    # plain attribute. On an ephemeral ``dagster.materialize(...)`` call that
    # isn't wrapped in a ``Definitions`` object the property raises
    # ``dagster_shared.check.CheckError`` ("No repository definition was set
    # on the step context"). ``getattr(..., None)`` only swallows
    # ``AttributeError``, so a bare access propagates the CheckError up
    # through ``database.write()`` and fails the asset. The integration test
    # runner (data-platform/tools/integration_test_runner.py) calls
    # ``materialize()`` without a Definitions wrapper for speed, so degrade
    # to ``None`` instead of crashing.
    try:
        repo_def = context.repository_def
    except Exception:  # noqa: BLE001 -- Dagster raises CheckError on ephemeral runs without a repo
        repo_def = None
    if repo_def is not None:
        raw_loc = getattr(repo_def, "name", None)
        if isinstance(raw_loc, str):
            code_location = raw_loc

    return asset_key_json, job_name, code_location


def _normalize_asset_deps(context: Any) -> dict[str, list[str]] | None:
    """Normalise ``context.asset_deps`` to a flat-string mapping.

    Dagster exposes ``asset_deps`` on ``AssetExecutionContext`` as
    ``dict[AssetKey, list[AssetKey]]``. ``data_lineage.asset_name`` stores
    the flat ``AssetKey.to_user_string()`` form, so both keys and values
    are converted with ``.to_user_string()`` for direct join compatibility.

    Type-strict: only ``Mapping`` inputs are normalised. A ``MagicMock``
    auto-generated ``asset_deps`` (when a test harness does not pin it
    to a real dict or ``None``) is rejected and returns ``None``, the
    same as a missing or ``None`` attribute.

    Descriptor-raises guard (#341): in the pinned Dagster (1.13.x)
    ``asset_deps`` is not exposed on ``AssetExecutionContext`` at all, so
    ``getattr(..., None)`` returns ``None`` via a clean ``AttributeError``.
    Other Dagster versions expose it as a ``@property`` whose getter can
    raise (e.g. on an ``OpExecutionContext`` with no assets definition),
    and ``getattr(..., None)`` only swallows ``AttributeError``. The read
    is therefore wrapped in ``try`` / ``except`` for forward-compatibility,
    mirroring the guards on ``run`` / ``asset_key`` in the sibling helpers.
    """
    try:
        raw = getattr(context, "asset_deps", None)
    except Exception:  # noqa: BLE001 -- asset_deps may be a property; getter can raise on op contexts
        return None
    if raw is None or not isinstance(raw, Mapping):
        return None
    try:
        normalised: dict[str, list[str]] = {}
        for key, deps in raw.items():
            asset_name = key.to_user_string() if hasattr(key, "to_user_string") else str(key)
            dep_names: list[str] = []
            for dep in deps:
                dep_names.append(
                    dep.to_user_string() if hasattr(dep, "to_user_string") else str(dep)
                )
            normalised[asset_name] = dep_names
        return normalised
    except (AttributeError, TypeError):
        return None


@runtime_checkable
class LoggingContext(Protocol):
    """Minimal context interface for writer functions and reconciliation.

    Both ``dagster.OutputContext`` and ``dagster.AssetExecutionContext``
    satisfy this protocol structurally (duck typing). Writer functions,
    the contract reconciler, and other shared modules should accept this
    type instead of the concrete Dagster context types.
    """

    @property
    def log(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class WriteContext:
    """Execution context for a ``write()`` call.

    Bridges the differences between ``dagster.OutputContext`` (from
    ``PostgresIOManager.handle_output``) and ``dagster.AssetExecutionContext``
    (from direct ``PostgresResource.write()`` usage). Both callers construct
    a ``WriteContext`` from their native Dagster context type.
    """

    asset_name: str
    """Dagster asset name (e.g., ``dim_provider_gold``)."""

    run_id: str
    """Dagster run ID."""

    log: Any
    """Logger with ``.info()``, ``.warning()``, ``.error()``, ``.debug()`` methods."""

    has_partition_key: bool = False
    """Whether the current execution has a Dagster partition context."""

    partition_keys: list[str] | None = None
    """Active partition key values, or ``None`` if not partitioned."""

    backfill_id: str | None = None
    """Stable identifier of the backfill batch this run belongs to, or ``None``.

    Sourced from ``context.run.backfill_id`` when available. ``None`` for
    non-backfill runs and for older Dagster versions that do not expose the
    attribute.
    """

    is_backfill: bool = False
    """Whether this run is part of a Dagster backfill.

    Sourced from the *presence* of the ``dagster/backfill`` run tag — tag
    presence (not boolean truth) is the signal Dagster sets when a run is
    part of a backfill batch.
    """

    dagster_asset_deps: dict[str, list[str]] | None = None
    """Asset graph captured from Dagster, or ``None``.

    Populated by ``from_asset_context`` from ``context.asset_deps`` and
    normalised into the flat ``AssetKey.to_user_string()`` form that
    ``data_lineage.asset_name`` uses. Always ``None`` on the IO-manager
    path (``from_output_context``) because the IO-manager surface does not
    expose the asset graph in the same shape; cross-asset Dagster-graph
    parent resolution (Phase 6 of migration 018) is therefore only
    available on the direct-resource path.
    """

    dagster_asset_key: str | None = None
    """``AssetKey`` in Dagster's JSON-array form, e.g. ``["asset_name"]``.

    Migration 019 (#308) Phase 2: stored on ``pipeline_registry`` so a
    direct text-equality join to ``dagster.public.asset_keys`` works
    without JSON parsing. The flat slash-joined ``asset_name`` is the
    moncpipelib-internal identity; ``dagster_asset_key`` is the Dagster-
    native identity.
    """

    dagster_job_name: str | None = None
    """Dagster job name the run is part of, or ``None``.

    Migration 019 (#308) Phase 2: cached on ``pipeline_registry`` for
    operational queries ("which job materialized this asset"). Sourced
    from ``context.job_def.name`` (asset-context) or ``context.job_name``
    (output-context).
    """

    code_location_name: str | None = None
    """Dagster code-location name, or ``None``.

    Migration 019 (#308) Phase 2: cached on ``pipeline_registry`` for
    multi-code-location deployments. May be ``None`` on the IO-manager
    path -- ``OutputContext`` does not always expose ``repository_def``
    in the same shape as ``AssetExecutionContext``.
    """

    @classmethod
    def from_output_context(cls, context: OutputContext) -> WriteContext:
        """Construct from a Dagster ``OutputContext`` (IO manager path).

        ``asset_name`` is load-bearing and read *unguarded* via
        ``context.asset_key.to_user_string()`` -- deliberately, unlike the
        best-effort ``asset_key_json`` *metadata* handle that
        ``_extract_dagster_handles`` degrades to ``None`` (#341). The
        IO-manager path is only entered after Dagster has resolved the
        output's asset key, so a raise here is a genuine invariant
        violation and failing fast is correct.
        """
        partition_keys: list[str] | None = None
        if context.has_partition_key:
            try:
                partition_keys = list(context.asset_partition_keys)
            except Exception:
                partition_keys = [context.partition_key]

        backfill_id, is_backfill = _extract_backfill_signals(context)
        asset_key_json, job_name, code_location = _extract_dagster_handles(context)

        return cls(
            asset_name=context.asset_key.to_user_string(),
            run_id=context.run_id,
            log=context.log,
            has_partition_key=context.has_partition_key,
            partition_keys=partition_keys,
            backfill_id=backfill_id,
            is_backfill=is_backfill,
            dagster_asset_deps=None,
            dagster_asset_key=asset_key_json,
            dagster_job_name=job_name,
            code_location_name=code_location,
        )

    @classmethod
    def from_asset_context(cls, context: AssetExecutionContext) -> WriteContext:
        """Construct from a Dagster ``AssetExecutionContext`` (direct resource path).

        As in ``from_output_context``, ``asset_name`` is read *unguarded*
        via ``context.asset_key.to_user_string()``. This encodes a
        deliberate contract: the direct ``write()`` path requires a single
        resolvable asset key. A ``multi_asset`` (where ``asset_key`` raises
        ``DagsterInvariantViolationError``) must use
        ``asset_key_for_output`` upstream and pass a ``WriteContext`` /
        explicit ``target`` rather than calling ``write(context=...)`` with
        the raw multi-asset context -- the factory will fail fast here, before
        the guarded best-effort read in ``_extract_dagster_handles``. The
        guard there protects only the optional ``asset_key_json`` metadata
        handle, not this required identity.
        """
        partition_keys: list[str] | None = None
        if context.has_partition_key:
            try:
                partition_keys = list(context.partition_keys)
            except Exception:
                partition_keys = [context.partition_key]

        backfill_id, is_backfill = _extract_backfill_signals(context)
        dagster_asset_deps = _normalize_asset_deps(context)
        asset_key_json, job_name, code_location = _extract_dagster_handles(context)

        return cls(
            asset_name=context.asset_key.to_user_string(),
            run_id=context.run_id,
            log=context.log,
            has_partition_key=context.has_partition_key,
            partition_keys=partition_keys,
            backfill_id=backfill_id,
            is_backfill=is_backfill,
            dagster_asset_deps=dagster_asset_deps,
            dagster_asset_key=asset_key_json,
            dagster_job_name=job_name,
            code_location_name=code_location,
        )

    def resolve_partition_dates(
        self,
        write_config: dict[str, Any],
    ) -> tuple[date | None, tuple[date, date] | None]:
        """Derive ``(data_date, data_date_range)`` for the lineage row.

        Migration 018 Phase 4: the partition-scoped ``replaces_lineage_id``
        lookup keys on ``data_date`` (single-date partition) or
        ``data_date_range`` (multi-date partition), so both columns must
        be populated on the in-flight lineage row. Without this they
        stay NULL forever and the lookup degenerates to NULL.

        Returns:
            - ``(date, None)`` for a single-date partition write (one
              ``partition_keys`` entry parseable as an ISO date).
            - ``(None, (start, end))`` for a multi-date range write
              (``partition_keys`` entries parsed as ISO dates; range
              ends are ``min`` / ``max`` of the parsed list).
            - ``(None, None)`` when not partitioned, when
              ``partition_column`` is absent from ``write_config``, or
              when any partition key fails to parse as an ISO date
              (with a ``DEBUG`` log line).
        """
        partition_column = write_config.get("partition_column")
        if partition_column is None or not self.partition_keys:
            return None, None

        parsed: list[date] = []
        for key in self.partition_keys:
            try:
                parsed.append(date.fromisoformat(key))
            except (TypeError, ValueError):
                # Non-ISO partition shapes (e.g., multi-dimensional keys
                # or hour-grain stamps) are not yet supported by this
                # helper. Bail to NULL rather than guessing.
                self.log.debug(
                    "resolve_partition_dates: partition key %r is not an ISO date; "
                    "leaving data_date / data_date_range NULL",
                    key,
                )
                return None, None

        if len(parsed) == 1:
            return parsed[0], None
        return None, (min(parsed), max(parsed))


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Result of a ``PostgresResource.write()`` call.

    Contains all statistics and metadata produced during the write operation.
    The IO manager converts this to Dagster ``MetadataValue`` dicts via
    ``to_dagster_metadata()``. Direct resource callers can inspect it
    programmatically.
    """

    table_name: str
    """Fully-qualified target table (``schema.table``)."""

    schema: str
    """Resolved PostgreSQL schema."""

    layer: str | None
    """Derived data layer (``bronze``/``silver``/``gold``) or ``None``."""

    write_mode: WriteMode
    """Write mode that was actually used (after contract reconciliation)."""

    stats: dict[str, Any]
    """Mode-specific statistics from the writer functions.

    Keys vary by write mode:
    - ``full_refresh``: rows_deleted, rows_inserted, clear_method, insert_method
    - ``upsert``: rows_upserted
    - ``append``: rows_inserted, insert_method
    - ``scd2``: rows_new, rows_expired, rows_inserted, rows_unchanged, rows_deleted
    - ``partition_scoped``: rows_deleted, rows_inserted, insert_method
    """

    row_count: int
    """Total rows written across all batches."""

    batch_count: int = 1
    """Number of batches processed (1 for single DataFrame)."""

    contract_summary: ContractValidationSummary | None = None
    """Contract validation results, or ``None`` if no contract validated."""

    contract: DataContract | None = None
    """Loaded contract, or ``None``."""

    lineage_id: str | None = None
    """UUID7 lineage record ID, or ``None`` if lineage disabled."""

    lineage_key: str | None = None
    """Human-readable lineage key, or ``None``."""

    columns: list[str] = field(default_factory=list)
    """Column names from the written DataFrame."""

    source_file: str | None = None
    """Source file path, if provided."""

    primary_key: list[str] | None = None
    """Primary key columns used for upsert, or ``None``."""

    partition_column: str | None = None
    """Partition column used for scoped writes, or ``None``."""

    business_key: list[str] | None = None
    """Business key columns used for SCD2, or ``None``."""

    is_backfill: bool = False
    """Whether the write ran as part of a Dagster backfill.

    Mirrors ``WriteContext.is_backfill`` so the materialization-event view
    surfaces backfill participation without a database query.
    """

    backfill_id: str | None = None
    """Stable identifier of the backfill batch this write belongs to, or
    ``None`` for non-backfill runs.

    Mirrors ``WriteContext.backfill_id``.
    """

    replaces_lineage_id: str | None = None
    """UUID of the immediately prior ``data_lineage`` row this write
    replaces, or ``None``.

    Populated by migration 018 Phase 4 for ``FULL_REFRESH`` writes (whole
    table or partition-scoped). ``UPSERT`` / ``APPEND`` / ``SCD2`` always
    leave this ``None`` because they accumulate rather than replace.

    Concurrency caveat (READ COMMITTED): two concurrent ``FULL_REFRESH``
    runs of the same asset may produce *either* a sibling pair (both
    link to the same predecessor) *or* a chain (whichever commits second
    links to the first), depending on commit timing. Consumers must not
    assume one shape. See ``LineageTracker.find_prior_lineage_id`` for
    the underlying ``MAX(processed_at)`` lookup semantics.
    """

    parent_lineage_count: int = 0
    """Number of unique upstream ``_lineage_id`` values recorded as
    ``parent_lineage_ids`` on the ``data_lineage`` row.

    Populated by migration 018 Phase 5. The full UUID list lives on the
    ``data_lineage`` row only; only the count surfaces on Dagster
    metadata to keep the materialization-event payload bounded.
    """

    duration_seconds: float | None = None
    """Wall-clock duration of the ``write()`` call, in seconds.

    Measured by ``time.perf_counter`` around the top-level ``write()``
    dispatch so it captures contract loading, the write itself, lineage
    persistence, and period-registry stamping — the full operator-visible
    cost of the write. ``None`` only on the rare in-test path that
    constructs a ``WriteResult`` directly without going through
    ``write()``; the production write path always sets it.
    """

    partition_keys: list[str] | None = None
    """Active partition key values for the write, or ``None``.

    Mirrors ``WriteContext.partition_keys`` so the materialization-event
    view surfaces *which partition* was written without consulting the
    Dagster run partition separately. Multi-partition writes (a single
    ``write()`` call that batches across partitions) emit the full list.
    """

    source_uri: str | None = None
    """Resolved source URI for the write, or ``None``.

    Mirrors the ``source_uri`` argument to ``write()``. Load-bearing for
    from_ingest bronze writes (the resolved blob path the data was
    loaded from); ``None`` for non-bronze writes and for enumerated
    bronze periods where the URI is taken from the matched period and
    not the kwarg.
    """

    pipeline_id: str | None = None
    """Stable pipeline UUID from the contract, or ``None``.

    Mirrors the ``pipeline_id`` resolved during ``write()`` (either the
    explicit kwarg or the contract's ``pipeline_id``). Surfaces in
    materialization metadata so consumers can join to
    ``pipeline_registry`` without a lineage hop.
    """

    effective_date: date | None = None
    """Override ``effective_date`` used for SCD2 stamps, or ``None``.

    Mirrors the ``effective_date`` kwarg to ``write()``. Audit-critical
    because it changes what ``effective_from`` / ``effective_to`` mean
    on the row; non-SCD2 writes always leave this ``None``.
    """

    def to_dagster_metadata(self) -> dict[str, MetadataValue]:
        """Convert to Dagster output metadata dict.

        Returns a dict suitable for passing to
        ``context.add_output_metadata()``.
        """
        from dagster import MetadataValue

        metadata: dict[str, MetadataValue] = {
            "write_mode": MetadataValue.text(str(self.write_mode.value)),
            "target_table": MetadataValue.text(self.table_name),
        }

        if self.layer is not None:
            metadata["layer"] = MetadataValue.text(self.layer)
        if self.source_file is not None:
            metadata["source_file"] = MetadataValue.text(self.source_file)
        if self.primary_key is not None:
            metadata["primary_key"] = MetadataValue.text(", ".join(self.primary_key))
        if self.partition_column is not None:
            metadata["partition_column"] = MetadataValue.text(self.partition_column)
        if self.business_key is not None:
            metadata["business_key"] = MetadataValue.text(", ".join(self.business_key))
        if self.lineage_id is not None:
            metadata["lineage_id"] = MetadataValue.text(self.lineage_id)
        if self.lineage_key is not None:
            metadata["lineage_key"] = MetadataValue.text(self.lineage_key)
        # Backfill signals are always emitted (boolean is informative even
        # when False; backfill_id is omitted when None to keep the
        # materialization-event payload clean for normal runs).
        metadata["is_backfill"] = MetadataValue.bool(self.is_backfill)
        if self.backfill_id is not None:
            metadata["backfill_id"] = MetadataValue.text(self.backfill_id)
        # ``replaces_lineage_id`` is omitted when ``None``; on the rare
        # FULL_REFRESH that links a chain it's load-bearing for audit.
        if self.replaces_lineage_id is not None:
            metadata["replaces_lineage_id"] = MetadataValue.text(self.replaces_lineage_id)
        # ``parent_lineage_count`` always emits as ``int`` -- ``0`` is
        # informative ("no upstream lineage found in this write") and
        # differentiates the no-parent case from "lineage disabled".
        metadata["parent_lineage_count"] = MetadataValue.int(self.parent_lineage_count)

        # Wall-clock duration of the write() call. Emitted only when set;
        # the production write path always sets it, but a
        # ``WriteResult(...)`` constructed by test code without going
        # through ``write()`` legitimately leaves it ``None``.
        if self.duration_seconds is not None:
            metadata["duration_seconds"] = MetadataValue.float(self.duration_seconds)
            # Throughput is a derived metric, but it's the one operators
            # actually use to spot run-over-run perf regressions; compute
            # it here so callers don't have to.
            if self.duration_seconds > 0 and self.row_count > 0:
                metadata["throughput_rows_per_sec"] = MetadataValue.float(
                    round(self.row_count / self.duration_seconds, 1)
                )

        # Partition key value(s) -- not the column name (that's
        # ``partition_column``), the actual key being written.
        if self.partition_keys:
            metadata["partition_key"] = MetadataValue.text(", ".join(self.partition_keys))

        # from_ingest bronze writes carry a resolved blob path here;
        # surface it so consumers can answer "which file was loaded"
        # without joining to data_lineage.
        if self.source_uri is not None:
            metadata["source_uri"] = MetadataValue.text(self.source_uri)

        if self.pipeline_id is not None:
            metadata["pipeline_id"] = MetadataValue.text(self.pipeline_id)

        # SCD2 effective_date override is audit-critical when set; never
        # emitted on non-SCD2 writes because it's always ``None`` there.
        if self.effective_date is not None:
            metadata["effective_date"] = MetadataValue.text(self.effective_date.isoformat())

        # Column info
        if self.columns:
            metadata["column_count"] = MetadataValue.int(len(self.columns))
            metadata["columns"] = MetadataValue.text(", ".join(self.columns))

        # Row count (always present for programmatic access)
        metadata["row_count"] = MetadataValue.int(self.row_count)

        # Write mode specific stats
        for key, value in self.stats.items():
            # ``bool`` is a subclass of ``int``; check it first so it routes
            # to ``MetadataValue.bool`` and not ``MetadataValue.int``.
            if isinstance(value, bool):
                metadata[key] = MetadataValue.bool(value)
            elif isinstance(value, int):
                metadata[key] = MetadataValue.int(value)
            elif isinstance(value, float):
                metadata[key] = MetadataValue.float(value)
            elif isinstance(value, str):
                metadata[key] = MetadataValue.text(value)

        # Batch info
        if self.batch_count > 1:
            metadata["rows_written"] = MetadataValue.int(self.row_count)
            metadata["batches_written"] = MetadataValue.int(self.batch_count)

        # Contract summary
        if self.contract_summary is not None:
            metadata["contract_version"] = MetadataValue.text(
                self.contract_summary.contract_version
            )
            metadata["contract_status"] = MetadataValue.text(self.contract_summary.status)
            metadata["contract_checks_passed"] = MetadataValue.int(
                self.contract_summary.passed_checks
            )
            metadata["contract_checks_failed"] = MetadataValue.int(
                self.contract_summary.failed_checks
            )
            metadata["contract_checks_warned"] = MetadataValue.int(
                self.contract_summary.warned_checks
            )
            metadata["contract_checks_total"] = MetadataValue.int(
                self.contract_summary.total_checks
            )
            if self.contract_summary.violations:
                metadata["contract_violations"] = MetadataValue.text(
                    "; ".join(self.contract_summary.violations)
                )

        # PII column names from contract
        if self.contract is not None:
            pii_names = self.contract.get_pii_column_names()
            if pii_names:
                metadata["pii_columns"] = MetadataValue.text(", ".join(pii_names))
                metadata["pii_column_count"] = MetadataValue.int(len(pii_names))

        # Dagster TableSchema for native column schema viewer
        self._add_column_schema(metadata)

        return metadata

    def _add_column_schema(self, metadata: dict[str, MetadataValue]) -> None:
        """Build and attach ``dagster/column_schema`` to metadata.

        When a contract is available, emits rich column metadata (type,
        description, constraints, PII flags). Falls back to bare column
        names from the DataFrame when no contract exists.
        """
        from dagster import MetadataValue, TableColumn, TableColumnConstraints, TableSchema

        if self.contract is not None:
            table_columns: list[TableColumn] = []
            for col in self.contract.schema.columns:
                if col.managed:
                    continue

                dagster_type = COLUMN_TYPE_MAP.get(str(col.type), str(col.type))

                description = col.description or ""
                if col.pii:
                    description = f"[PHI] {description}".strip()

                has_unique = any(t.test_type == "unique" for t in col.tests)

                constraints = TableColumnConstraints(
                    nullable=col.nullable,
                    unique=has_unique,
                    other=["Contains PHI"] if col.pii else [],
                )

                table_columns.append(
                    TableColumn(
                        name=col.name,
                        type=dagster_type,
                        description=description or None,
                        constraints=constraints,
                    )
                )

            metadata["dagster/column_schema"] = MetadataValue.table_schema(
                TableSchema(columns=table_columns)
            )

        elif self.columns:
            import logging

            logging.getLogger("moncpipelib.resources").warning(
                "No contract available for %s; column schema derived from "
                "DataFrame columns (no type or description metadata)",
                self.table_name,
            )
            table_columns = [TableColumn(name=c) for c in self.columns]
            metadata["dagster/column_schema"] = MetadataValue.table_schema(
                TableSchema(columns=table_columns)
            )
