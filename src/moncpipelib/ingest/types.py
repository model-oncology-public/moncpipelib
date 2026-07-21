"""Dataclasses shared across ingest patterns and the consumer-side resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from moncpipelib.resources.keyvault import KeyVaultSecretResource
    from moncpipelib.resources.types import LoggingContext


_T = TypeVar("_T")


@dataclass(frozen=True)
class PartitionSpec:
    """A single partition to be materialized by an ingest pattern.

    ``metadata`` carries pattern-specific context needed both during
    materialization (e.g. the period record for ``http_urls``) and
    downstream when the ingest manifest is written (Phase 2). The shape
    of ``metadata`` is intentionally unconstrained -- each pattern
    defines its own layout.
    """

    key: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of materializing a single file within an ingest partition.

    One ingest partition may produce multiple files (e.g. one URL that
    unzips into several CSVs). Each file yields one ``IngestResult``.

    Attributes:
        path: Blob path relative to the container.
        sha256: Hex-encoded SHA-256 of the file's content.
        action: ``"uploaded"`` if the file was written this run,
            ``"skipped"`` if the existing blob's sha256 matched.
        size_bytes: Size of the file's content in bytes.
    """

    path: str
    sha256: str
    action: Literal["uploaded", "skipped"]
    size_bytes: int


@dataclass(frozen=True)
class BlobRef:
    """Pointer to a landed blob, returned by the consumer-side resolver."""

    sensitivity: Literal["public", "confidential", "phi"]
    path: str


@dataclass(frozen=True)
class RawUrl:
    """Legacy fallback: an upstream URL the consumer must fetch directly.

    Returned by the resolver when a ``DataSource`` has not been migrated
    to the ingest boundary yet (``ingest_source is None``). Preserved
    during the gradual rollout; deprecated once every source declares an
    ingest contract.
    """

    url: str


@dataclass(frozen=True, slots=True)
class IngestContext:
    """Execution context threaded through ingest dispatcher / pattern / resolver.

    Bundles the per-call mutables (logger, secret broker, run identifier)
    so signatures stay stable as the framework grows.  Mirrors the
    :class:`~moncpipelib.resources.types.WriteContext` pattern from the
    postgres write path.

    Adopted ahead of strict need (per the resolved planning decision in
    moncpipelib#216): future additions -- tracing span IDs, retry policy
    overrides, structured-error sinks -- become field additions on the
    dataclass rather than Protocol revisions.

    Attributes:
        log: Logger satisfying
            :class:`~moncpipelib.resources.types.LoggingContext`.
            Required.  Used for the per-file audit trail
            (``resolver=<name> source=<source_name>
            partition_key=<key> path=<blob> action=<uploaded|skipped>``).
        secrets: Key Vault resource for fetching API credentials.
            Optional -- ``http_urls`` does not need it; ``api_resolver``
            requires it (the dispatcher resolves the secret at
            materialization time, never caches across calls per the
            credential-lifecycle decision in moncpipelib#216).
        run_id: Dagster run ID for audit correlation.  ``None`` is
            permissible in unit tests; production callers always populate
            it from the Dagster execution context.
    """

    log: LoggingContext
    secrets: KeyVaultSecretResource | None = None
    run_id: str | None = None
    _cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)
    """Per-ctx memoization scratch for resolver lookups.

    Scope: one :class:`IngestContext` instance -- typically a single sensor
    tick or a single materialization.  Used by ``get_or_compute`` to dedupe
    resolver calls within that scope (e.g. ``UtsReleaseResolver.historical_release``
    is hit by both ``resolve_url`` and ``ApiResolverPattern.partition_metadata``;
    the cache collapses them to one HTTP fetch per ctx).

    Not for cross-run state. The dataclass remains ``frozen=True`` -- the field
    rebinds are forbidden, but mutating the dict in place is permitted and is
    how cache writes happen.
    """

    def get_or_compute(
        self,
        key: Hashable,
        factory: Callable[[], _T],
    ) -> _T:
        """Per-ctx memoization. Calls ``factory`` once per unique ``key`` per ctx.

        Resolvers and patterns may use this to dedupe expensive lookups when
        more than one call site within a single materialization (or sensor
        tick) needs the same upstream data -- e.g.
        :meth:`UtsReleaseResolver.historical_release` is invoked by both
        ``resolve_url`` and the manifest's ``partition_metadata`` path; the
        cache collapses them to one HTTP fetch.

        ``key`` MUST be hashable; resolvers should compose it from the
        resolver name plus the subset of ``config`` that affects the result
        (e.g. ``("uts.historical", release_type, start_date_str)``).

        ``factory`` is invoked exactly once per cache miss.  Exceptions
        propagate uncaught and the cache is NOT populated -- a retry on the
        same key re-runs the factory.
        """
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]  # type: ignore[no-any-return]
