"""Ingest pattern protocol and registry.

Patterns are registered by name at import time.  ``get_pattern`` is the
only lookup callers should use -- it raises a clear error if the
pattern name is unknown, surfacing contract-level typos early.

Patterns do **not** write the per-partition manifest themselves --
that lands in PR 5 via a centralized ``materialize_with_manifest``
dispatcher so new patterns cannot forget the manifest write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext, IngestResult, PartitionSpec
    from moncpipelib.resources.blob import BlobStorageResource


@runtime_checkable
class IngestPattern(Protocol):
    """Protocol every ingest pattern implements.

    Both methods take an :class:`~moncpipelib.ingest.types.IngestContext`
    that bundles the per-call mutables (logger, secret broker, run
    identifier).  Patterns that do not need ``ctx.secrets`` (e.g.
    :class:`HttpUrlsPattern`) ignore it; ``api_resolver`` requires it
    and raises a clear error when absent.
    """

    name: ClassVar[str]

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Return the list of partitions this contract should materialize.

        For static patterns (``http_urls``) this enumerates the
        declared period list.  For dynamic patterns (``api_resolver``)
        this issues an authenticated call to the upstream API via
        ``ctx.secrets``.

        Discovery sensors (PR 6) call this at every tick.  Patterns
        that hit the network MUST do so only inside this method, never
        at module import time -- ``Definitions(...)`` construction
        must remain network-free.
        """
        ...

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Materialize a single partition into blob storage.

        Each returned :class:`~moncpipelib.ingest.types.IngestResult`
        captures one landed file (``"uploaded"``) or one HEAD-check
        that matched an existing sha256 (``"skipped"``).
        """
        ...


INGEST_PATTERNS: dict[str, IngestPattern] = {}


def register_pattern(pattern: IngestPattern) -> None:
    """Register ``pattern`` under its ``name`` class variable.

    Subsequent calls with the same name overwrite the previous entry --
    useful for testing with a stub, never intended for production.
    """
    INGEST_PATTERNS[pattern.name] = pattern


def get_pattern(name: str) -> IngestPattern:
    """Look up a registered pattern by name.

    Raises:
        KeyError: If no pattern with that name is registered.
    """
    try:
        return INGEST_PATTERNS[name]
    except KeyError as e:
        known = sorted(INGEST_PATTERNS)
        raise KeyError(f"Unknown ingest pattern {name!r}. Known patterns: {known}") from e


def _register_builtin_patterns() -> None:
    """Register patterns shipped with moncpipelib.

    Called once at import time. Kept as a function so tests can clear
    the registry and reinitialise without reloading the module.
    """
    # Imported lazily to avoid a circular import at package load time
    # (each pattern imports from this module for the Protocol).
    from moncpipelib.ingest.patterns.api_crawl import ApiCrawlPattern
    from moncpipelib.ingest.patterns.api_resolver import ApiResolverPattern
    from moncpipelib.ingest.patterns.http_urls import HttpUrlsPattern

    register_pattern(HttpUrlsPattern())
    register_pattern(ApiResolverPattern())
    register_pattern(ApiCrawlPattern())


_register_builtin_patterns()


__all__ = [
    "INGEST_PATTERNS",
    "IngestPattern",
    "get_pattern",
    "register_pattern",
]
