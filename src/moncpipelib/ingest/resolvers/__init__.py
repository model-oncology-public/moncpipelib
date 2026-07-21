"""Release resolver Protocol and registry.

Resolvers are the bridge between an ``api_resolver`` ingest contract
and the upstream API that publishes the data.  Each resolver knows
how to call its specific API to discover the current release and
return an authenticated download URL.

Lifecycle (per ADR-2,
``docs/migrations/20260426_phase2-ingest-decisions.md``):

- Resolvers are stateless singletons.  The registry holds one
  instance per :attr:`ReleaseResolver.name`, constructed once at
  registration time.  ``__init__`` is parameterless; all per-call
  state flows through method arguments (``api_key``, ``config``,
  ``ctx``).
- ``validate_config`` runs at contract-load time -- including in CI.
  It MUST be deterministic, MUST NOT make network calls, MUST NOT
  perform filesystem I/O beyond the contract file, and MUST reject
  unknown keys.
- ``current_release`` and ``resolve_url`` run at sensor / dispatcher
  tick time.  They use :func:`moncpipelib.ingest._http.build_redacting_client`
  for any HTTP I/O so the audit log never includes the api_key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from moncpipelib.ingest.types import IngestContext


@dataclass(frozen=True)
class ResolvedDownload:
    """Authenticated download URL plus an optional semantic filename hint.

    Returned by :meth:`ReleaseResolver.resolve_url` per #270. The
    ``api_resolver`` materializer fetches ``url`` via
    :func:`~moncpipelib.ingest._http.build_redacting_client` and threads
    ``filename`` (when not :data:`None`) into the non-archive payload
    filename precedence chain (template -> resolver hint ->
    Content-Disposition -> sanitized URL basename -> raise).

    Attributes:
        url: Fully-resolved download URL. May embed an ``apiKey`` query
            parameter; the redacting transport hooks ensure it never
            appears in audit logs.
        filename: Optional semantic filename hint -- e.g. the UMLS
            release's ``download_url`` basename when the resolver knows
            the release version. :data:`None` when the resolver does
            not know a semantic filename for the partition; the
            precedence chain falls through to the next level.
            Resolver-supplied hints are NOT passed through
            :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename`
            -- a malformed authored hint should fail loudly at upload
            time rather than be silently rewritten.
    """

    url: str
    filename: str | None


@runtime_checkable
class ReleaseResolver(Protocol):
    """Protocol every release resolver implements.

    Stateless; one instance per registered name.  See module docstring
    for the lifecycle and validation contract pinned in ADR-2.
    """

    name: ClassVar[str]

    discovery_requires_auth: ClassVar[bool] = True
    """Whether ``current_release`` / ``historical_release`` need ``api_key``.

    When ``False``, :class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern`
    skips :meth:`_fetch_api_key` during ``discover_partitions`` and
    passes ``api_key=None`` to the resolver -- even when the contract
    declares a ``credential`` block.  This lets discovery sensors run
    on daemon pods without Key Vault access for resolvers whose list /
    current endpoints are public (e.g. UTS ``/releases``); the api_key
    is still fetched at ``materialize_partition`` time, where
    :meth:`resolve_url` typically needs it.

    Default is ``True`` -- preserves prior behavior for any resolver
    that does not opt out, including third-party resolvers structurally
    satisfying this Protocol without declaring the attribute.
    """

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Return a list of validation error strings (empty if valid).

        Called by ``_validate_api_resolver_block`` at contract-load
        time with the contents of
        ``ingest.api_resolver.resolver_config``.  Resolvers should
        validate required keys, value types, resolver-specific format
        rules, AND reject unknown keys (per ADR-2).

        Network calls are forbidden; filesystem I/O is forbidden; the
        function must be deterministic and fast (target < 1ms).
        """
        ...

    def current_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return metadata for the current release of this source.

        The returned mapping MUST include at minimum a
        ``"partition_key"`` key (used by the discovery sensor to add
        the partition to the registry) and any fields a downstream
        ``FromIngestTemplate`` consumer references via
        ``effective_from_field``.  Convention: include
        ``"download_url"`` for downstream materialization.

        ``api_key`` is ``None`` when the contract omits the
        ``credential`` block (e.g.
        :class:`~moncpipelib.ingest.resolvers.calendar.CalendarReleaseResolver`,
        per #218). Resolvers that require authentication should raise
        :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`
        when ``api_key is None``; resolvers that don't authenticate
        ignore the value.

        Per #256, the returned dict is persisted into ``_manifest.json``
        as the ``fields`` block.  Resolvers MUST NOT include API keys,
        signed URLs, or PHI in the returned mapping; only audit-safe
        release metadata (release version, release date, public
        download URL) belongs there.  See ``SECURITY.md`` for the
        durable-audit-surface contract.
        """
        ...

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> ResolvedDownload:
        """Return the authenticated download URL plus optional filename hint.

        Per #270, the return type is :class:`ResolvedDownload`. The
        ``url`` field MAY embed the api_key as a query param; the
        ``api_resolver`` materializer MUST fetch it with
        :func:`moncpipelib.ingest._http.build_redacting_client` so the
        URL never appears in any log. The ``filename`` field is an
        optional semantic hint surfaced into the non-archive payload
        filename precedence chain; resolvers that do not know a
        semantic filename for the partition return
        ``ResolvedDownload(url=..., filename=None)``.

        ``api_key`` is ``None`` for credential-less contracts; see
        :meth:`current_release` for the same opt-out semantics.
        """
        ...

    def historical_release(
        self,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> list[dict[str, Any]]:
        """Return release dicts for every still-available historical release.

        Per #228: includes the current release as the last (or first)
        element; downstream
        :class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern`
        diffs the returned list against the existing dynamic-partitions
        registry (state-based diff -- see
        ``docs/migrations/20260426_phase2-ingest-decisions.md``).

        Resolvers that don't have a "list historical" upstream (e.g.
        :class:`~moncpipelib.ingest.resolvers.calendar.CalendarReleaseResolver`)
        return ``[]`` to opt out -- the pattern then falls back to
        :meth:`current_release`.

        Bounds (``start_date`` etc.) live in ``config`` under whatever
        keys the resolver chose to expose; this Protocol method takes
        no kwargs so resolvers can pick their own config shape.

        Per #256: each entry is persisted into ``_manifest.json`` as
        the ``fields`` block when its ``partition_key`` matches the
        partition being materialized.  Same redaction contract as
        :meth:`current_release` -- no api_keys, no signed URLs, no PHI.

        Per #256: implementations SHOULD memoize the result on ``ctx``
        via :meth:`IngestContext.get_or_compute` so a single
        materialization's :meth:`resolve_url` call and the dispatcher's
        ``partition_metadata`` lookup share one upstream fetch.
        """
        ...


RESOLVERS: dict[str, ReleaseResolver] = {}


def register_resolver(resolver: ReleaseResolver) -> None:
    """Register ``resolver`` under its :attr:`ReleaseResolver.name`.

    Subsequent calls with the same name overwrite the previous entry
    -- useful for testing with a stub, never intended for production.
    """
    RESOLVERS[resolver.name] = resolver


def get_resolver(name: str) -> ReleaseResolver:
    """Look up a registered resolver by name.

    Raises:
        KeyError: If no resolver with that name is registered.  The
            message lists the known resolvers, so a YAML typo surfaces
            at contract-load time with a useful suggestion.
    """
    try:
        return RESOLVERS[name]
    except KeyError as e:
        known = sorted(RESOLVERS)
        raise KeyError(f"Unknown release resolver {name!r}. Known resolvers: {known}") from e


def _register_builtin_resolvers() -> None:
    """Register resolvers shipped with moncpipelib.

    Called once at import time.  Kept as a function so tests can clear
    the registry and re-initialise without reloading the module.
    """
    # Imported lazily so resolvers/__init__.py can finish loading
    # before resolver modules import from this module for the Protocol.
    from moncpipelib.ingest.resolvers.calendar import CalendarReleaseResolver
    from moncpipelib.ingest.resolvers.uts import UtsReleaseResolver

    register_resolver(UtsReleaseResolver())
    register_resolver(CalendarReleaseResolver())


_register_builtin_resolvers()


__all__ = [
    "RESOLVERS",
    "ReleaseResolver",
    "ResolvedDownload",
    "get_resolver",
    "register_resolver",
]
