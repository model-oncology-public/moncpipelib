"""Shared resolver-driven discovery + credential logic (#415 Phase 5).

Both resolver-backed patterns -- ``api_resolver`` and ``api_crawl`` --
discover partitions the same way: delegate to a registered
:class:`~moncpipelib.ingest.resolvers.ReleaseResolver`, diff
``historical_release`` (falling back to ``current_release``), and key
partitions via ``partition.key_from``.  They also share the #216
credential lifecycle (secret resolved per call via ``ctx.secrets``, no
caching) and the #253 ``discovery_requires_auth`` opt-out.  That logic
was previously private to :class:`ApiResolverPattern`; it lives here so
``api_crawl`` reuses it verbatim instead of duplicating it.

``pattern_name`` parameterizes error messages only -- behavior is
identical across callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.resolvers import get_resolver
from moncpipelib.ingest.types import PartitionSpec

if TYPE_CHECKING:
    from moncpipelib.ingest.types import IngestContext


def fetch_api_key(
    ctx: IngestContext,
    cfg: dict[str, Any],
    *,
    pattern_name: str,
) -> str | None:
    """Fetch the api_key via ``ctx.secrets``, or ``None`` for
    credential-less contracts (per #218).

    Lifecycle: when the contract declares a ``credential`` block, the
    dispatcher resolves the secret per call (no caching across calls
    per #216).  When the contract omits ``credential`` (e.g. the
    calendar resolver, or public crawl APIs like RxClass), this returns
    ``None`` and ``ctx.secrets`` need not be populated.
    """
    credential = cfg.get("credential")
    if credential is None:
        return None
    if ctx.secrets is None:
        raise IngestResolutionError(
            f"{pattern_name} contract declared a 'credential' block "
            "but ctx.secrets is None.  The dispatcher / sensor must "
            "populate ctx.secrets with a KeyVaultSecretResource per "
            "the #216 credential-lifecycle decision."
        )
    secret_name = credential["secret_name"]
    return ctx.secrets.get_secret(secret_name)


def _discovery_api_key(
    ctx: IngestContext,
    cfg: dict[str, Any],
    resolver: Any,
    *,
    pattern_name: str,
) -> str | None:
    """Resolve the api_key for a discovery-time call, honoring #253.

    When the resolver declares ``discovery_requires_auth = False`` the
    api_key is NOT fetched even if the contract has a ``credential``
    block -- lets discovery sensors run on daemon pods without Key
    Vault access for resolvers whose list endpoints are public.
    Default ``True`` preserves prior behavior for resolvers that don't
    declare the attribute.
    """
    if getattr(resolver, "discovery_requires_auth", True):
        return fetch_api_key(ctx, cfg, pattern_name=pattern_name)
    return None


def discover_partitions_via_resolver(
    cfg: dict[str, Any],
    ctx: IngestContext,
    *,
    pattern_name: str,
) -> list[PartitionSpec]:
    """Resolver-driven partition discovery (per #228 / #253).

    Calls the resolver's ``historical_release`` and emits one
    :class:`PartitionSpec` per returned release; when it returns ``[]``
    (resolvers that opt out of historical), falls back to
    ``current_release`` for a single spec.  The discovery sensor does
    state-based diff against the dynamic-partitions registry, so the
    every-tick call is safe + post-DR-friendly.
    """
    resolver = get_resolver(cfg["resolver"])
    resolver_config = cfg.get("resolver_config") or {}
    api_key = _discovery_api_key(ctx, cfg, resolver, pattern_name=pattern_name)

    releases = resolver.historical_release(api_key, resolver_config, ctx)
    if not releases:
        releases = [resolver.current_release(api_key, resolver_config, ctx)]

    key_from = cfg["partition"]["key_from"]
    specs: list[PartitionSpec] = []
    for release in releases:
        if key_from not in release:
            raise IngestResolutionError(
                f"resolver {cfg['resolver']!r} did not return field {key_from!r} "
                f"required by partition.key_from; got fields: {sorted(release)}"
            )
        specs.append(PartitionSpec(key=str(release[key_from]), metadata=dict(release)))
    return specs


def partition_metadata_via_resolver(
    cfg: dict[str, Any],
    partition_key: str,
    ctx: IngestContext,
    *,
    pattern_name: str,
) -> dict[str, Any]:
    """Re-resolve the release dict for ``partition_key`` (per #256).

    Lookup order: ``historical_release`` match -> ``current_release``
    match -> ``{}`` (the dispatcher then falls back to
    ``partition_spec.metadata``).  Resolver-returned fields are
    persisted durably into ``_manifest.json`` -- resolvers MUST NOT
    include api_keys, signed URLs, or PHI in the returned dict.
    """
    resolver = get_resolver(cfg["resolver"])
    resolver_config = cfg.get("resolver_config") or {}
    api_key = _discovery_api_key(ctx, cfg, resolver, pattern_name=pattern_name)

    releases = resolver.historical_release(api_key, resolver_config, ctx)
    match = next(
        (r for r in releases if str(r.get("partition_key")) == partition_key),
        None,
    )
    if match is not None:
        return dict(match)

    current = resolver.current_release(api_key, resolver_config, ctx)
    if str(current.get("partition_key")) == partition_key:
        return dict(current)
    return {}
