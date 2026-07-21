"""``api_resolver`` ingest pattern.

For sources whose download URL is resolved at fetch time via an
authenticated API call (e.g. UMLS Metathesaurus + RxNorm via the NLM
UTS Release API).  Pairs the pattern's download / extract / hash-compare
flow with a registered :class:`~moncpipelib.ingest.resolvers.ReleaseResolver`.

Per the cross-cutting decisions in moncpipelib#216:

- **Credential lifecycle**: the dispatcher resolves the secret per
  :meth:`materialize_partition` call (no caching across calls).  On a
  rotated key, the next tick picks up the new value.  ``ctx.secrets``
  must be set; an unset value raises a clear error.
- **No load-time side effects**: the resolver call lives inside
  :meth:`discover_partitions`; importing this module does not touch
  the network.  ``Definitions(...)`` construction stays clean.
- **Audit redaction**: all HTTP I/O routes through
  :func:`~moncpipelib.ingest._http.build_redacting_client` so the
  ``apiKey=...`` query param embedded in the resolved download URL
  never appears in transport-layer logs.
- **Streaming**: the download is written to a temp file via
  :func:`~moncpipelib.ingest._http.stream_to_tempfile`; extraction
  yields one terminal file at a time.  Peak memory is ~one extracted
  file even for the 5+ GB UMLS Metathesaurus archive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from moncpipelib.ingest._http import build_redacting_client, stream_to_tempfile
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns._extract import _DEFAULT_PAYLOAD_NAME, extract_and_filter_iter
from moncpipelib.ingest.patterns._payload_naming import resolve_payload_filename
from moncpipelib.ingest.patterns._resolver_discovery import (
    discover_partitions_via_resolver,
    fetch_api_key,
    partition_metadata_via_resolver,
)
from moncpipelib.ingest.patterns._upload import hash_compare_and_upload
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.resolvers import get_resolver
from moncpipelib.ingest.types import IngestResult, PartitionSpec

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext
    from moncpipelib.resources.blob import BlobStorageResource


class ApiResolverPattern:
    """Authenticated-resolver ingest pattern.

    The contract's ``api_resolver`` block points at a registered
    :class:`~moncpipelib.ingest.resolvers.ReleaseResolver` plus a
    ``credential.secret_name`` resolved via
    :class:`~moncpipelib.resources.keyvault.KeyVaultSecretResource`.

    Each tick:

    1. :meth:`discover_partitions` calls the resolver's
       ``current_release`` and emits a single :class:`PartitionSpec`
       for the current release.  Backfill of historical partitions is
       deferred to Phase 3 (``historical_release`` Protocol method).
    2. :meth:`materialize_partition` calls
       ``resolver.resolve_url(api_key, partition_key, ...)`` to get
       the authenticated download URL, streams the response to a
       tempfile, and runs the same hash-compare upload loop as
       :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`.

    Manifest writing lands in PR 5 via
    ``materialize_with_manifest``; this pattern returns an
    :class:`IngestResult` list for the dispatcher to write the
    manifest from.
    """

    name: ClassVar[str] = "api_resolver"

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Call the resolver's ``historical_release`` (per #228) and
        emit one :class:`PartitionSpec` per returned release.

        When ``historical_release`` returns ``[]`` (resolvers that
        don't support historical -- the calendar resolver opts out
        this way), fall back to ``current_release`` for a single spec.

        The discovery sensor at
        :func:`moncpipelib.ingest.sensors.build_discovery_sensor`
        does state-based diff: it adds any returned partition keys
        that aren't already in the dynamic-partitions registry.
        Re-adding existing keys is a no-op, so the every-tick
        ``historical_release`` call is safe + post-DR-friendly.

        When the resolver declares ``discovery_requires_auth = False``
        (per #253) the api_key is NOT fetched here even if the contract
        has a ``credential`` block -- ``current_release`` /
        ``historical_release`` receive ``api_key=None``.  Lets the
        discovery sensor run on daemon pods without Key Vault access
        for resolvers whose list endpoints are public (e.g. UTS).
        ``materialize_partition`` continues to fetch the key
        unconditionally because :meth:`resolve_url` typically needs it.
        """
        cfg = self._read_pattern_config(contract)
        return discover_partitions_via_resolver(cfg, ctx, pattern_name="ApiResolverPattern")

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Resolve the URL, stream-download, extract, hash-compare, upload."""
        cfg = self._read_pattern_config(contract)
        api_key = self._fetch_api_key(ctx, cfg)
        resolver = get_resolver(cfg["resolver"])
        resolved = resolver.resolve_url(
            api_key,
            partition_spec.key,
            cfg.get("resolver_config") or {},
            ctx,
        )

        prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)
        fetch_cfg: dict[str, Any] = cfg.get("fetch", {}) or {}
        retries = int(fetch_cfg.get("retries", 3))
        timeout_s = float(fetch_cfg.get("timeout_s", 3600))  # large UMLS download
        connect_timeout_s = float(fetch_cfg.get("connect_timeout_s", 30))
        ua_cfg = fetch_cfg.get("user_agent")
        user_agent = str(ua_cfg) if ua_cfg is not None else None

        results: list[IngestResult] = []
        with (
            build_redacting_client(
                timeout_s=timeout_s,
                connect_timeout_s=connect_timeout_s,
                retries=retries,
                follow_redirects=True,
                user_agent=user_agent,
            ) as client,
            stream_to_tempfile(client, resolved.url) as payload,
        ):
            for filename, member_path, sha, size_bytes in extract_and_filter_iter(
                payload.path,
                contract.extract,
                contract.strip_extensions,
                contract.extract_filter,
            ):
                if filename == _DEFAULT_PAYLOAD_NAME and not contract.extract:
                    # Non-archive: substitute the precedence-derived
                    # filename for the helper sentinel (#270).  The
                    # resolver hint is the api_resolver-only branch of
                    # the chain; templates / Content-Disposition / URL
                    # basename remain available below it.
                    filename = resolve_payload_filename(
                        contract,
                        partition_spec.key,
                        resolver_filename=resolved.filename,
                        content_disposition_filename=payload.content_disposition_filename,
                        url=resolved.url,
                        prefix=prefix,
                        ctx=ctx,
                    )
                results.append(
                    hash_compare_and_upload(
                        blob,
                        contract.sensitivity,
                        prefix,
                        filename,
                        member_path,
                        sha,
                        size_bytes,
                    )
                )
        return results

    def partition_metadata(
        self,
        contract: IngestContract,
        partition_key: str,
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return the resolver's release dict for ``partition_key``.

        Per #256: the manifest's ``fields`` block must carry the same
        release dict that :meth:`discover_partitions` placed on the
        :class:`PartitionSpec`, so consumers that hydrate via
        :class:`~moncpipelib.contracts.models.FromIngestTemplate` can
        read ``effective_from_field`` from it.  Asset bodies have only
        ``context.partition_key`` at materialize time -- the in-memory
        ``PartitionSpec.metadata`` from discovery is not retrievable
        from Dagster's dynamic-partitions registry -- so the dispatcher
        re-asks the pattern at manifest-write time via this method.

        Lookup order:

        1. ``resolver.historical_release`` -- if the partition_key is
           present, return the matching dict.  Cached on ``ctx`` per
           :meth:`UtsReleaseResolver.historical_release`, so when
           :meth:`materialize_partition` already ran ``resolve_url`` in
           the same context this is a free in-memory hit.
        2. ``resolver.current_release`` -- fallback for resolvers that
           opt out of historical (e.g.
           :class:`~moncpipelib.ingest.resolvers.calendar.CalendarReleaseResolver`).
        3. ``{}`` -- last resort if neither lookup yields a match.  The
           dispatcher then falls back to ``partition_spec.metadata``,
           preserving back-compat for any caller that populates spec
           metadata directly.

        Resolver-returned fields are persisted into ``_manifest.json``
        durably -- resolvers MUST NOT include api_keys, signed URLs, or
        PHI in the returned dict (see ``SECURITY.md`` and the resolver
        Protocol contract).
        """
        cfg = self._read_pattern_config(contract)
        return partition_metadata_via_resolver(
            cfg, partition_key, ctx, pattern_name="ApiResolverPattern"
        )

    def manifest_resolver_block(self, contract: IngestContract) -> dict[str, Any]:
        """Return the manifest's ``resolver`` audit block (per #415).

        Carries the resolver name + resolver_config so the durable
        audit trail names which API produced the partition.  Formerly
        built inline by the dispatcher's ``_build_manifest`` behind a
        ``pattern.name == "api_resolver"`` check; the dispatcher now
        asks the pattern via this optional method (same ``getattr``
        opt-in style as :meth:`partition_metadata`) and normalizes /
        JSON-coerces the returned values.  Manifest bytes for existing
        sources are unchanged.

        Redaction contract: persisted durably in ``_manifest.json`` --
        no api_keys, signed URLs, or PHI (see ``SECURITY.md``).
        """
        return {
            "name": str(contract.pattern_config.get("resolver", "unknown")),
            "config": dict(contract.pattern_config.get("resolver_config") or {}),
        }

    @staticmethod
    def _read_pattern_config(contract: IngestContract) -> dict[str, Any]:
        cfg = contract.pattern_config
        if not cfg:
            raise IngestResolutionError(
                f"Contract {contract.source_name!r} has empty api_resolver config"
            )
        return cfg

    @staticmethod
    def _fetch_api_key(ctx: IngestContext, cfg: dict[str, Any]) -> str | None:
        """Fetch the api_key via ctx.secrets, or return ``None`` for
        resolvers that don't authenticate (per #218).

        Thin wrapper over the shared
        :func:`~moncpipelib.ingest.patterns._resolver_discovery.fetch_api_key`
        (per #415 both resolver-backed patterns share the #216
        credential lifecycle).
        """
        return fetch_api_key(ctx, cfg, pattern_name="ApiResolverPattern")
