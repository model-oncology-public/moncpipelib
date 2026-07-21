"""``http_urls`` ingest pattern.

For sources with known, enumerable URLs per period.  Covers CMS ASP,
FDA NDC (static manifest URL), and FDA Purplebook (per-period fallback
via ordered URL list).

Idempotency is ``hash_compare``: stream-download to a tempfile, walk
extracted files one at a time, sha256-compare each against the landed
blob's ``x-ms-meta-sha256`` metadata, upload only on mismatch.  Peak
memory is one extracted file (typically MBs) -- the archive itself is
never fully materialized.

Per #228, an optional ``validate_content`` block in the contract's
``http_urls`` inner block switches the URL list semantics for a period
from "process all as union" (default) to "try in order; take first
valid" (fallback list).  Predicates are evaluated against the
Content-Type header + first chunk of bytes BEFORE the rest of the body
is buffered to disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from moncpipelib.ingest._http import (
    ContentValidationRejected,
    DownloadedPayload,
    ValidateFn,
    build_redacting_client,
    stream_to_tempfile,
)
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns._extract import _DEFAULT_PAYLOAD_NAME, extract_and_filter_iter
from moncpipelib.ingest.patterns._payload_naming import resolve_payload_filename
from moncpipelib.ingest.patterns._upload import hash_compare_and_upload
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.types import IngestResult, PartitionSpec

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext
    from moncpipelib.resources.blob import BlobStorageResource


_DEFAULT_MAX_FIRST_BYTES_CHECK: int = 256


def _build_validator(cfg: dict[str, Any]) -> ValidateFn:
    """Compose a predicate from a ``validate_content`` config dict.

    Returns a callable suitable for passing to
    :func:`~moncpipelib.ingest._http.stream_to_tempfile` as ``validate``.
    The predicate is whitelist-then-blacklist: ``content_type_in`` must
    match (when set), then ``reject_first_bytes_match`` must NOT match
    (when set).
    """
    content_type_in: list[str] = list(cfg.get("content_type_in") or [])
    reject_first_bytes_match: list[bytes] = [
        s.encode("utf-8").lower() for s in (cfg.get("reject_first_bytes_match") or [])
    ]
    max_first_bytes_check: int = int(
        cfg.get("max_first_bytes_check", _DEFAULT_MAX_FIRST_BYTES_CHECK)
    )

    def _validate(content_type: str | None, first_bytes: bytes) -> ContentValidationRejected | None:
        if content_type_in:
            ct_prefix = (content_type or "").split(";", 1)[0].strip().lower()
            if not any(ct_prefix == allowed.lower() for allowed in content_type_in):
                return ContentValidationRejected(
                    "content_type",
                    content_type=content_type,
                    first_bytes=first_bytes[:max_first_bytes_check],
                )
        if reject_first_bytes_match:
            stripped = first_bytes.lstrip()[:max_first_bytes_check].lower()
            if any(stripped.startswith(needle) for needle in reject_first_bytes_match):
                return ContentValidationRejected(
                    "first_bytes",
                    content_type=content_type,
                    first_bytes=first_bytes[:max_first_bytes_check],
                )
        return None

    return _validate


class HttpUrlsPattern:
    """Static HTTP-URL ingest pattern."""

    name: ClassVar[str] = "http_urls"

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Enumerate partitions from the contract's declared periods."""
        del ctx  # http_urls does not need ctx.secrets / ctx.log here
        periods = contract.pattern_config.get("periods") or []
        specs: list[PartitionSpec] = []
        for period in periods:
            if not isinstance(period, dict):
                continue
            pk = period.get("partition_key")
            if pk is None:
                continue
            specs.append(PartitionSpec(key=str(pk), metadata=dict(period)))
        return specs

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Stream-download, extract, hash-compare, and upload one partition.

        URL list semantics depend on whether the contract declares a
        ``validate_content`` block (per #228):

        - **Unset:** all URLs in ``period.urls`` are processed; their
          extracted files form the partition's union.  v0.27 behavior.
        - **Set:** URLs are tried in order; the first one whose response
          passes the predicate is used; rejected responses are logged
          and skipped; if all URLs are exhausted with rejections, an
          :class:`IngestResolutionError` is raised with aggregated
          rejection context.
        """
        period = self._find_period(contract, partition_spec.key)
        prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)
        fetch_cfg: dict[str, Any] = contract.pattern_config.get("fetch", {}) or {}
        validate_content_cfg: dict[str, Any] | None = contract.pattern_config.get(
            "validate_content"
        )

        retries = int(fetch_cfg.get("retries", 3))
        timeout_s = float(fetch_cfg.get("timeout_s", 120))
        connect_timeout_s = float(fetch_cfg.get("connect_timeout_s", 10))
        follow_redirects = bool(fetch_cfg.get("follow_redirects", True))
        ua_cfg = fetch_cfg.get("user_agent")
        user_agent = str(ua_cfg) if ua_cfg is not None else None

        results: list[IngestResult] = []
        with build_redacting_client(
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            retries=retries,
            follow_redirects=follow_redirects,
            user_agent=user_agent,
        ) as client:
            if validate_content_cfg is None:
                # Default semantics (v0.27): process all URLs as a union.
                for url in period["urls"]:
                    with stream_to_tempfile(client, url) as payload:
                        results.extend(
                            self._extract_and_upload(
                                contract,
                                blob,
                                prefix,
                                payload,
                                url=url,
                                partition_key=partition_spec.key,
                                ctx=ctx,
                            )
                        )
            else:
                # Fallback-list semantics (per #228): try URLs in order;
                # take the first one whose response passes the predicate.
                results.extend(
                    self._materialize_with_validate_content(
                        contract,
                        blob,
                        ctx,
                        client,
                        period,
                        prefix,
                        validate_content_cfg,
                    )
                )
        return results

    def _materialize_with_validate_content(
        self,
        contract: IngestContract,
        blob: BlobStorageResource,
        ctx: IngestContext,
        client: Any,
        period: dict[str, Any],
        prefix: str,
        validate_content_cfg: dict[str, Any],
    ) -> list[IngestResult]:
        """Try URLs in order; take the first one that passes validation."""
        validate = _build_validator(validate_content_cfg)
        max_first_bytes_check = int(
            validate_content_cfg.get("max_first_bytes_check", _DEFAULT_MAX_FIRST_BYTES_CHECK)
        )
        partition_key = str(period.get("partition_key"))
        rejections: list[ContentValidationRejected] = []

        for url_index, url in enumerate(period["urls"]):
            try:
                with stream_to_tempfile(client, url, validate=validate) as payload:
                    return self._extract_and_upload(
                        contract,
                        blob,
                        prefix,
                        payload,
                        url=url,
                        partition_key=partition_key,
                        ctx=ctx,
                    )
            except ContentValidationRejected as rejection:
                # Per #228: piggyback on the redacting transport hook
                # (which already logged host+path); add only the new
                # signal -- period, url_index, reason, content_type,
                # first_bytes excerpt.  Crucially, no raw URL.
                # ctx.log is annotated LoggingContext but at runtime is a
                # ``logging.Logger``-like (e.g. Dagster's ``context.log``);
                # cast to keep mypy quiet without rippling a Protocol change.
                _log: Any = ctx.log
                _log.info(
                    "validate_content rejected: period=%r url_index=%d "
                    "reason=%s observed_content_type=%r first_bytes=%r",
                    partition_key,
                    url_index,
                    rejection.reason,
                    rejection.content_type,
                    rejection.first_bytes[:max_first_bytes_check],
                )
                rejections.append(rejection)

        # All URLs exhausted: aggregate rejection context (no raw URLs).
        details = "; ".join(
            f"url_index={i} reason={r.reason} "
            f"observed_content_type={r.content_type!r} "
            f"first_bytes={r.first_bytes[:max_first_bytes_check]!r}"
            for i, r in enumerate(rejections)
        )
        raise IngestResolutionError(
            f"no URL passed validate_content for partition {partition_key!r}: {details}"
        )

    def _extract_and_upload(
        self,
        contract: IngestContract,
        blob: BlobStorageResource,
        prefix: str,
        payload: DownloadedPayload,
        *,
        url: str,
        partition_key: str,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Extract one downloaded archive + hash-compare + upload.

        For non-archive contracts (``extract: []``), the helper-supplied
        :data:`_DEFAULT_PAYLOAD_NAME` sentinel is replaced with a
        precedence-derived filename via
        :func:`~moncpipelib.ingest.patterns._payload_naming.resolve_payload_filename`
        before upload (#270).  Archive contracts retain the per-member
        names yielded by :func:`extract_and_filter_iter`.
        """
        results: list[IngestResult] = []
        for filename, member_path, sha, size_bytes in extract_and_filter_iter(
            payload.path,
            contract.extract,
            contract.strip_extensions,
            contract.extract_filter,
        ):
            if filename == _DEFAULT_PAYLOAD_NAME and not contract.extract:
                # Non-archive: substitute the precedence-derived filename
                # for the helper sentinel.  Archive members never collide
                # with the sentinel (they keep their in-archive paths).
                filename = resolve_payload_filename(
                    contract,
                    partition_key,
                    resolver_filename=None,  # http_urls has no resolver
                    content_disposition_filename=payload.content_disposition_filename,
                    url=url,
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
        """Return the period dict for ``partition_key`` from the contract.

        Per #256: symmetric with
        :meth:`ApiResolverPattern.partition_metadata` so the dispatcher's
        manifest-write path is uniform across patterns.  Returns the
        same shape :meth:`discover_partitions` placed on
        :class:`PartitionSpec.metadata` for the same key.

        Today's ``http_urls`` consumers don't use
        :class:`~moncpipelib.contracts.models.FromIngestTemplate`, so
        ``manifest.fields`` is read by no one in the existing code.
        Populating it anyway keeps the symmetry with ``api_resolver``
        and lets future consumers (e.g. an ``http_urls`` source whose
        downstream wants ``effective_from_field``) work without further
        changes.

        Returns ``{}`` if no period matches -- the dispatcher then falls
        back to ``partition_spec.metadata`` (preserving back-compat).
        """
        del ctx
        try:
            return dict(self._find_period(contract, partition_key))
        except KeyError:
            return {}

    @staticmethod
    def _find_period(contract: IngestContract, partition_key: str) -> dict[str, Any]:
        for period in contract.pattern_config.get("periods") or []:
            if isinstance(period, dict) and str(period.get("partition_key")) == partition_key:
                return period
        raise KeyError(
            f"No period with partition_key={partition_key!r} in ingest contract "
            f"{contract.source_name!r}"
        )
