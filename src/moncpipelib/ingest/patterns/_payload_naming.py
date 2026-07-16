"""Non-archive payload filename derivation (#270).

Shared between :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`
and :class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern`
so both produce identical filename-resolution semantics + a single
INFO log shape that is searchable across pipelines.

The precedence chain (highest -> lowest):

1. ``IngestContract.payload_filename_template`` -- author-supplied,
   rendered through :func:`~moncpipelib.ingest.prefix.render_payload_filename`.
   Used verbatim (no sanitization); a malformed authored name fails
   loudly at upload time.
2. Resolver-supplied filename hint
   (:attr:`~moncpipelib.ingest.resolvers.ResolvedDownload.filename`,
   ``api_resolver`` only).  Used verbatim (no sanitization); the
   resolver code is in-repo and audited.
3. ``Content-Disposition: attachment; filename="..."`` from the
   server response.  PASSED THROUGH
   :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename` because
   the value is server-controlled.
4. URL basename (``Path(urlparse(url).path).name``, URL-decoded +
   sanitized via :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename`).
5. Raise :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`
   when every level above produces an empty-after-sanitize result.

Each successful resolution emits a single INFO line so an operator can
answer "why did this land here?" from log search alone (no shell access
to landing pods).  The line is consistent with the redacting transport
hooks: it never includes the api_key, query string, or response body.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.filenames import sanitize_blob_filename
from moncpipelib.ingest.prefix import render_payload_filename

if TYPE_CHECKING:
    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext


def resolve_payload_filename(
    contract: IngestContract,
    partition_key: str,
    *,
    resolver_filename: str | None,
    content_disposition_filename: str | None,
    url: str,
    prefix: str,
    ctx: IngestContext,
) -> str:
    """Resolve the blob filename for a non-archive (``extract: []``) payload.

    Walks the precedence chain documented at the module level.  Emits a
    single INFO log line via ``ctx.log`` recording which precedence
    level produced the name and the resulting prefix + filename.

    Args:
        contract: The ingest contract -- read for
            ``payload_filename_template`` and ``source_name``.
        partition_key: The partition being materialized; substituted
            into the template if present.
        resolver_filename: Resolver-supplied filename hint (raw, not
            sanitized).  ``None`` for ``http_urls`` (which has no
            resolver) or for resolvers that opt out.
        content_disposition_filename: Naive-parsed ``filename=``
            parameter from the response's ``Content-Disposition``
            header.  ``None`` when the header is absent.  Sanitized
            here before use.
        url: The fetched URL, used as the URL-basename precedence-fallback.
        prefix: The rendered blob prefix (for log context only -- not
            used in the derivation).
        ctx: Ingest context; ``ctx.log`` receives the audit line.

    Returns:
        The chosen filename (a non-empty string).

    Raises:
        IngestResolutionError: When every precedence level produces an
            empty-after-sanitize result.  The error message names the
            inputs that were considered so the contract author can fix
            the underlying gap (e.g. add a ``payload_filename_template``).
    """
    log: Any = ctx.log

    template = contract.payload_filename_template
    if template is not None:
        rendered = render_payload_filename(template, partition_key, contract)
        _emit_decision_log(
            log,
            contract=contract,
            partition_key=partition_key,
            source_level="template",
            name=rendered,
            prefix=prefix,
        )
        return rendered

    if resolver_filename:
        _emit_decision_log(
            log,
            contract=contract,
            partition_key=partition_key,
            source_level="resolver_hint",
            name=resolver_filename,
            prefix=prefix,
        )
        return resolver_filename

    if content_disposition_filename:
        sanitized = sanitize_blob_filename(content_disposition_filename)
        if sanitized:
            _emit_decision_log(
                log,
                contract=contract,
                partition_key=partition_key,
                source_level="content_disposition",
                name=sanitized,
                prefix=prefix,
            )
            return sanitized

    url_basename = PurePosixPath(urlparse(url).path).name
    if url_basename:
        sanitized = sanitize_blob_filename(url_basename)
        if sanitized:
            _emit_decision_log(
                log,
                contract=contract,
                partition_key=partition_key,
                source_level="url_basename",
                name=sanitized,
                prefix=prefix,
            )
            return sanitized

    raise IngestResolutionError(
        f"Cannot derive payload filename for partition {partition_key!r} of "
        f"ingest source {contract.source_name!r}: no payload_filename_template, "
        f"no resolver filename hint, no Content-Disposition filename, and the "
        f"URL has no usable basename. Consider setting "
        f"'payload_filename_template' on the ingest contract."
    )


def _emit_decision_log(
    log: Any,
    *,
    contract: IngestContract,
    partition_key: str,
    source_level: str,
    name: str,
    prefix: str,
) -> None:
    """Single uniform INFO line for the precedence decision.

    ``log`` is typed Any because at runtime it is a ``logging.Logger``-like
    (e.g. Dagster's ``context.log``); the protocol shape would ripple a
    Protocol change for negligible gain.  This mirrors the existing
    cast in :class:`HttpUrlsPattern` validate_content rejection logging.
    """
    log.info(
        "ingest.payload_filename: source=%s partition=%s source_level=%s name=%r prefix=%s",
        contract.source_name,
        partition_key,
        source_level,
        name,
        prefix,
    )
