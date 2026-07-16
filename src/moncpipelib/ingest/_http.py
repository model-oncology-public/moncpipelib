"""HTTP client factory + streaming download helper for ingest patterns.

Provides:

- :func:`build_redacting_client` -- an :class:`httpx.Client` factory
  whose request/response logging never includes URL query strings,
  request headers, or response bodies.  Audit boundary for the
  api_resolver flow: an ``apiKey=...`` query param or an
  ``Authorization: Bearer ...`` header cannot leak via the redacted
  transport hooks.
- :func:`stream_to_tempfile` -- a context manager that streams a GET
  response body to a temporary file on disk, yielding the file's
  :class:`pathlib.Path`.  Used by both :class:`HttpUrlsPattern` and
  :class:`ApiResolverPattern` so peak memory during download is
  bounded by the chunk size, not the file size.  Load-bearing for
  UMLS Metathesaurus (5+ GB on disk).  Accepts an optional
  ``validate`` callback (per #228) so the caller can reject a
  response on Content-Type or first-bytes BEFORE the body is
  buffered to disk.
- :class:`ContentValidationRejected` -- raised inside
  :func:`stream_to_tempfile` when the optional ``validate`` callback
  rejects the response.  Carries the rejection reason +
  observed Content-Type + first-bytes excerpt so the caller can log /
  aggregate rejection context without re-emitting the URL.

Per the credential-lifecycle decisions in moncpipelib#216:

- One audit-log line per request (``method=<METHOD> host=<HOST>
  path=<PATH>``); one per response (``status=<CODE>``).  No query
  string, no headers, no body.
- Resolvers and ``api_resolver`` materializers MUST use this factory
  rather than constructing their own :class:`httpx.Client`.  An
  unredacted client breaks the SOC 2 / HITRUST audit posture for
  authenticated upstream APIs.
- Hooks are best-effort: a caller who explicitly logs ``response.url``
  or ``request.headers`` from their own code still leaks.  The
  redacting client guards the *transport* layer; callers must not
  echo URLs/headers from their application code.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedPayload:
    """A streamed download's tempfile path plus optional ``Content-Disposition`` filename hint.

    Per #270, :func:`stream_to_tempfile` surfaces the response's
    ``Content-Disposition: attachment; filename="..."`` header alongside
    the tempfile path so the non-archive payload filename precedence
    chain (template -> resolver hint -> Content-Disposition -> sanitized
    URL basename -> raise) can use the server-supplied name when no
    higher-precedence input is set.

    The ``content_disposition_filename`` is the RAW server-supplied
    string (or :data:`None`); it is sanitized at the boundary by the
    pattern-level filename resolver via
    :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename` before
    reaching blob storage.

    Attributes:
        path: Tempfile containing the response body.  Owned by the
            :func:`stream_to_tempfile` context manager; unlinked on
            context exit.
        content_disposition_filename: The naive-parsed ``filename=``
            parameter from the response's ``Content-Disposition``
            header, or :data:`None` when the header is absent / has
            no ``filename=`` parameter / RFC 5987 ``filename*=`` only.
            RFC 5987 encoded form (``filename*=UTF-8''...``) is
            intentionally NOT parsed today; that is a follow-on if a
            real upstream needs it.
    """

    path: Path
    content_disposition_filename: str | None


def _parse_content_disposition_filename(header: str | None) -> str | None:
    """Naive RFC 6266 ``filename=`` parameter extraction.

    Splits on ``;`` and looks for a parameter whose key is exactly
    ``filename`` (case-insensitive); strips surrounding double-quotes.
    Returns :data:`None` when:

    - The header is :data:`None` or empty.
    - No ``filename=`` parameter is present (e.g.
      ``Content-Disposition: attachment`` alone).
    - Only ``filename*=...`` is present (the RFC 5987 encoded form;
      key matches ``filename*`` not ``filename``).

    Per #270, this is intentionally naive: the input domain today is
    well-behaved upstreams.  Hostile / malformed headers are sanitized
    at the boundary by the pattern-level filename resolver.
    """
    if not header:
        return None
    for raw_param in header.split(";"):
        param = raw_param.strip()
        if "=" not in param:
            continue
        key, _, value = param.partition("=")
        if key.strip().lower() != "filename":
            continue
        value = value.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value or None
    return None


class ContentValidationRejected(Exception):
    """Raised inside :func:`stream_to_tempfile` when ``validate`` rejects.

    Carries the rejection ``reason`` (a short tag like ``"content_type"``
    or ``"first_bytes"``), the observed ``content_type`` header value,
    and a bounded ``first_bytes`` excerpt.  Callers (e.g.
    :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`)
    catch this to fall back to the next URL in a period's ``urls`` list
    and aggregate the rejection context without re-emitting the URL itself.
    """

    def __init__(
        self,
        reason: str,
        *,
        content_type: str | None,
        first_bytes: bytes,
    ) -> None:
        self.reason = reason
        self.content_type = content_type
        self.first_bytes = first_bytes
        super().__init__(
            f"validate_content rejected response: reason={reason} "
            f"content_type={content_type!r} first_bytes={first_bytes!r}"
        )


# A ``validate`` callback returns ``None`` to accept the response; the caller
# either raises :class:`ContentValidationRejected` directly OR returns it for
# the helper to raise (we do the raising centrally so the call sites stay
# uniform).
ValidateFn = Callable[[str | None, bytes], "ContentValidationRejected | None"]

_DOWNLOAD_CHUNK_BYTES: int = 65_536
"""Chunk size for streaming downloads.  64 KB is a typical sweet spot
for httpx + Azure SDK throughput; larger chunks pay diminishing
returns above ~256 KB."""


def build_redacting_client(
    *,
    timeout_s: float = 30.0,
    connect_timeout_s: float = 10.0,
    retries: int = 3,
    follow_redirects: bool = True,
    user_agent: str | None = None,
) -> httpx.Client:
    """Construct an :class:`httpx.Client` whose transport logs are redacted.

    The returned client has ``event_hooks`` wired so the per-request
    and per-response log lines never include the URL query string,
    request headers, or response body / headers.

    Use as a context manager::

        with build_redacting_client() as client:
            response = client.get(url, params={"apiKey": secret})
            response.raise_for_status()

    Args:
        timeout_s: Overall request timeout in seconds.  Default 30.
        connect_timeout_s: TCP connect timeout in seconds.  Default 10.
        retries: Transport-level retries for transient failures.
            Default 3.
        follow_redirects: Whether to follow 3xx responses.  Default True
            (consistent with ``HttpUrlsPattern``).
        user_agent: Optional ``User-Agent`` header value sent on every
            request made through the client.  ``None`` or empty sends
            httpx's default UA (``python-httpx/<version>``).  Per #413
            this is deliberately a single string, not an arbitrary
            header mapping -- see SECURITY.md ("Ingest HTTP Transport
            Redaction and User-Agent") for the credential-leak
            rationale.  As a request header it never appears in the
            redacted transport logs.

    Returns:
        Configured :class:`httpx.Client`.  The caller is responsible for
        closing it (use a ``with`` block).
    """
    transport = httpx.HTTPTransport(retries=retries)
    timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)

    return httpx.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={"User-Agent": user_agent} if user_agent else None,
        event_hooks={
            "request": [_log_request_redacted],
            "response": [_log_response_redacted],
        },
    )


def _log_request_redacted(request: httpx.Request) -> None:
    """Log only method + host + path; never query string or headers."""
    logger.info(
        "ingest http request: method=%s host=%s path=%s",
        request.method,
        request.url.host,
        request.url.path,
    )


def _log_response_redacted(response: httpx.Response) -> None:
    """Log only status code + method + host + path; never body or headers."""
    request = response.request
    logger.info(
        "ingest http response: method=%s host=%s path=%s status=%d",
        request.method,
        request.url.host,
        request.url.path,
        response.status_code,
    )


@contextmanager
def stream_to_tempfile(
    client: httpx.Client,
    url: str,
    *,
    chunk_size: int = _DOWNLOAD_CHUNK_BYTES,
    validate: ValidateFn | None = None,
) -> Iterator[DownloadedPayload]:
    """Stream a GET response body to a temp file; yield path + filename hint.

    The body is written chunk-by-chunk so peak memory is bounded by
    ``chunk_size`` rather than the response size.  This is load-bearing
    for the UMLS Metathesaurus (5+ GB) -- ``response.content`` would OOM.

    Per #270 the yield type is :class:`DownloadedPayload`.  In addition
    to the tempfile path, the helper captures the response's
    ``Content-Disposition`` header value (a bounded constant-size string,
    not a payload buffer -- the I/O-at-Boundaries memory invariant
    holds) and parses the ``filename=`` parameter so callers can thread
    the server-supplied name into the non-archive payload filename
    precedence chain.

    Cleanup: the temp file is unlinked when the context exits.  The
    extractor consumes the file under the same context so the bytes
    are still on disk when extraction runs.

    The redacting transport hooks fire as usual (request + response
    audit lines), so the URL's query string / api_key never appears
    in transport-layer logs.

    When ``validate`` is set (per #228), the helper:

    1. Opens the stream and calls ``raise_for_status()``.
    2. Pulls the first chunk.
    3. Calls ``validate(content_type, first_chunk)`` -- the caller's
       predicate inspects the Content-Type header and the first bytes.
    4. If ``validate`` returns a :class:`ContentValidationRejected`,
       closes the stream and raises it.  No bytes are written to disk.
    5. Otherwise, writes the first chunk + remaining stream to the
       temp file and yields the :class:`DownloadedPayload`.

    Usage::

        with build_redacting_client() as client, stream_to_tempfile(
            client, url
        ) as payload:
            for filename, data in extract_and_filter_iter(payload.path, ...):
                ...

    Args:
        client: An :class:`httpx.Client` (typically from
            :func:`build_redacting_client`).
        url: Target URL.  May embed an api_key in the query string;
            the URL itself is never logged via the redacted hooks.
        chunk_size: Bytes per ``iter_bytes`` chunk.  Default 64 KB.
        validate: Optional predicate evaluated against the response's
            ``Content-Type`` header and the first chunk of body bytes.
            Returning a :class:`ContentValidationRejected` instance
            (rather than raising) lets the helper raise centrally so
            cleanup happens uniformly.  Returning ``None`` accepts the
            response.  Default ``None`` -- predicate skipped, behavior
            identical to the v0.27 helper.

    Yields:
        :class:`DownloadedPayload` containing the tempfile path plus the
        naive-parsed ``Content-Disposition`` filename hint (or
        :data:`None` when the header is absent).

    Raises:
        httpx.HTTPStatusError: On a 4xx / 5xx response.
        ContentValidationRejected: When ``validate`` rejects the
            response.  The temp file has not been written to disk.
    """
    # ``delete=False`` because we need to close the file before yielding
    # the path -- on Windows, opening the path while the NamedTemporaryFile
    # holds it would fail.  We unlink in the finally below.
    handle = tempfile.NamedTemporaryFile(suffix=".download", delete=False)  # noqa: SIM115
    path = Path(handle.name)
    content_disposition: str | None = None
    try:
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_disposition = response.headers.get("content-disposition")
                chunks = response.iter_bytes(chunk_size=chunk_size)
                if validate is not None:
                    first_chunk = next(chunks, b"")
                    content_type = response.headers.get("content-type")
                    rejection = validate(content_type, first_chunk)
                    if rejection is not None:
                        raise rejection
                    handle.write(first_chunk)
                for chunk in chunks:
                    handle.write(chunk)
        finally:
            handle.close()
        yield DownloadedPayload(
            path=path,
            content_disposition_filename=_parse_content_disposition_filename(content_disposition),
        )
    finally:
        path.unlink(missing_ok=True)
