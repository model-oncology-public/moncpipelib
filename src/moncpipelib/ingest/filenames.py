"""Filename sanitization for the ingest landing boundary.

Public helper used by :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`
and :class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern` to
normalize URL-derived basenames and ``Content-Disposition``-supplied filenames
into safe blob names BEFORE they reach the upload layer.

Sanitization rules (in order):

1. Percent-escapes are URL-decoded (``%20`` -> space, ``%2F`` -> ``/`` --
   the latter is then caught by the separator strip).  ``+`` is NOT
   decoded as space; we use :func:`urllib.parse.unquote`, not
   ``unquote_plus``.
2. Path separators (``/``, ``\\``) are stripped.  This blocks any
   path-traversal attempt (``../foo`` -> ``..foo``) that survives URL
   decoding.
3. Filesystem-unsafe characters (``:*?"<>|``) are stripped.
4. ASCII control characters (0x00-0x1F and 0x7F) are stripped.  Blocks
   header-injection-style sequences from a hostile
   ``Content-Disposition`` server response.
5. Whitespace runs (any Unicode whitespace, including TAB / NL / CR) are
   collapsed to a single ``_``.  The visual filename remains readable but
   blob paths stay parseable from logs.
6. An empty-after-sanitize result returns :data:`None`, so callers can
   fall through to the next precedence level cleanly.

Per #270, this helper is intentionally local rather than depending on
``pathvalidate`` or ``python-slugify``: HIPAA / SOC 2 compliance puts a
real cost on every transitive dependency, the input domain is narrow
(URL basenames bounded by RFC 3986; ``Content-Disposition`` strings
sanitized at the boundary; not arbitrary user uploads), and the helper
fits in ~30 lines.

Trust boundary: this helper sanitizes UNTRUSTED inputs (URL basenames
from the upstream URL, ``Content-Disposition`` server-supplied strings).
Authored inputs (``payload_filename_template`` from the contract,
resolver-supplied ``ResolvedDownload.filename`` hints from in-repo
resolver code) are NOT passed through this helper -- a malformed
authored name should fail loudly at upload time rather than be silently
rewritten.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

_PATH_SEPARATORS: str = "/\\"
"""POSIX and Windows path separators stripped before upload."""

_FILESYSTEM_UNSAFE: str = ':*?"<>|'
"""Characters disallowed across blob backends + Windows; stripped."""

_CONTROL_CHARS: frozenset[int] = frozenset({*range(0x00, 0x20), 0x7F})
"""ASCII control characters; stripped to block header-injection vectors."""

_WHITESPACE_RUN_RE: re.Pattern[str] = re.compile(r"\s+")
"""Matches one-or-more whitespace characters (any Unicode class).
Collapsed to a single ``_`` so the visual name stays readable."""


def sanitize_blob_filename(name: str) -> str | None:
    """Sanitize an untrusted filename for blob storage.

    See module docstring for the full rule list and trust boundary.

    Args:
        name: The raw filename (URL basename or ``Content-Disposition``
            header value).  May be empty.

    Returns:
        The sanitized filename, or :data:`None` when the result is empty
        after sanitization (so callers fall through to the next
        precedence level cleanly).
    """
    if not name:
        return None
    decoded = unquote(name)
    cleaned = "".join(
        ch
        for ch in decoded
        if ch not in _PATH_SEPARATORS
        and ch not in _FILESYSTEM_UNSAFE
        and ord(ch) not in _CONTROL_CHARS
    )
    collapsed = _WHITESPACE_RUN_RE.sub("_", cleaned)
    return collapsed or None


__all__ = ["sanitize_blob_filename"]
