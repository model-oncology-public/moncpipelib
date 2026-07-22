"""Tests for ``build_redacting_client`` and ``stream_to_tempfile``.

Covers the audit-boundary contract: the redacting transport hooks
must never log URL query strings, request headers, or response
bodies/headers.  api_key (which lives inside the URL query for the
UTS download endpoint) cannot leak through this client even on
error responses.

Also covers the ``Content-Disposition`` filename capture added in #270:
``stream_to_tempfile`` surfaces the parsed ``filename=`` parameter
alongside the tempfile path so the non-archive payload precedence
chain can use the server-supplied name.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import respx

from moncpipelib.ingest._http import (
    _parse_content_disposition_filename,
    build_redacting_client,
    stream_to_tempfile,
)


def test_request_log_omits_query_string(caplog: Any) -> None:
    """The request hook logs host + path, never the query string.

    A URL with ``apiKey=super-secret-value`` must NOT produce a log
    record containing that value.
    """
    caplog.set_level(logging.INFO, logger="moncpipelib.ingest._http")
    with respx.mock:
        respx.get("https://uts-ws.nlm.nih.gov/download").respond(200, content=b"x")
        with build_redacting_client() as client:
            response = client.get(
                "https://uts-ws.nlm.nih.gov/download",
                params={"url": "https://example/foo", "apiKey": "super-secret-value"},
            )
            assert response.status_code == 200

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "super-secret-value" not in captured
    assert "uts-ws.nlm.nih.gov" in captured  # host IS allowed in audit log
    assert "/download" in captured  # path IS allowed in audit log
    assert "?" not in captured  # query string is not


def test_response_log_omits_body(caplog: Any) -> None:
    """The response hook logs status + path; never the response body."""
    caplog.set_level(logging.INFO, logger="moncpipelib.ingest._http")
    with respx.mock:
        respx.get("https://example.test/foo").respond(
            200, content=b'{"sensitive": "leaked-body-content"}'
        )
        with build_redacting_client() as client:
            client.get("https://example.test/foo")

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "leaked-body-content" not in captured
    assert "status=200" in captured


def test_request_log_omits_authorization_header(caplog: Any) -> None:
    """Headers are never logged -- an Authorization Bearer token cannot
    appear in the audit log even when the caller sets it explicitly."""
    caplog.set_level(logging.INFO, logger="moncpipelib.ingest._http")
    with respx.mock:
        respx.get("https://example.test/x").respond(200)
        with build_redacting_client() as client:
            client.get(
                "https://example.test/x",
                headers={"Authorization": "Bearer leaked-token-content"},
            )

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "leaked-token-content" not in captured
    assert "Authorization" not in captured


def test_user_agent_sent_when_provided(caplog: Any) -> None:
    """Per #413 a caller-supplied ``user_agent`` is set client-level so
    every request carries it (upstreams with script-UA abuse detection,
    e.g. FDA accessdata, need a descriptive organizational UA).  Being a
    request header, it must also stay out of the redacted audit log."""
    caplog.set_level(logging.INFO, logger="moncpipelib.ingest._http")
    ua = "ExampleOrgDataPlatform/1.0 (contact: data@example.org)"
    with respx.mock:
        route = respx.get("https://example.test/data.csv").respond(200, content=b"x")
        with build_redacting_client(user_agent=ua) as client:
            client.get("https://example.test/data.csv")

    assert route.calls.last.request.headers["User-Agent"] == ua
    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "ExampleOrgDataPlatform" not in captured


def test_default_user_agent_when_omitted() -> None:
    """Omitting ``user_agent`` preserves httpx's default UA (#413)."""
    with respx.mock:
        route = respx.get("https://example.test/data.csv").respond(200, content=b"x")
        with build_redacting_client() as client:
            client.get("https://example.test/data.csv")

    assert route.calls.last.request.headers["User-Agent"].startswith("python-httpx/")


def test_empty_user_agent_falls_back_to_default() -> None:
    """``user_agent=""`` must not put an empty UA header on the wire.

    The loader rejects empty strings for contract-declared UAs, but the
    factory is a public helper -- a direct caller passing "" gets the
    httpx default instead of ``User-Agent: <blank>`` (which many servers
    reject with 400/403)."""
    with respx.mock:
        route = respx.get("https://example.test/data.csv").respond(200, content=b"x")
        with build_redacting_client(user_agent="") as client:
            client.get("https://example.test/data.csv")

    assert route.calls.last.request.headers["User-Agent"].startswith("python-httpx/")


def test_redaction_holds_on_error_responses(caplog: Any) -> None:
    """A 401 response must redact just the same as a 200.  Most leaks
    happen on the error path because that's where developers reach for
    the URL in their error formatting."""
    caplog.set_level(logging.INFO, logger="moncpipelib.ingest._http")
    with respx.mock:
        respx.get("https://example.test/auth").respond(401, content=b"unauthorized")
        with build_redacting_client() as client:
            response = client.get(
                "https://example.test/auth",
                params={"apiKey": "super-secret-on-error"},
            )
            assert response.status_code == 401

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "super-secret-on-error" not in captured
    assert "status=401" in captured


# ---------------------------------------------------------------------------
# _parse_content_disposition_filename (#270)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        # Quoted filename, the canonical RFC 6266 shape.
        ('attachment; filename="foo.csv"', "foo.csv"),
        # Unquoted filename (RFC 6266 token form).
        ("attachment; filename=foo.csv", "foo.csv"),
        # Whitespace tolerance around the parameter.
        ('attachment;  filename = "foo.csv" ', "foo.csv"),
        # Case-insensitive parameter name.
        ('attachment; FILENAME="foo.csv"', "foo.csv"),
        # Parameter ordering shouldn't matter; charset before filename.
        ('attachment; charset=utf-8; filename="foo.csv"', "foo.csv"),
        # filename* (RFC 5987 encoded) on its own is intentionally
        # ignored -- naive parser, follow-on per #270.  Falls through
        # to the next precedence level.
        ("attachment; filename*=UTF-8''utf8-name.csv", None),
        # filename* before filename: the naive parser SHOULD pick the
        # plain filename and NOT confuse the *= form for an = form.
        (
            "attachment; filename*=UTF-8''utf8.csv; filename=\"ascii.csv\"",
            "ascii.csv",
        ),
        # No filename parameter at all.
        ("attachment", None),
        ("inline", None),
        # Empty value collapses to None so callers fall through.
        ('attachment; filename=""', None),
        ("attachment; filename=", None),
        # Header missing entirely.
        (None, None),
        ("", None),
    ],
)
def test_parse_content_disposition_filename(header: str | None, expected: str | None) -> None:
    """Per-header expectation for the naive parser.  See module docstring
    for what is in / out of scope."""
    assert _parse_content_disposition_filename(header) == expected


# ---------------------------------------------------------------------------
# stream_to_tempfile yields DownloadedPayload (#270)
# ---------------------------------------------------------------------------


def test_stream_to_tempfile_captures_content_disposition_filename() -> None:
    """When the response carries Content-Disposition: filename=...,
    the yielded DownloadedPayload surfaces the parsed name."""
    with respx.mock:
        respx.get("https://example.test/download").respond(
            200,
            content=b"hello",
            headers={"Content-Disposition": 'attachment; filename="data.csv"'},
        )
        with (
            build_redacting_client() as client,
            stream_to_tempfile(client, "https://example.test/download") as payload,
        ):
            assert payload.path.read_bytes() == b"hello"
            assert payload.content_disposition_filename == "data.csv"


def test_stream_to_tempfile_no_content_disposition_yields_none_filename() -> None:
    """When the upstream omits Content-Disposition, the yielded payload
    has filename=None so the precedence chain falls through cleanly."""
    with respx.mock:
        respx.get("https://example.test/download").respond(200, content=b"body")
        with (
            build_redacting_client() as client,
            stream_to_tempfile(client, "https://example.test/download") as payload,
        ):
            assert payload.path.read_bytes() == b"body"
            assert payload.content_disposition_filename is None
