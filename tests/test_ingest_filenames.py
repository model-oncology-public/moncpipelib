"""Tests for ``sanitize_blob_filename``.

Covers the full rule matrix called out in the
``moncpipelib.ingest.filenames`` module docstring + #270 acceptance
criteria:

- Percent-escape decoding (``%20`` -> space, then collapsed to ``_``).
- Path-separator stripping (``/``, ``\\``) -- includes the path-traversal
  defence for ``%2F`` decoded forms.
- Filesystem-unsafe character stripping (``:*?"<>|``).
- ASCII control character stripping (0x00-0x1F + 0x7F).
- Whitespace-run collapsing to a single ``_``.
- Empty / empty-after-sanitize -> :data:`None`.
- Names that pass through unchanged (regression guard for the common
  case: URL basenames already in safe form).
"""

from __future__ import annotations

import pytest

from moncpipelib.ingest import sanitize_blob_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Pass-through cases: the common URL basename should be unchanged.
        ("V2024B_V2025B_V2026A_CPC_SMVL.csv", "V2024B_V2025B_V2026A_CPC_SMVL.csv"),
        ("data.json", "data.json"),
        ("archive.tar.gz", "archive.tar.gz"),
        ("File-Name_v2.txt", "File-Name_v2.txt"),
        # Percent-escape decoding: %20 -> space -> collapsed to _.
        ("test%20file.csv", "test_file.csv"),
        ("a%20b%20c.txt", "a_b_c.txt"),
        # %2F decodes to '/' which is then stripped by the separator pass.
        ("path%2Fto%2Ffile.csv", "pathtofile.csv"),
        # Path separators stripped (Windows + POSIX).
        ("a/b/c.txt", "abc.txt"),
        ("a\\b\\c.txt", "abc.txt"),
        ("mixed/style\\path.csv", "mixedstylepath.csv"),
        # Path traversal defence: dots survive but separators don't, so
        # the result cannot escape the upload prefix.
        ("../../etc/passwd", "....etcpasswd"),
        # Filesystem-unsafe characters stripped.
        ("a:b*c?.txt", "abc.txt"),
        ('quote"inside<>.txt', "quoteinside.txt"),
        ("pipe|name.txt", "pipename.txt"),
        # ASCII control characters stripped (NULL, BEL, ESC, DEL).
        ("a\x00b.csv", "ab.csv"),
        ("alert\x07.txt", "alert.txt"),
        ("escape\x1bseq.txt", "escapeseq.txt"),
        ("delete\x7f.txt", "delete.txt"),
        # Whitespace runs collapsed to a single _.  Only space (0x20)
        # survives the control-char strip; TAB / LF / CR (0x09 / 0x0A /
        # 0x0D) are in the 0x00-0x1F range and are stripped as control
        # characters, so they never reach the whitespace-collapse pass.
        ("hello world.txt", "hello_world.txt"),
        ("hello   world.txt", "hello_world.txt"),
        ("tab\tand\nnewline.txt", "tabandnewline.txt"),
        ("trailing space .csv", "trailing_space_.csv"),
        # Mixed: percent-escape decode + control + unsafe + whitespace.
        ('odd %20"file*\x00.csv', "odd_file.csv"),
    ],
)
def test_sanitize_blob_filename_matrix(raw: str, expected: str) -> None:
    """The rule matrix above is the spec; each row is one decision."""
    assert sanitize_blob_filename(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # Empty string up front.
        "",
        # All-stripped: only path separators.
        "///",
        "\\\\",
        # All-stripped: only filesystem-unsafe.
        ":*?",
        '"<>|',
        # All-stripped: only control characters.
        "\x00\x01\x02",
        # All-stripped: percent-encoded separators.
        "%2F%5C",
        # Percent-encoded control chars.
        "%00%01",
    ],
)
def test_sanitize_blob_filename_returns_none_when_empty(raw: str) -> None:
    """Empty-after-sanitize returns ``None`` so callers fall through."""
    assert sanitize_blob_filename(raw) is None


def test_sanitize_blob_filename_does_not_decode_plus() -> None:
    """``+`` is NOT decoded as space (we use ``unquote``, not ``unquote_plus``).

    Form-encoded URLs use ``+`` for space, but RFC 3986 path components
    treat ``+`` literally.  Filenames in path components rarely use
    ``+`` for space, and decoding it would surprise authors.
    """
    assert sanitize_blob_filename("a+b.csv") == "a+b.csv"


def test_sanitize_blob_filename_whitespace_only_collapses_to_underscore() -> None:
    """Whitespace-only input collapses to ``"_"``, not :data:`None`.

    Per the rule list: whitespace runs collapse BEFORE the empty check.
    A literal ``"_"`` is non-empty, so it is returned.  The precedence
    chain higher up (``payload_filename_template`` -> resolver hint ->
    Content-Disposition -> URL basename) does not invoke the sanitizer
    on authored inputs, so a single ``"_"`` here only happens for
    pathological URL basenames or server-supplied headers, where it is
    no worse than the legacy ``__payload__`` name.
    """
    assert sanitize_blob_filename("   ") == "_"
