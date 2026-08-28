"""Tests for ``extract_and_filter_iter``: nested archives + extract_filter.

ADR-1 reference:
``docs/migrations/20260426_phase2-ingest-decisions.md``.

Covers:

- Phase 1 single-zip cases (regression).
- Nested zip-of-zip extraction.
- ``extract_filter`` applied to terminal members at every level.
- Archives matching ``extract`` are recursed regardless of the filter
  (the UMLS case: outer archive name does not match ``meta/**``).
- ``*`` matches ``/`` under fnmatch -- ``meta/*`` and ``meta/**`` give
  identical match sets.
- Filter is applied AFTER ``strip_extensions``.
- Unsupported extract format raises a clear ValueError.
- Streaming source: passing a :class:`pathlib.Path` extracts without
  loading the archive into memory.
- #239 streaming surface: yielded sha256 + size_bytes match the
  extracted bytes; the tempfile is unlinked once the iterator advances.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from moncpipelib.ingest.patterns._extract import (
    ExtractSource,
    extract_and_filter_iter,
)


def _zip_of(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _run(
    source: ExtractSource,
    extract: Sequence[str] = (),
    strip_extensions: Sequence[str] = (),
    extract_filter: Sequence[str] = (),
) -> dict[str, bytes]:
    """Drain the iterator into a dict for ergonomic equality assertions.

    Reads each yielded tempfile path back into bytes for comparison.
    The tempfile is unlinked when the iterator advances past the yield;
    we accumulate the bytes before that happens.
    """
    return {
        name: path.read_bytes()
        for name, path, _sha, _size in extract_and_filter_iter(
            source, extract, strip_extensions, extract_filter
        )
    }


# ---------------------------------------------------------------------------
# Phase 1 regression cases
# ---------------------------------------------------------------------------


def test_no_extract_returns_payload_as_terminal() -> None:
    payload = b"raw-bytes"
    out = _run(payload, extract=())
    assert out == {"__payload__": payload}


def test_single_zip_extracts_all_members() -> None:
    payload = _zip_of({"a.csv": b"a", "sub/b.csv": b"b"})
    out = _run(payload, extract=("zip",))
    assert out == {"a.csv": b"a", "sub/b.csv": b"b"}


def test_strip_extensions_applied_to_terminal_names() -> None:
    payload = _zip_of({"foo.csv.xls": b"data"})
    out = _run(payload, extract=("zip",), strip_extensions=(".xls",))
    assert out == {"foo.csv": b"data"}


def test_directory_entries_skipped() -> None:
    """Zip directory entries (names ending with `/`) must not be emitted."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dir/", b"")
        zf.writestr("dir/foo.txt", b"x")
    out = _run(buf.getvalue(), extract=("zip",))
    assert out == {"dir/foo.txt": b"x"}


def test_unsupported_format_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported extract format 'tar'"):
        list(extract_and_filter_iter(b"", extract=("tar",), strip_extensions=()))


# ---------------------------------------------------------------------------
# Nested archive cases (Phase 2)
# ---------------------------------------------------------------------------


def _umls_shaped_payload() -> bytes:
    """Outer zip containing one inner zip with mixed meta/ + otherks/ files.

    Mirrors the UMLS Metathesaurus shape from ADR-1's worked example.
    """
    inner = _zip_of(
        {
            "meta/MRCONSO.RRF": b"conso",
            "meta/MRDEF.RRF": b"def",
            "otherks/SKIP.RRF": b"skip",
        }
    )
    return _zip_of({"2026AB-meta.nlm.zip": inner})


def test_nested_zip_extracts_all_inner_members_when_no_filter() -> None:
    out = _run(_umls_shaped_payload(), extract=("zip", "zip"))
    assert out == {
        "meta/MRCONSO.RRF": b"conso",
        "meta/MRDEF.RRF": b"def",
        "otherks/SKIP.RRF": b"skip",
    }


def test_nested_zip_filter_keeps_only_matching_members() -> None:
    """ADR-1 worked example: ``meta/**`` keeps the meta tree only."""
    out = _run(
        _umls_shaped_payload(),
        extract=("zip", "zip"),
        extract_filter=("meta/**",),
    )
    assert sorted(out) == ["meta/MRCONSO.RRF", "meta/MRDEF.RRF"]


def test_outer_archive_member_recursed_even_when_it_does_not_match_filter() -> None:
    """Per ADR-1: archives matching ``extract`` extensions are recursed
    regardless of ``extract_filter``.  The UMLS outer member
    ``2026AB-meta.nlm.zip`` does NOT match the filter ``meta/**``, but
    the implementation must still recurse into it before applying the
    filter to its contents -- otherwise the filter at the outer level
    would prune the only archive carrying anything we want."""
    out = _run(
        _umls_shaped_payload(),
        extract=("zip", "zip"),
        extract_filter=("meta/**",),
    )
    assert out, (
        "Filter applied at the outer level would prune everything; "
        "implementation regressed the always-recurse rule from ADR-1."
    )


def test_filter_applied_at_every_level() -> None:
    """A filter passed at the top level applies recursively at each
    archive level it descends through."""
    inner_a = _zip_of({"meta/x.RRF": b"x", "skip.RRF": b"s"})
    outer = _zip_of({"a.zip": inner_a})

    out = _run(outer, extract=("zip", "zip"), extract_filter=("meta/**",))
    assert sorted(out) == ["meta/x.RRF"]


def test_double_star_and_single_star_are_equivalent_under_fnmatch() -> None:
    """ADR-1: ``*`` matches ``/`` under fnmatch, so ``meta/*`` and
    ``meta/**`` produce identical match sets.  This test pins that
    behavior so any future switch to path-aware globbing is a
    deliberate, reviewed change."""
    payload = _umls_shaped_payload()
    star = _run(payload, extract=("zip", "zip"), extract_filter=("meta/*",))
    double_star = _run(payload, extract=("zip", "zip"), extract_filter=("meta/**",))
    assert star == double_star


# ---------------------------------------------------------------------------
# Filter / strip ordering
# ---------------------------------------------------------------------------


def test_filter_sees_post_strip_path() -> None:
    """ADR-1 test surface: the filter sees the post-``strip_extensions``
    path -- contract authors write globs against the same shape
    downstream consumers see."""
    payload = _zip_of({"foo.csv.xls": b"a", "bar.csv.xls": b"b"})
    out = _run(
        payload,
        extract=("zip",),
        strip_extensions=(".xls",),
        extract_filter=("foo.csv",),  # post-strip path
    )
    assert out == {"foo.csv": b"a"}


def test_filter_against_pre_strip_path_does_not_match() -> None:
    """Sister assertion: a filter targeting the pre-strip extension does
    NOT match.  Locks the post-strip ordering down explicitly."""
    payload = _zip_of({"foo.csv.xls": b"a"})
    out = _run(
        payload,
        extract=("zip",),
        strip_extensions=(".xls",),
        extract_filter=("*.xls",),  # would match pre-strip path only
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Single-zip with extract_filter (no nesting)
# ---------------------------------------------------------------------------


def test_single_zip_with_filter_keeps_only_matches() -> None:
    payload = _zip_of(
        {
            "data/file1.csv": b"1",
            "data/file2.csv": b"2",
            "metadata/about.txt": b"about",
        }
    )
    out = _run(payload, extract=("zip",), extract_filter=("data/**",))
    assert sorted(out) == ["data/file1.csv", "data/file2.csv"]


def test_no_filter_at_inner_level_recurses_normally() -> None:
    """Default (empty filter) behavior matches Phase 1: extract everything."""
    out = _run(_umls_shaped_payload(), extract=("zip", "zip"))
    assert "otherks/SKIP.RRF" in out  # no filter -> kept


# ---------------------------------------------------------------------------
# Streaming sources (file path)
# ---------------------------------------------------------------------------


def test_extracts_from_file_path_source(tmp_path: Path) -> None:
    """Passing a :class:`pathlib.Path` streams the archive from disk
    without copying it into memory first.  Load-bearing for UMLS
    Metathesaurus where the archive is 5+ GB."""
    archive_path = tmp_path / "outer.zip"
    archive_path.write_bytes(_umls_shaped_payload())

    out = _run(archive_path, extract=("zip", "zip"), extract_filter=("meta/**",))
    assert sorted(out) == ["meta/MRCONSO.RRF", "meta/MRDEF.RRF"]


def test_iterator_yields_one_file_at_a_time() -> None:
    """The iterator semantics mean callers never hold the full file set
    in memory simultaneously.  Sanity-check: iterate and count without
    accumulating into a dict."""
    payload = _zip_of({"a": b"x", "b": b"y", "c": b"z"})
    seen = 0
    for _name, _path, _sha, _size in extract_and_filter_iter(payload, ("zip",), ()):
        seen += 1
    assert seen == 3


# ---------------------------------------------------------------------------
# #239 streaming surface
# ---------------------------------------------------------------------------


def test_yielded_sha256_and_size_match_member_bytes() -> None:
    """The extractor's streaming write computes sha256 + size in one pass;
    callers rely on those values for the hash-compare upload (#239).

    The yielded path is unlinked when the iterator advances, so the
    bytes must be read inside the for-loop body -- not after.
    """
    payload = _zip_of({"alpha.csv": b"A,B\n1,2\n", "beta.csv": b"x"})
    seen: list[str] = []
    for name, path, sha, size in extract_and_filter_iter(payload, ("zip",), ()):
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == sha, name
        assert size == len(body), name
        seen.append(name)
    assert sorted(seen) == ["alpha.csv", "beta.csv"]


def test_yielded_path_is_unlinked_after_iterator_advances() -> None:
    """Tempfile lifetime is bound to the iteration step that yielded it;
    the next ``next(...)`` (or generator close) must unlink the path.
    This pins the no-leak invariant promised by the module docstring."""
    payload = _zip_of({"a.csv": b"a", "b.csv": b"b"})
    iterator = extract_and_filter_iter(payload, ("zip",), ())

    _name1, path1, _sha1, _size1 = next(iterator)
    assert path1.exists()
    # Advance: previous tempfile must be cleaned up before the next yield.
    _name2, path2, _sha2, _size2 = next(iterator)
    assert not path1.exists(), "previous tempfile leaked across iterator advance"
    assert path2.exists()
    # Drain the rest; the final tempfile is cleaned up on generator close.
    list(iterator)
    assert not path2.exists(), "final tempfile leaked after generator close"


def test_consumer_exception_does_not_leak_tempfile() -> None:
    """If a consumer raises mid-iteration, the generator's finally must
    still unlink the in-flight tempfile.  Backstops the cleanup path
    that the streaming-memory acceptance test cannot easily exercise."""
    payload = _zip_of({"a.csv": b"a", "b.csv": b"b"})
    iterator = extract_and_filter_iter(payload, ("zip",), ())

    _name, path, _sha, _size = next(iterator)
    assert path.exists()

    iterator.close()  # simulates a consumer exception breaking the for-loop
    assert not path.exists(), "tempfile leaked after generator close"
