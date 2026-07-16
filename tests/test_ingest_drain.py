"""Tests for :func:`moncpipelib.ingest.drain_to_bytes` (#354).

Pins the bounded-drain idiom that ingest consumers need: the production
blob reader (:class:`_ChunkedBlobReader`) refuses unbounded ``read()``,
so consumers that want a small reference file in memory must loop
``read(n)``.  ``drain_to_bytes`` is that loop, with a ``max_bytes``
safety ceiling.  The key regression these tests guard is that the helper
works against the *real* reader -- not just an ``io.BytesIO`` stand-in
that transparently supports unbounded reads.
"""

from __future__ import annotations

import io

import pytest

from moncpipelib.ingest import StreamTooLargeError, drain_to_bytes
from moncpipelib.resources.blob import _ChunkedBlobReader


def _real_reader(chunks: list[bytes]) -> _ChunkedBlobReader:
    """A real ``_ChunkedBlobReader`` over a fake SDK downloader.

    The reader serves at most one chunk per ``read`` call and raises on
    unbounded ``read()`` -- exactly the production contract that broke
    the old cookbook example.
    """

    class _FakeDownloader:
        def __init__(self, chunk_list: list[bytes]) -> None:
            self._chunk_list = chunk_list

        def chunks(self) -> object:
            return iter(self._chunk_list)

    return _ChunkedBlobReader(_FakeDownloader(chunks))


def test_drain_concatenates_full_payload() -> None:
    assert drain_to_bytes(io.BytesIO(b"hello world")) == b"hello world"


def test_drain_empty_stream_returns_empty_bytes() -> None:
    assert drain_to_bytes(io.BytesIO(b"")) == b""


def test_drain_against_real_chunked_reader() -> None:
    """The whole point of #354: the helper must drain the production
    reader, which serves one chunk per ``read`` and refuses ``read()``.
    """
    reader = _real_reader([b"id,value\n", b"1,42\n", b"2,7\n"])
    assert drain_to_bytes(reader, chunk_size=4) == b"id,value\n1,42\n2,7\n"


def test_drain_handles_short_reads_smaller_than_chunk_size() -> None:
    """Each SDK chunk is shorter than ``chunk_size``; the loop must keep
    going on short (non-empty) reads until EOF rather than stopping
    early."""
    reader = _real_reader([b"ab", b"cd", b"ef"])
    assert drain_to_bytes(reader, chunk_size=64) == b"abcdef"


def test_drain_at_exactly_max_bytes_passes() -> None:
    payload = b"x" * 100
    assert drain_to_bytes(io.BytesIO(payload), max_bytes=100) == payload


def test_drain_over_max_bytes_raises_stream_too_large() -> None:
    with pytest.raises(StreamTooLargeError, match="exceeded max_bytes=100"):
        drain_to_bytes(io.BytesIO(b"x" * 101), max_bytes=100)


def test_stream_too_large_is_value_error_subclass() -> None:
    """Existing ``except ValueError`` handlers keep catching the ceiling
    breach; callers can still catch the narrower type."""
    assert issubclass(StreamTooLargeError, ValueError)
    with pytest.raises(ValueError, match="refusing to materialize"):
        drain_to_bytes(io.BytesIO(b"too big"), max_bytes=1)


def test_drain_ceiling_enforced_against_real_reader() -> None:
    """The ceiling must trip mid-drain on the chunked reader too -- not
    only on a fully-buffered BytesIO."""
    reader = _real_reader([b"a" * 64, b"b" * 64, b"c" * 64])
    with pytest.raises(StreamTooLargeError):
        drain_to_bytes(reader, max_bytes=100, chunk_size=64)


@pytest.mark.parametrize("bad", [0, -1])
def test_drain_rejects_nonpositive_max_bytes(bad: int) -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        drain_to_bytes(io.BytesIO(b"data"), max_bytes=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_drain_rejects_nonpositive_chunk_size(bad: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        drain_to_bytes(io.BytesIO(b"data"), chunk_size=bad)


def test_drain_does_not_close_stream() -> None:
    """The helper consumes but does not own the stream -- the context
    manager that yielded it (``read_partition_with_manifest``) closes
    it."""
    stream = io.BytesIO(b"payload")
    drain_to_bytes(stream)
    assert not stream.closed
