"""Tests for the shared hash-while-writing tempfile helper (#415 Phase 1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from moncpipelib.ingest._hashing import hashing_tempfile


class TestHashingTempfile:
    def test_multi_chunk_write_matches_hashlib_reference(self) -> None:
        chunks = [b"alpha", b"", b"beta" * 1000, b"\x00\xff binary"]
        reference = hashlib.sha256(b"".join(chunks)).hexdigest()

        with hashing_tempfile(suffix=".ndjson") as writer:
            for chunk in chunks:
                assert writer.write(chunk) == len(chunk)
            writer.close()

            assert writer.sha256_hexdigest() == reference
            assert writer.size_bytes == sum(len(c) for c in chunks)
            assert writer.path.read_bytes() == b"".join(chunks)

    def test_empty_write_stream(self) -> None:
        with hashing_tempfile(suffix=".empty") as writer:
            writer.close()
            assert writer.sha256_hexdigest() == hashlib.sha256(b"").hexdigest()
            assert writer.size_bytes == 0
            assert writer.path.read_bytes() == b""

    def test_close_is_idempotent(self) -> None:
        with hashing_tempfile(suffix=".x") as writer:
            writer.write(b"data")
            writer.close()
            writer.close()  # second close must not raise

    def test_path_readable_after_close_inside_context(self) -> None:
        with hashing_tempfile(suffix=".x") as writer:
            writer.write(b"payload")
            writer.close()
            with writer.path.open("rb") as fp:
                assert fp.read() == b"payload"

    def test_tempfile_unlinked_after_context_exit(self) -> None:
        path: Path
        with hashing_tempfile(suffix=".x") as writer:
            writer.write(b"payload")
            writer.close()
            path = writer.path
            assert path.exists()
        assert not path.exists()

    def test_tempfile_unlinked_on_exception_mid_write(self) -> None:
        path: Path | None = None
        with pytest.raises(RuntimeError, match="boom"), hashing_tempfile(suffix=".x") as writer:
            path = writer.path
            writer.write(b"partial")
            raise RuntimeError("boom")
        assert path is not None
        assert not path.exists()

    def test_suffix_applied(self) -> None:
        with hashing_tempfile(suffix=".manifest.json") as writer:
            assert writer.path.name.endswith(".manifest.json")
