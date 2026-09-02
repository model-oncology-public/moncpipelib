"""Streaming-memory acceptance test for blob_mirror (#437).

Mirrors a synthetic 64 MiB foreign object through the full
``BlobMirrorPattern`` flow (source stream -> hashing tempfile -> upload)
and asserts peak Python heap stays well under the object size, proving
the mirror never materializes the whole object.

Both boundaries are exercised in streaming form:

- the foreign source yields the object in bounded ``read(n)`` chunks
  (never a full ``bytes`` payload), and
- the destination blob drains the upload handle in bounded reads
  (mirroring the real Azure SDK, which streams the file handle in
  chunks) rather than ``.read()``-ing it whole.

If either the copy loop or the upload buffered the full object, peak
would exceed the 32 MiB threshold for a 64 MiB object.
"""

from __future__ import annotations

import logging
import tracemalloc
from types import SimpleNamespace
from typing import IO, Literal

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.patterns.blob_mirror import BlobMirrorPattern
from moncpipelib.ingest.types import IngestContext, PartitionSpec

_OBJECT_SIZE_BYTES = 64 * 1024 * 1024
_PEAK_THRESHOLD_BYTES = 32 * 1024 * 1024
_GEN_CHUNK = 1024 * 1024  # 1 MiB


class _GeneratedReader:
    """Forward-only reader that fabricates ``size`` bytes without buffering.

    Serves ``b"a"`` bytes on demand so the *source* side never holds the
    full object in memory (which would defeat the baseline).
    """

    def __init__(self, size: int) -> None:
        self._remaining = size

    def __enter__(self) -> _GeneratedReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self._remaining = 0

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        take = self._remaining if n is None or n < 0 else min(n, self._remaining)
        self._remaining -= take
        return b"a" * take

    def close(self) -> None:
        self._remaining = 0


class _StreamingSource:
    """ForeignBlobReader whose object is generated, never materialized."""

    def __init__(self, path: str, size: int) -> None:
        self._path = path
        self._size = size

    def iter_list(self, prefix: str):
        del prefix
        yield self._path

    def iter_child_prefixes(self, prefix: str):
        del prefix
        return iter(())

    def stream(self, path: str) -> IO[bytes]:
        del path
        return _GeneratedReader(self._size)  # type: ignore[return-value]

    def get_properties(self, path: str) -> object:
        del path
        return SimpleNamespace(etag='"stream-etag"', size=self._size)


class _DrainingBlob:
    """Destination blob that drains the upload handle in bounded reads."""

    def __init__(self) -> None:
        self.uploaded_sizes: dict[str, int] = {}

    def read_metadata_value(self, sensitivity: str, path: str, key: str) -> str | None:
        del sensitivity, path, key
        return None

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity, path
        return None

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
        extra_metadata: dict[str, str] | None = None,
    ) -> None:
        del sensitivity, sha256, extra_metadata
        total = 0
        if isinstance(data, bytes):
            total = len(data)
        else:
            while True:
                chunk = data.read(_GEN_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
        self.uploaded_sizes[path] = total


def _contract() -> IngestContract:
    sensitivity: Literal["public", "confidential", "phi"] = "confidential"
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="trilliant-visits-oncology",
        sensitivity=sensitivity,
        pattern="blob_mirror",
        prefix_template="trilliant/visits_oncology/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "source": {
                "account_url": "https://examplestorageacct.blob.core.windows.net",
                "container": "delivery",
                "object_prefix": "{partition_key}/visits_oncology",
            },
            "object_glob": "*.parquet",
        },
        data_owner="vp-data-platform",
        compliance_review="SECURITY.md#trilliant",
    )


def test_mirror_peak_memory_is_bounded() -> None:
    src_path = "202501/visits_oncology/part-00001.snappy.parquet"
    source = _StreamingSource(src_path, _OBJECT_SIZE_BYTES)
    blob = _DrainingBlob()
    pattern = BlobMirrorPattern(source_factory=lambda *_: source)
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.blob_mirror.mem"))
    spec = PartitionSpec(key="202501", metadata={"partition_key": "202501"})

    tracemalloc.start()
    tracemalloc.reset_peak()
    results = pattern.materialize_partition(_contract(), spec, blob, ctx)  # type: ignore[arg-type]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(results) == 1
    assert results[0].size_bytes == _OBJECT_SIZE_BYTES
    landed = "trilliant/visits_oncology/202501/part-00001.snappy.parquet"
    assert blob.uploaded_sizes[landed] == _OBJECT_SIZE_BYTES
    assert peak < _PEAK_THRESHOLD_BYTES, (
        f"peak {peak / 1024 / 1024:.1f} MiB exceeded "
        f"{_PEAK_THRESHOLD_BYTES / 1024 / 1024:.0f} MiB for a "
        f"{_OBJECT_SIZE_BYTES / 1024 / 1024:.0f} MiB object -- mirror is buffering"
    )
