"""Tests for download_partition_parts_with_manifest (#438 + #439 glue).

Resolves a partition's N parts, downloads each to a seekable local
tempfile, and yields the paths + manifest fields.  Uses an in-memory
FakeBlob whose ``download_to_path`` writes real files so the yielded
paths can be read back (and, in the parquet case, scanned).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import IO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from moncpipelib.contracts.models import (
    ContractCorpus,
    DataSource,
    FromIngestTemplate,
    IngestContract,
    Period,
)
from moncpipelib.ingest import download_partition_parts_with_manifest
from moncpipelib.streaming import stream_parquet_batches

_PREFIX = "trilliant/visits_oncology/202501"
_MANIFEST_PATH = f"{_PREFIX}/_manifest.json"
_MANIFEST = (
    b'{"fields": {"partition_key": "202501"}, "files": [], "manifest_version": 1, '
    b'"materialized_at": "2026-07-17T00:00:00Z", "partition_key": "202501", '
    b'"resolver": {"config": {}, "name": "blob_mirror"}, '
    b'"source_id": "11111111-1111-1111-1111-111111111111", '
    b'"source_name": "trilliant-visits-oncology"}'
)


class FakeBlob:
    def __init__(self, listing: list[str], contents: dict[str, bytes]) -> None:
        self._listing = list(listing)
        self._contents = dict(contents)

    def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
        del sensitivity, prefix
        return iter(self._listing)

    def exists(self, sensitivity: str, path: str) -> bool:
        del sensitivity
        return path in self._contents

    def stream(self, sensitivity: str, path: str) -> IO[bytes]:
        del sensitivity
        return BytesIO(self._contents[path])

    def download_to_path(self, sensitivity: str, src: str, dest: Path | str) -> None:
        del sensitivity
        Path(dest).write_bytes(self._contents[src])


def _corpus(source: DataSource) -> ContractCorpus:
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="trilliant-visits-oncology",
        sensitivity="confidential",
        pattern="blob_mirror",
        prefix_template="trilliant/visits_oncology/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={},
    )
    return ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )


def _source_many() -> DataSource:
    return DataSource(
        source_id="44444444-4444-4444-4444-444444444444",
        source_name="trilliant-bronze",
        periods=FromIngestTemplate(
            source="*.parquet", effective_from_field="partition_key", match="many"
        ),
        ingest_source="trilliant-visits-oncology",
    )


def _parquet_bytes(ids: list[int]) -> bytes:
    from io import BytesIO as _B

    buf = _B()
    pq.write_table(pa.table({"id": ids, "v": [f"v{i}" for i in ids]}), buf)
    return buf.getvalue()


def test_downloads_all_parts_and_yields_paths_and_fields() -> None:
    contents = {
        _MANIFEST_PATH: _MANIFEST,
        f"{_PREFIX}/part-00001.parquet": _parquet_bytes([1, 2]),
        f"{_PREFIX}/part-00002.parquet": _parquet_bytes([3]),
    }
    blob = FakeBlob(listing=list(contents), contents=contents)
    with download_partition_parts_with_manifest(
        source=_source_many(),
        partition_key="202501",
        corpus=_corpus(_source_many()),
        blob=blob,  # type: ignore[arg-type]
    ) as (paths, fields):
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert fields == {"partition_key": "202501"}
        # end-to-end: the parts scan as one logical parquet stream
        rows = sum(b.height for b in stream_parquet_batches(paths))
        assert rows == 3
    # tempdir cleaned up on context exit
    assert all(not p.exists() for p in paths)


def test_required_fields_enforced() -> None:
    from moncpipelib.ingest import ManifestFieldError

    contents = {
        _MANIFEST_PATH: _MANIFEST,
        f"{_PREFIX}/part-00001.parquet": _parquet_bytes([1]),
    }
    blob = FakeBlob(listing=list(contents), contents=contents)
    with (
        pytest.raises(ManifestFieldError),
        download_partition_parts_with_manifest(
            source=_source_many(),
            partition_key="202501",
            corpus=_corpus(_source_many()),
            blob=blob,  # type: ignore[arg-type]
            required_fields=["release_version"],  # absent from manifest
        ),
    ):
        pass


def test_rejects_legacy_url_source() -> None:
    source = DataSource(
        source_id="55555555-5555-5555-5555-555555555555",
        source_name="legacy",
        periods=[
            Period(
                source="https://example.com/x.csv",
                effective_from=date(2025, 1, 1),
                partition_key="202501",
            )
        ],
        ingest_source=None,  # legacy -> RawUrl
    )
    blob = FakeBlob(listing=[], contents={})
    with (
        pytest.raises(TypeError, match="BlobRefs"),
        download_partition_parts_with_manifest(
            source=source,
            partition_key="202501",
            corpus=_corpus(source),
            blob=blob,  # type: ignore[arg-type]
        ),
    ):
        pass
