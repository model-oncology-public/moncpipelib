"""Tests for match: many multi-file resolution (#438).

Covers the opt-in ``match: many`` cardinality on both the static
(enumerated-period) and ``FromIngestTemplate`` branches of
``resolve_source_for_partition``: N sorted BlobRefs when many parts
land, a raise on zero, and the unchanged exactly-one default.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from io import BytesIO
from typing import IO, Literal

import pytest

from moncpipelib.contracts.models import (
    ContractCorpus,
    DataSource,
    FromIngestTemplate,
    IngestContract,
    Period,
)
from moncpipelib.ingest import (
    BlobRef,
    IngestResolutionError,
    resolve_source_for_partition,
)


class FakeBlob:
    def __init__(
        self,
        listing: list[str] | None = None,
        contents: dict[str, bytes] | None = None,
    ) -> None:
        self._listing = list(listing or [])
        self._contents = dict(contents or {})

    def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
        del sensitivity, prefix
        return iter(self._listing)

    def exists(self, sensitivity: str, path: str) -> bool:
        del sensitivity
        return path in self._contents

    def stream(self, sensitivity: str, path: str) -> IO[bytes]:
        del sensitivity
        return BytesIO(self._contents[path])


def _ingest(
    *,
    sensitivity: Literal["public", "confidential", "phi"] = "confidential",
    prefix_template: str = "trilliant/visits_oncology/{partition_key}",
    pattern: str = "blob_mirror",
) -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="trilliant-visits-oncology",
        sensitivity=sensitivity,
        pattern=pattern,
        prefix_template=prefix_template,
        extract=(),
        strip_extensions=(),
        pattern_config={},
    )


def _static_source(match: str = "many", glob: str = "*.parquet") -> DataSource:
    return DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="trilliant-visits-oncology-bronze",
        periods=[
            Period(
                source=glob,
                effective_from=date(2025, 1, 1),
                partition_key="202501",
                match=match,  # type: ignore[arg-type]
            )
        ],
        ingest_source="trilliant-visits-oncology",
    )


def _corpus(source: DataSource) -> ContractCorpus:
    ingest = _ingest()
    return ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )


def _parts_listing() -> list[str]:
    # Deliberately unsorted to prove the resolver sorts.
    return [
        "trilliant/visits_oncology/202501/part-00003.parquet",
        "trilliant/visits_oncology/202501/part-00001.parquet",
        "trilliant/visits_oncology/202501/_manifest.json",
        "trilliant/visits_oncology/202501/part-00002.parquet",
    ]


def test_match_many_returns_all_parts_sorted() -> None:
    source = _static_source(match="many")
    blob = FakeBlob(listing=_parts_listing())
    refs = resolve_source_for_partition(source, "202501", _corpus(source), blob)  # type: ignore[arg-type]

    assert all(isinstance(r, BlobRef) for r in refs)
    assert [r.path for r in refs] == [  # type: ignore[union-attr]
        "trilliant/visits_oncology/202501/part-00001.parquet",
        "trilliant/visits_oncology/202501/part-00002.parquet",
        "trilliant/visits_oncology/202501/part-00003.parquet",
    ]
    # manifest excluded even under a permissive match
    assert all(not r.path.endswith("_manifest.json") for r in refs)  # type: ignore[union-attr]


def test_match_many_zero_raises() -> None:
    source = _static_source(match="many")
    blob = FakeBlob(listing=["trilliant/visits_oncology/202501/_manifest.json"])
    with pytest.raises(IngestResolutionError, match=r"Expected >=1 match"):
        resolve_source_for_partition(source, "202501", _corpus(source), blob)  # type: ignore[arg-type]


def test_match_one_default_still_raises_on_multiple() -> None:
    source = _static_source(match="one")
    blob = FakeBlob(listing=_parts_listing())
    with pytest.raises(IngestResolutionError, match=r"Expected exactly 1 match"):
        resolve_source_for_partition(source, "202501", _corpus(source), blob)  # type: ignore[arg-type]


def test_match_defaults_to_one() -> None:
    # A period with no explicit match behaves as before (exactly-one).
    source = DataSource(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="trilliant-single",
        periods=[
            Period(source="*.parquet", effective_from=date(2025, 1, 1), partition_key="202501")
        ],
        ingest_source="trilliant-visits-oncology",
    )
    assert source.periods[0].match == "one"  # type: ignore[index]
    blob = FakeBlob(listing=["trilliant/visits_oncology/202501/part-00001.parquet"])
    refs = resolve_source_for_partition(source, "202501", _corpus(source), blob)  # type: ignore[arg-type]
    assert len(refs) == 1


# ---------------------------------------------------------------------------
# FromIngestTemplate branch
# ---------------------------------------------------------------------------

_MANIFEST = """{
  "fields": {"partition_key": "202501"},
  "files": [
    {"path": "trilliant/visits_oncology/202501/part-00001.parquet", "sha256": "a", "size_bytes": 1}
  ],
  "manifest_version": 1,
  "materialized_at": "2026-07-17T00:00:00Z",
  "partition_key": "202501",
  "resolver": {"config": {}, "name": "blob_mirror"},
  "source_id": "11111111-1111-1111-1111-111111111111",
  "source_name": "trilliant-visits-oncology"
}"""


def _template_source(match: str = "many") -> DataSource:
    return DataSource(
        source_id="44444444-4444-4444-4444-444444444444",
        source_name="trilliant-visits-oncology-bronze",
        periods=FromIngestTemplate(
            source="*.parquet",
            effective_from_field="partition_key",
            match=match,  # type: ignore[arg-type]
        ),
        ingest_source="trilliant-visits-oncology",
    )


def test_from_template_match_many_returns_all_parts() -> None:
    source = _template_source(match="many")
    manifest_path = "trilliant/visits_oncology/202501/_manifest.json"
    blob = FakeBlob(
        listing=[
            manifest_path,
            "trilliant/visits_oncology/202501/part-00002.parquet",
            "trilliant/visits_oncology/202501/part-00001.parquet",
        ],
        contents={manifest_path: _MANIFEST.encode("utf-8")},
    )
    refs = resolve_source_for_partition(source, "202501", _corpus(source), blob)  # type: ignore[arg-type]
    assert [r.path for r in refs] == [  # type: ignore[union-attr]
        "trilliant/visits_oncology/202501/part-00001.parquet",
        "trilliant/visits_oncology/202501/part-00002.parquet",
    ]
