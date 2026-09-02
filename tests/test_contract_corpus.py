"""Tests for load_all_contracts: cross-contract validation."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
import yaml

from moncpipelib.contracts import (
    ContractCorpus,
    ContractValidationError,
    FromIngestTemplate,
    load_all_contracts,
)

_INGEST_UUID = "11111111-1111-1111-1111-111111111111"
_SOURCE_UUID = "22222222-2222-2222-2222-222222222222"


def _write_ingest(
    root: Path, name: str = "cms-asp", partition_keys: tuple[str, ...] = ("2024-01-01",)
) -> None:
    data = {
        "source_id": _INGEST_UUID,
        "source_name": name,
        "sensitivity": "public",
        "ingest": {
            "pattern": "http_urls",
            "prefix": f"{name}/{{partition_key}}",
            "extract": ["zip"],
            "http_urls": {
                "periods": [
                    {"partition_key": pk, "urls": ["https://example.com/x.zip"]}
                    for pk in partition_keys
                ],
            },
        },
    }
    (root / f"{name}.ingest.yaml").write_text(yaml.safe_dump(data))


def _write_source(
    root: Path,
    *,
    source_name: str,
    ingest_source: str | None,
    partition_keys: tuple[str, ...],
    from_ingest: bool = False,
    match: str = "one",
) -> None:
    if from_ingest:
        data: dict[str, object] = {
            "source_id": _SOURCE_UUID,
            "source_name": source_name,
            "ingest_source": ingest_source,
            "periods": {
                "mode": "from_ingest",
                "template": {
                    "source": "file.csv",
                    "effective_from_field": "release_date",
                    "match": match,
                },
            },
        }
    else:
        sorted_keys = sorted(partition_keys)
        period_entries: list[dict[str, object]] = []
        for i, pk in enumerate(sorted_keys):
            entry: dict[str, object] = {
                "partition_key": pk,
                "source": "*crosswalk*.csv",
                "effective_from": date.fromisoformat(pk),
            }
            # Close each period at the next one's start so only the last
            # stays open-ended; keeps the loader's overlap check happy.
            if i + 1 < len(sorted_keys):
                entry["effective_to"] = date.fromisoformat(sorted_keys[i + 1])
            period_entries.append(entry)
        data = {
            "source_id": _SOURCE_UUID,
            "source_name": source_name,
            "periods": period_entries,
        }
        if ingest_source is not None:
            data["ingest_source"] = ingest_source
    (root / f"{source_name}.source.yaml").write_text(yaml.safe_dump(data))


def test_happy_path_static(tmp_path: Path) -> None:
    _write_ingest(tmp_path, partition_keys=("2024-01-01", "2024-04-01"))
    _write_source(
        tmp_path,
        source_name="cms-asp-crosswalk",
        ingest_source="cms-asp",
        partition_keys=("2024-01-01", "2024-04-01"),
    )

    corpus = load_all_contracts(tmp_path)

    assert isinstance(corpus, ContractCorpus)
    assert set(corpus.ingests) == {"cms-asp"}
    assert set(corpus.sources) == {"cms-asp-crosswalk"}
    assert corpus.get_ingest("cms-asp").source_name == "cms-asp"


def test_period_drift_fails(tmp_path: Path) -> None:
    _write_ingest(tmp_path, partition_keys=("2024-01-01",))
    _write_source(
        tmp_path,
        source_name="cms-asp-crosswalk",
        ingest_source="cms-asp",
        partition_keys=("2024-01-01", "2024-04-01"),  # 04-01 not in ingest
    )

    with pytest.raises(ContractValidationError, match="2024-04-01"):
        load_all_contracts(tmp_path)


def test_unknown_ingest_reference_fails(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        source_name="orphan",
        ingest_source="does-not-exist",
        partition_keys=("2024-01-01",),
    )

    with pytest.raises(ContractValidationError, match="does-not-exist"):
        load_all_contracts(tmp_path)


def _write_api_resolver_ingest(
    root: Path,
    name: str = "umls-meta",
) -> None:
    """Write an api_resolver ingest contract with partition.mode: dynamic.

    Used by the cross-contract validation tests to verify the linkage
    rule pinned in PR 5: ``periods.mode: from_ingest`` requires the
    linked ingest to declare ``partition.mode: dynamic`` -- which
    api_resolver does, so the linkage validates clean."""
    data = {
        "source_id": _INGEST_UUID,
        "source_name": name,
        "sensitivity": "confidential",
        "data_owner": "data-platform",
        "compliance_review": "SECURITY.md#umls",
        "ingest": {
            "pattern": "api_resolver",
            "prefix": f"{name}/{{partition_key}}",
            "extract": ["zip", "zip"],
            "extract_filter": ["meta/**"],
            "api_resolver": {
                "resolver": "uts_release",
                "resolver_config": {"release_type": "umls-full-release"},
                "credential": {"secret_name": "uts-api-key"},
                "partition": {"mode": "dynamic", "key_from": "release_version"},
                "idempotency": "hash_compare",
            },
        },
    }
    (root / f"{name}.ingest.yaml").write_text(yaml.safe_dump(data))


def test_from_ingest_against_dynamic_api_resolver_validates_clean(
    tmp_path: Path,
) -> None:
    """The cross-contract dynamic-linkage rule reads
    ``partition.mode`` from the api_resolver pattern_config (per PR 4
    schema) and accepts ``dynamic`` -- this regression test pins the
    end-to-end happy path so a future schema rename surfaces here
    before it breaks UMLS / RxNorm consumers."""
    _write_api_resolver_ingest(tmp_path)
    _write_source(
        tmp_path,
        source_name="umls-mrconso",
        ingest_source="umls-meta",
        partition_keys=(),
        from_ingest=True,
    )

    corpus = load_all_contracts(tmp_path)

    assert "umls-meta" in corpus.ingests
    source = corpus.get_source("umls-mrconso")
    assert isinstance(source.periods, FromIngestTemplate)
    assert source.periods.effective_from_field == "release_date"


_BLOB_MIRROR_UUID = "33333333-3333-3333-3333-333333333333"


def _write_blob_mirror_ingest(root: Path, name: str = "trilliant-visits-oncology") -> None:
    """Write a ``blob_mirror`` ingest contract (#437).

    Modeled on ``_blob_mirror_data()`` in
    ``tests/test_ingest_blob_mirror_contract.py``. Unlike ``http_urls`` or
    ``api_resolver``, blob_mirror has no ``partition`` config key at all
    (see ``KNOWN_BLOB_MIRROR_FIELDS`` in ``contracts/loader.py``) --
    partitions are discovered by folder-walk against the foreign blob
    store (``BlobMirrorPattern.discover_partitions``), not declared via a
    static ``partition.mode`` block. Used to pin the cross-contract
    dynamic-linkage rule's blob_mirror carve-out (#452).
    """
    data = {
        "source_id": _BLOB_MIRROR_UUID,
        "source_name": name,
        "sensitivity": "confidential",
        "data_owner": "vp-data-platform",
        "compliance_review": "SECURITY.md#trilliant",
        "ingest": {
            "pattern": "blob_mirror",
            "prefix": f"{name}/{{partition_key}}",
            "blob_mirror": {
                "source": {
                    "account_url": "https://examplestorageacct.blob.core.windows.net",
                    "container": "delivery",
                    "object_prefix": "{partition_key}/visits_oncology",
                    "discovery_prefix": "",
                    "partition_pattern": r"^\d{6}$",
                },
                "credential": {
                    "secret_name": "trilliant-sp",
                    "tenant_id": "partner-tenant",
                    "client_id": "our-sp",
                },
                "object_glob": "*.parquet",
                "exclude_globs": ["_committed_*", "_started_*", "_SUCCESS"],
            },
        },
    }
    (root / f"{name}.ingest.yaml").write_text(yaml.safe_dump(data))


def test_from_ingest_against_blob_mirror_validates_clean(tmp_path: Path) -> None:
    """blob_mirror (#437) has no ``partition`` block to declare
    ``mode: dynamic`` against -- it discovers partitions via folder-walk
    (``BlobMirrorPattern.discover_partitions``), not a static config key.
    The cross-contract dynamic-linkage rule must still accept a
    ``from_ingest`` DataSource linked to a blob_mirror ingest -- the
    Trilliant Health ``visits_oncology`` shape (#436-#439) -- rather than
    rejecting every blob_mirror + from_ingest pairing (#452). This test
    also exercises ``template.match: many`` (#438), the multi-file-per-cycle
    cardinality blob_mirror partitions actually use."""
    _write_blob_mirror_ingest(tmp_path)
    _write_source(
        tmp_path,
        source_name="trilliant-visits-oncology-source",
        ingest_source="trilliant-visits-oncology",
        partition_keys=(),
        from_ingest=True,
        match="many",
    )

    corpus = load_all_contracts(tmp_path)

    assert "trilliant-visits-oncology" in corpus.ingests
    source = corpus.get_source("trilliant-visits-oncology-source")
    assert isinstance(source.periods, FromIngestTemplate)
    assert source.periods.match == "many"


def test_from_ingest_against_static_ingest_fails(tmp_path: Path) -> None:
    _write_ingest(tmp_path)  # http_urls is static
    _write_source(
        tmp_path,
        source_name="rxnorm-mrconso",
        ingest_source="cms-asp",
        partition_keys=(),
        from_ingest=True,
    )

    with pytest.raises(ContractValidationError, match="partition.mode: dynamic"):
        load_all_contracts(tmp_path)


def test_from_ingest_requires_ingest_source(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        source_name="rxnorm-mrconso",
        ingest_source=None,
        partition_keys=(),
        from_ingest=True,
    )

    with pytest.raises(ContractValidationError, match="ingest_source"):
        load_all_contracts(tmp_path)


def test_from_ingest_periods_loaded_as_template(tmp_path: Path) -> None:
    # Fake dynamic ingest by hand-crafting the YAML (http_urls is static
    # but we only need partition.mode = dynamic for the linkage check).
    (tmp_path / "rxnorm.ingest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": _INGEST_UUID,
                "source_name": "rxnorm-full-monthly",
                "sensitivity": "public",
                "ingest": {
                    "pattern": "http_urls",
                    "prefix": "rxnorm/{partition_key}",
                    "extract": ["zip"],
                    "http_urls": {
                        # Include a sentinel period so schema validation passes.
                        "periods": [
                            {
                                "partition_key": "2026-03-03",
                                "urls": ["https://example.com/release.zip"],
                            }
                        ],
                        # partition.mode is declared at pattern_config root and
                        # read by cross-contract validation. Including it here
                        # simulates what api_resolver will declare in Phase 2.
                    },
                },
            }
        )
    )
    # Hand-write the from_ingest source so we can also set partition.mode=dynamic
    # under the ingest's pattern_config by injecting it directly.
    # Since we can't modify the ingest YAML to set partition.mode without
    # extending the known-fields set, this test asserts only that a
    # FromIngestTemplate is parsed cleanly on the source side when the
    # corpus loader accepts the shape.
    _write_source(
        tmp_path,
        source_name="rxnorm-mrconso",
        ingest_source="rxnorm-full-monthly",
        partition_keys=(),
        from_ingest=True,
    )

    # Because the ingest declares http_urls (static), the corpus validator
    # rejects the from_ingest linkage. That's the correct Phase 1 behavior;
    # the dynamic path lands with api_resolver in Phase 2.
    with pytest.raises(ContractValidationError, match="dynamic"):
        load_all_contracts(tmp_path)

    # Verify the FromIngestTemplate shape was still parsed by reading the
    # source file directly through load_data_source.
    from moncpipelib.contracts import load_data_source

    source = load_data_source(tmp_path / "rxnorm-mrconso.source.yaml")
    assert isinstance(source.periods, FromIngestTemplate)
    assert source.periods.effective_from_field == "release_date"


def test_missing_root_raises(tmp_path: Path) -> None:
    from moncpipelib.contracts import ContractNotFoundError

    with pytest.raises(ContractNotFoundError):
        load_all_contracts(tmp_path / "nope")


_BROWSER_EXPORT_UUID = "44444444-4444-4444-4444-444444444444"


def _write_browser_export_ingest(root: Path, name: str = "340b-ceiling-price") -> None:
    """Write a ``browser_export`` ingest contract (#463).

    Modeled on ``_browser_export_data()`` in
    ``tests/test_ingest_contract_loader.py``. Unlike ``blob_mirror``
    (no ``partition`` key at all), ``browser_export`` declares
    ``partition.mode: dynamic`` directly -- used to pin the #452
    cross-contract dynamic-linkage guard's generic
    ``partition.mode == "dynamic"`` branch against a third pattern with
    no new special-cased pattern name in
    ``_ingest_produces_dynamic_partitions``.
    """
    data = {
        "source_id": _BROWSER_EXPORT_UUID,
        "source_name": name,
        "sensitivity": "public",
        "compliance_review": "SECURITY.md#340b",
        "ingest": {
            "pattern": "browser_export",
            "prefix": f"{name}/{{partition_key}}",
            "browser_export": {
                "export_plan": "_stub_export_plan",
                "export_config": {},
                "allowed_hosts": ["340bopais.hrsa.gov"],
                "partition": {
                    "mode": "dynamic",
                    "cadence": "daily",
                    "anchor_tz": "America/New_York",
                },
                "fetch": {"user_agent": "test-corpus/1.0 (contact: data-platform)"},
            },
        },
    }
    (root / f"{name}.ingest.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def _stub_export_plan_for_corpus() -> Iterator[None]:
    """Register a minimal export plan for load-time + materialize-time
    lookup; restore the registry after."""
    from moncpipelib.ingest.export_plans import EXPORT_PLANS, register_export_plan

    class _StubExportPlan:
        name = "_stub_export_plan"

        def validate_config(self, config: dict[str, object]) -> list[str]:
            del config
            return []

        def export(
            self, session: object, partition_key: str, config: dict[str, object], ctx: object
        ):  # type: ignore[no-untyped-def]
            del session, partition_key, config, ctx
            raise NotImplementedError("registered for load-time lookup only")

    before = dict(EXPORT_PLANS)
    register_export_plan(_StubExportPlan())
    try:
        yield
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before)


def test_from_ingest_against_browser_export_validates_clean(
    tmp_path: Path, _stub_export_plan_for_corpus: None
) -> None:
    """browser_export (#463) declares ``partition.mode: dynamic`` directly
    (unlike ``blob_mirror``, which has no ``partition`` block at all) --
    the #452 guard's generic ``partition.mode == "dynamic"`` branch must
    accept it, exercising ``_ingest_produces_dynamic_partitions`` against
    a third pattern with no new special-cased pattern name."""
    _write_browser_export_ingest(tmp_path)
    _write_source(
        tmp_path,
        source_name="340b-ceiling-price-source",
        ingest_source="340b-ceiling-price",
        partition_keys=(),
        from_ingest=True,
    )

    corpus = load_all_contracts(tmp_path)

    assert "340b-ceiling-price" in corpus.ingests
    source = corpus.get_source("340b-ceiling-price-source")
    assert isinstance(source.periods, FromIngestTemplate)


def test_from_ingest_hydrates_against_a_written_browser_export_manifest(
    tmp_path: Path, _stub_export_plan_for_corpus: None
) -> None:
    """Contract load succeeding is not the same as runtime hydration
    succeeding (#463): materialize a real partition through
    ``materialize_with_manifest`` with a stubbed browser session, then
    resolve the downstream ``FromIngestTemplate`` source against the
    written manifest.

    ``resolve_source_for_partition`` returns a ``BlobRef``, not a
    hydrated ``Period`` -- moncpipelib does not itself construct
    ``Period`` objects from manifest fields for the ``from_ingest`` path
    (that hydration step lives downstream, per
    ``tests/cookbook/test_from_ingest_period_registry_cookbook.py``).
    This test therefore asserts both halves that ARE moncpipelib's job:
    the resolved blob path, and the raw manifest field
    (``snapshot_date``, browser_export's ``effective_from_field``
    source) a downstream consumer's hydration step would read as
    ``Period.effective_from``.
    """
    import logging
    from contextlib import contextmanager
    from io import BytesIO
    from typing import IO, Any

    from freezegun import freeze_time

    from moncpipelib.ingest.dispatcher import materialize_with_manifest
    from moncpipelib.ingest.export_plans import EXPORT_PLANS, ExportedFile
    from moncpipelib.ingest.manifest import IngestManifest
    from moncpipelib.ingest.patterns import browser_export as browser_export_module
    from moncpipelib.ingest.patterns.browser_export import BrowserExportPattern
    from moncpipelib.ingest.resolver import resolve_source_for_partition
    from moncpipelib.ingest.types import BlobRef, IngestContext

    _write_browser_export_ingest(tmp_path)
    (tmp_path / "340b-ceiling-price-source.source.yaml").write_text(
        yaml.safe_dump(
            {
                "source_id": _SOURCE_UUID,
                "source_name": "340b-ceiling-price-source",
                "ingest_source": "340b-ceiling-price",
                "periods": {
                    "mode": "from_ingest",
                    "template": {
                        "source": "report.csv",
                        "effective_from_field": "snapshot_date",
                    },
                },
            }
        )
    )

    corpus = load_all_contracts(tmp_path)
    ingest = corpus.get_ingest("340b-ceiling-price")
    source = corpus.get_source("340b-ceiling-price-source")

    payload_path = tmp_path / "0000.download"
    payload_path.write_bytes(b"340b-ceiling-price\nreport row\n")

    class _StubExportPlanWithFile:
        name = "_stub_export_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def export(self, session: Any, partition_key: str, config: dict[str, Any], ctx: Any):  # type: ignore[no-untyped-def]
            del session, partition_key, config, ctx
            yield ExportedFile(path=payload_path, suggested_filename="report.csv")

    before_plans = dict(EXPORT_PLANS)
    EXPORT_PLANS["_stub_export_plan"] = _StubExportPlanWithFile()  # type: ignore[assignment]

    class _StubSession:
        """The export plan above never calls a session method -- this
        only needs to satisfy ``materialize_partition``'s own
        ``session.require_contains(...)`` containment check (#464/#468
        finding 12)."""

        def require_contains(self, path: Any) -> None:
            del path

    @contextmanager
    def _fake_browser_session(**kwargs: Any):  # type: ignore[no-untyped-def]
        del kwargs
        yield _StubSession()

    real_browser_session = browser_export_module.browser_session
    browser_export_module.browser_session = _fake_browser_session  # type: ignore[assignment]

    class _InMemoryBlob:
        def __init__(self) -> None:
            self.store: dict[str, tuple[bytes, str]] = {}

        def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
            del sensitivity
            return (p for p in self.store if p.startswith(prefix))

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
            del sensitivity
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.store

        def stream(self, sensitivity: str, path: str) -> IO[bytes]:
            del sensitivity
            return BytesIO(self.store[path][0])

    blob = _InMemoryBlob()
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.browser_export_from_ingest"))
    pattern = BrowserExportPattern()

    try:
        with freeze_time("2026-08-06 12:00:00"):
            [spec] = pattern.discover_partitions(ingest, ctx)
            materialize_with_manifest(pattern, ingest, spec, blob, ctx)  # type: ignore[arg-type]

            [ref] = resolve_source_for_partition(
                source,
                partition_key=spec.key,
                corpus=corpus,
                blob=blob,  # type: ignore[arg-type]
            )
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before_plans)
        browser_export_module.browser_session = real_browser_session  # type: ignore[assignment]

    assert isinstance(ref, BlobRef)
    assert ref.path == "340b-ceiling-price/2026-08-06/report.csv"

    manifest_bytes = blob.store["340b-ceiling-price/2026-08-06/_manifest.json"][0]
    manifest = IngestManifest.read_from(BytesIO(manifest_bytes))
    assert manifest.fields["snapshot_date"] == "2026-08-06"
