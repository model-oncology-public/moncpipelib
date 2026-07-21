"""Tests for load_all_contracts: cross-contract validation."""

from __future__ import annotations

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
