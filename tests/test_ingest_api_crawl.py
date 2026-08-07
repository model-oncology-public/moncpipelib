"""Tests for the ``api_crawl`` ingest pattern (#415 Phase 5).

Everything runs against a mocked HTTP transport and stub crawl plans --
no network.  Coverage per the migration plan:

- happy path: records -> NDJSON blobs -> IngestResults
- multi-filename fan-out
- determinism: identical crawl twice -> byte-identical -> all-"skipped"
- credential handling (absent -> api_key None; present -> secret fetched;
  present without ctx.secrets -> clear error)
- plan exception propagates; nothing uploaded
- empty crawl raises (upstream failure signal)
- throttle wiring: the plan receives a ThrottledClient honoring
  rate_limit_rps
- manifest audit block names plan + resolver
- discovery delegates to the registered resolver
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import IO, Any, ClassVar
from unittest.mock import MagicMock

import pytest

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest._throttle import ThrottledClient
from moncpipelib.ingest.crawl_plans import (
    CRAWL_PLANS,
    CrawlRecord,
    register_crawl_plan,
)
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns import get_pattern
from moncpipelib.ingest.patterns.api_crawl import ApiCrawlPattern
from moncpipelib.ingest.types import IngestContext, PartitionSpec


class FakeBlob:
    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}
        self.upload_calls: list[str] = []

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity
        entry = self.blobs.get(path)
        return entry[1] if entry else None

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
    ) -> None:
        del sensitivity
        self.upload_calls.append(path)
        body = data if isinstance(data, bytes) else data.read()
        self.blobs[path] = (body, sha256)


class _FanOutPlan:
    """Deterministic two-file fan-out plan; records the client it got."""

    name: ClassVar[str] = "fan_out_plan"

    def __init__(self) -> None:
        self.seen_clients: list[Any] = []
        self.seen_api_keys: list[str | None] = []
        self.seen_configs: list[dict[str, Any]] = []

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def crawl(
        self,
        client: ThrottledClient,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[CrawlRecord]:
        del ctx
        self.seen_clients.append(client)
        self.seen_api_keys.append(api_key)
        self.seen_configs.append(dict(config))
        yield CrawlRecord(filename="edges.ndjson", record={"class_id": "A", "rxcui": 1})
        yield CrawlRecord(filename="classes.ndjson", record={"class_id": "A", "name": "alpha"})
        yield CrawlRecord(filename="edges.ndjson", record={"class_id": "B", "rxcui": 2})


class _EmptyPlan:
    name: ClassVar[str] = "empty_plan"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def crawl(
        self,
        client: ThrottledClient,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[CrawlRecord]:
        del client, api_key, config, ctx
        return iter(())


class _ExplodingPlan:
    name: ClassVar[str] = "exploding_plan"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def crawl(
        self,
        client: ThrottledClient,
        api_key: str | None,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[CrawlRecord]:
        del client, api_key, config, ctx
        yield CrawlRecord(filename="partial.ndjson", record={"n": 1})
        raise RuntimeError("upstream 500 at call 14999")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    before = dict(CRAWL_PLANS)
    try:
        yield
    finally:
        CRAWL_PLANS.clear()
        CRAWL_PLANS.update(before)


def _contract(
    *,
    crawl_plan: str = "fan_out_plan",
    credential: dict[str, Any] | None = None,
    rate_limit_rps: float = 1000.0,
    crawl_config: dict[str, Any] | None = None,
) -> IngestContract:
    pattern_config: dict[str, Any] = {
        "crawl_plan": crawl_plan,
        "crawl_config": crawl_config or {"rela_sources": ["ATC", "MESH"]},
        "resolver": "calendar",
        "resolver_config": {},
        "partition": {"mode": "dynamic", "key_from": "partition_key"},
        "rate_limit_rps": rate_limit_rps,
    }
    if credential is not None:
        pattern_config["credential"] = credential
    return IngestContract(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="rxclass",
        sensitivity="public",
        pattern="api_crawl",
        prefix_template="rxclass/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config=pattern_config,
    )


def _spec(key: str = "2026-07") -> PartitionSpec:
    return PartitionSpec(key=key, metadata={"partition_key": key})


def _ctx(secrets: Any = None) -> IngestContext:
    return IngestContext(log=MagicMock(name="LoggingContext"), secrets=secrets)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_api_crawl_is_a_registered_builtin() -> None:
    assert isinstance(get_pattern("api_crawl"), ApiCrawlPattern)


# ---------------------------------------------------------------------------
# materialize_partition
# ---------------------------------------------------------------------------


def test_happy_path_lands_ndjson_blobs_per_filename() -> None:
    plan = _FanOutPlan()
    register_crawl_plan(plan)
    blob = FakeBlob()

    results = ApiCrawlPattern().materialize_partition(
        _contract(),
        _spec(),
        blob,  # type: ignore[arg-type]
        _ctx(),
    )

    assert {r.path for r in results} == {
        "rxclass/2026-07/edges.ndjson",
        "rxclass/2026-07/classes.ndjson",
    }
    assert all(r.action == "uploaded" for r in results)

    edges = blob.blobs["rxclass/2026-07/edges.ndjson"][0].decode().splitlines()
    assert [json.loads(line) for line in edges] == [
        {"class_id": "A", "rxcui": 1},
        {"class_id": "B", "rxcui": 2},
    ]
    classes = blob.blobs["rxclass/2026-07/classes.ndjson"][0].decode().splitlines()
    assert [json.loads(line) for line in classes] == [{"class_id": "A", "name": "alpha"}]

    # size/sha accounting flows from the hashing writer
    for result in results:
        stored_body, stored_sha = blob.blobs[result.path]
        assert result.sha256 == stored_sha
        assert result.size_bytes == len(stored_body)


def test_ndjson_lines_use_sorted_keys() -> None:
    """D4: serialization is deterministic regardless of dict insertion order."""

    class _UnsortedPlan:
        name: ClassVar[str] = "unsorted_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def crawl(
            self,
            client: ThrottledClient,
            api_key: str | None,
            config: dict[str, Any],
            ctx: IngestContext,
        ) -> Iterator[CrawlRecord]:
            del client, api_key, config, ctx
            yield CrawlRecord(filename="a.ndjson", record={"zeta": 1, "alpha": 2})

    register_crawl_plan(_UnsortedPlan())
    blob = FakeBlob()
    ApiCrawlPattern().materialize_partition(
        _contract(crawl_plan="unsorted_plan"),
        _spec(),
        blob,  # type: ignore[arg-type]
        _ctx(),
    )
    body = blob.blobs["rxclass/2026-07/a.ndjson"][0]
    assert body == b'{"alpha":2,"zeta":1}\n'


def test_identical_crawl_reruns_are_skipped_by_hash_compare() -> None:
    plan = _FanOutPlan()
    register_crawl_plan(plan)
    blob = FakeBlob()
    pattern = ApiCrawlPattern()

    first = pattern.materialize_partition(_contract(), _spec(), blob, _ctx())  # type: ignore[arg-type]
    second = pattern.materialize_partition(_contract(), _spec(), blob, _ctx())  # type: ignore[arg-type]

    assert all(r.action == "uploaded" for r in first)
    assert all(r.action == "skipped" for r in second)
    # sha unchanged across runs -- deterministic assembly
    assert {r.sha256 for r in first} == {r.sha256 for r in second}


def test_empty_crawl_raises_and_uploads_nothing() -> None:
    register_crawl_plan(_EmptyPlan())
    blob = FakeBlob()
    with pytest.raises(IngestResolutionError, match="yielded zero records"):
        ApiCrawlPattern().materialize_partition(
            _contract(crawl_plan="empty_plan"),
            _spec(),
            blob,  # type: ignore[arg-type]
            _ctx(),
        )
    assert blob.upload_calls == []


def test_plan_exception_propagates_and_uploads_nothing() -> None:
    register_crawl_plan(_ExplodingPlan())
    blob = FakeBlob()
    with pytest.raises(RuntimeError, match="upstream 500"):
        ApiCrawlPattern().materialize_partition(
            _contract(crawl_plan="exploding_plan"),
            _spec(),
            blob,  # type: ignore[arg-type]
            _ctx(),
        )
    # the partial tempfile is never uploaded; dispatcher would not
    # write a manifest either (D5 -- full re-crawl on next run)
    assert blob.upload_calls == []


def test_unknown_crawl_plan_raises_with_known_names() -> None:
    register_crawl_plan(_FanOutPlan())
    with pytest.raises(KeyError, match="Unknown crawl plan 'nope'"):
        ApiCrawlPattern().materialize_partition(
            _contract(crawl_plan="nope"),
            _spec(),
            FakeBlob(),  # type: ignore[arg-type]
            _ctx(),
        )


def test_empty_pattern_config_raises() -> None:
    contract = IngestContract(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="rxclass",
        sensitivity="public",
        pattern="api_crawl",
        prefix_template="rxclass/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={},
    )
    with pytest.raises(IngestResolutionError, match="empty api_crawl config"):
        ApiCrawlPattern().materialize_partition(
            contract,
            _spec(),
            FakeBlob(),  # type: ignore[arg-type]
            _ctx(),
        )


# ---------------------------------------------------------------------------
# Credential handling (#216 / #218 semantics shared with api_resolver)
# ---------------------------------------------------------------------------


def test_credential_absent_passes_none_api_key() -> None:
    plan = _FanOutPlan()
    register_crawl_plan(plan)
    ApiCrawlPattern().materialize_partition(
        _contract(),
        _spec(),
        FakeBlob(),  # type: ignore[arg-type]
        _ctx(),
    )
    assert plan.seen_api_keys == [None]
    assert plan.seen_configs == [{"rela_sources": ["ATC", "MESH"]}]


def test_credential_present_fetches_secret() -> None:
    plan = _FanOutPlan()
    register_crawl_plan(plan)
    secrets = MagicMock()
    secrets.get_secret.return_value = "s3cret"
    ApiCrawlPattern().materialize_partition(
        _contract(credential={"secret_name": "rxnav-key"}),
        _spec(),
        FakeBlob(),  # type: ignore[arg-type]
        _ctx(secrets=secrets),
    )
    secrets.get_secret.assert_called_once_with("rxnav-key")
    assert plan.seen_api_keys == ["s3cret"]


def test_credential_without_secrets_resource_raises() -> None:
    register_crawl_plan(_FanOutPlan())
    with pytest.raises(IngestResolutionError, match="ApiCrawlPattern.*ctx.secrets is None"):
        ApiCrawlPattern().materialize_partition(
            _contract(credential={"secret_name": "rxnav-key"}),
            _spec(),
            FakeBlob(),  # type: ignore[arg-type]
            _ctx(),
        )


# ---------------------------------------------------------------------------
# Throttle wiring
# ---------------------------------------------------------------------------


def test_plan_receives_throttled_client_with_contract_rate() -> None:
    plan = _FanOutPlan()
    register_crawl_plan(plan)
    ApiCrawlPattern().materialize_partition(
        _contract(rate_limit_rps=4),
        _spec(),
        FakeBlob(),  # type: ignore[arg-type]
        _ctx(),
    )
    (client,) = plan.seen_clients
    assert isinstance(client, ThrottledClient)
    # min interval derived from rate_limit_rps=4
    assert client._min_interval_s == pytest.approx(0.25)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Manifest audit block + discovery delegation
# ---------------------------------------------------------------------------


def test_manifest_resolver_block_names_plan_and_resolver() -> None:
    block = ApiCrawlPattern().manifest_resolver_block(_contract())
    assert block == {
        "name": "fan_out_plan",
        "config": {
            "resolver": "calendar",
            "resolver_config": {},
            "crawl_config": {"rela_sources": ["ATC", "MESH"]},
        },
    }


def test_discover_partitions_delegates_to_resolver() -> None:
    """Uses the builtin calendar resolver (no network, no credential)
    to prove discovery flows through the shared resolver path."""
    contract = _contract()
    contract.pattern_config["resolver_config"].update(
        {
            "url": "https://rxnav.nlm.nih.gov/REST/rxclass/allClasses.json",
            "cadence": "monthly",
            "start_date": "2026-06-01",
        }
    )
    specs = ApiCrawlPattern().discover_partitions(contract, _ctx())
    assert specs, "calendar resolver should emit at least one partition"
    assert all(spec.key for spec in specs)
