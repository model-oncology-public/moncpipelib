"""Cookbook test for the ``api_crawl`` ingest pattern (#415).

Demonstrates landing a source that has **no bulk download** -- the
dataset is assembled by crawling a REST/JSON API (enumerate keys, then
fan out one call per key):

- Register a custom :class:`CrawlPlan` that walks a mocked RxClass-style
  API through the provided :class:`ThrottledClient` (so the example runs
  in CI without network access) and yields records.
- Register a stub :class:`ReleaseResolver` for period discovery
  (version-driven monthly keys in production; fixed here).
- Construct an ``api_crawl`` :class:`IngestContract` with the required
  ``rate_limit_rps`` budget.
- Call :func:`materialize_with_manifest` and observe the assembled
  NDJSON blob + per-partition manifest land; the manifest's audit block
  names the crawl plan.
- Re-run to demonstrate ``hash_compare`` idempotency (deterministic
  plans produce byte-identical NDJSON when the upstream is unchanged).

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.
"""

from __future__ import annotations

from typing import Any

import pytest

from moncpipelib.ingest.crawl_plans import CRAWL_PLANS
from moncpipelib.ingest.resolvers import RESOLVERS


@pytest.fixture
def cookbook_crawl_registrations() -> Any:
    """Restore the crawl-plan + resolver registries after the example."""
    plans_before = dict(CRAWL_PLANS)
    resolvers_before = dict(RESOLVERS)
    yield
    CRAWL_PLANS.clear()
    CRAWL_PLANS.update(plans_before)
    RESOLVERS.clear()
    RESOLVERS.update(resolvers_before)


@pytest.mark.cookbook(
    title="Assemble a no-bulk-file source with api_crawl + a crawl plan",
    description=(
        "Some reference APIs (e.g. NLM RxClass) publish no bulk release "
        "file -- the dataset only exists as the union of many small JSON "
        "responses. The ``api_crawl`` pattern executes a registered "
        ":class:`CrawlPlan` that walks the API through a rate-limited, "
        "redacting client and yields records; the pattern streams them "
        "into NDJSON blob(s) with hash-in-the-same-pass and inherits the "
        "manifest + idempotency machinery. This example registers a stub "
        "plan + resolver against a mocked API so it runs in CI. Real "
        "pipelines register a production plan (in consumer code, like "
        "resolvers) and set ``rate_limit_rps`` at or below the upstream's "
        "published cap (RxNav: 20 req/s per IP)."
    ),
    category="ingest",
)
def test_cookbook_api_crawl_roundtrip(cookbook_crawl_registrations: None) -> None:
    del cookbook_crawl_registrations
    # --- cookbook:start ---
    import json
    import logging
    from collections.abc import Iterator
    from typing import IO, ClassVar

    import respx

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest import (
        ApiCrawlPattern,
        CrawlRecord,
        IngestContext,
        IngestManifest,
        ThrottledClient,
        materialize_with_manifest,
        register_crawl_plan,
        register_resolver,
    )
    from moncpipelib.ingest.resolvers import ResolvedDownload

    # --- 1. Register a crawl plan ---
    # The plan is the per-source extension point (registered from consumer
    # code, like resolvers). It receives a ThrottledClient -- the ONLY
    # sanctioned network surface: requests are paced to the contract's
    # rate_limit_rps and routed through the redacting transport. Plans
    # MUST enumerate in a deterministic order (sorted) so re-runs of an
    # unchanged upstream produce byte-identical NDJSON and skip via
    # hash-compare.
    class _RxClassPlan:
        name: ClassVar[str] = "_cookbook_rxclass"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            unknown = set(config) - {"rela_source"}
            return [f"unknown field {k!r}" for k in sorted(unknown)]

        def crawl(
            self,
            client: ThrottledClient,
            api_key: str | None,
            config: dict[str, Any],
            ctx: Any,
        ) -> Iterator[CrawlRecord]:
            del api_key, ctx  # public API; nothing secret to send
            base = "https://rxnav.example/REST/rxclass"
            # Fan-out shape: enumerate keys, then one call per key.
            listing = client.get(f"{base}/allClasses.json").json()
            class_ids = sorted(c["classId"] for c in listing["classes"])
            for class_id in class_ids:
                members = client.get(
                    f"{base}/classMembers.json",
                    params={"classId": class_id, "relaSource": config["rela_source"]},
                ).json()
                for rxcui in sorted(members["rxcuis"]):
                    yield CrawlRecord(
                        filename="drug_class_edges.ndjson",
                        record={"class_id": class_id, "rxcui": rxcui},
                    )

    register_crawl_plan(_RxClassPlan())

    # --- 2. Register a resolver for period discovery ---
    # api_crawl discovers partitions exactly like api_resolver: a
    # resolver emits release dicts keyed by partition.key_from. For
    # version-driven crawl sources the production resolver reads a cheap
    # version endpoint and emits a new monthly key when a tracked source
    # version changes; this stub returns a fixed period. There is no
    # download URL to resolve -- resolve_url raising is correct.
    class _RxClassVersionResolver:
        name: ClassVar[str] = "_cookbook_rxclass_version"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def current_release(
            self, api_key: str | None, config: dict[str, Any], ctx: Any
        ) -> dict[str, Any]:
            del api_key, config, ctx
            return {"partition_key": "2026-07", "snapshot_date": "2026-07-01"}

        def resolve_url(
            self, api_key: str | None, partition_key: str, config: dict[str, Any], ctx: Any
        ) -> ResolvedDownload:
            raise NotImplementedError("api_crawl sources have no download URL")

        def historical_release(
            self, api_key: str | None, config: dict[str, Any], ctx: Any
        ) -> list[dict[str, Any]]:
            del api_key, config, ctx
            return []

    register_resolver(_RxClassVersionResolver())  # type: ignore[arg-type]

    # --- 3. Declare the api_crawl ingest contract ---
    # In production this is a *.ingest.yaml loaded via
    # load_ingest_contract; rate_limit_rps is REQUIRED (no default) and
    # must sit at or below the upstream's published cap. No credential
    # block: RxClass is a public API.
    ingest = IngestContract(
        source_id="55555555-5555-5555-5555-555555555555",
        source_name="cookbook-rxclass",
        sensitivity="public",
        pattern="api_crawl",
        prefix_template="rxclass/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "crawl_plan": "_cookbook_rxclass",
            "crawl_config": {"rela_source": "ATC"},
            "resolver": "_cookbook_rxclass_version",
            "resolver_config": {},
            "partition": {"mode": "dynamic", "key_from": "partition_key"},
            "rate_limit_rps": 100,  # tests move fast; production: <= upstream cap
        },
        data_owner="data-platform",
    )

    # --- 4. In-memory blob stand-in ---
    class _InMemoryBlob:
        def __init__(self) -> None:
            self.store: dict[str, tuple[bytes, str]] = {}

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
            del sensitivity
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

    blob = _InMemoryBlob()
    ctx = IngestContext(log=logging.getLogger("api-crawl-cookbook"))

    # --- 5. Discover the partition, then materialize with the API mocked ---
    pattern = ApiCrawlPattern()
    [partition_spec] = pattern.discover_partitions(ingest, ctx)
    assert partition_spec.key == "2026-07"

    def _mock_rxclass_api() -> None:
        respx.get("https://rxnav.example/REST/rxclass/allClasses.json").respond(
            200, json={"classes": [{"classId": "A01"}, {"classId": "B02"}]}
        )
        respx.get(
            "https://rxnav.example/REST/rxclass/classMembers.json",
            params={"classId": "A01", "relaSource": "ATC"},
        ).respond(200, json={"rxcuis": [1191, 5640]})
        respx.get(
            "https://rxnav.example/REST/rxclass/classMembers.json",
            params={"classId": "B02", "relaSource": "ATC"},
        ).respond(200, json={"rxcuis": [7052]})

    with respx.mock:
        _mock_rxclass_api()
        results = materialize_with_manifest(pattern, ingest, partition_spec, blob, ctx)  # type: ignore[arg-type]

    # One NDJSON blob (the plan chose a single filename) + the manifest.
    [edges] = results
    assert edges.action == "uploaded"
    assert edges.path == "rxclass/2026-07/drug_class_edges.ndjson"

    lines = blob.store[edges.path][0].decode().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"class_id": "A01", "rxcui": 1191},
        {"class_id": "A01", "rxcui": 5640},
        {"class_id": "B02", "rxcui": 7052},
    ]

    # The manifest's audit block names the crawl plan + resolver, and
    # fields carry the resolver's release dict (snapshot_date is
    # available to FromIngestTemplate consumers via effective_from_field).
    from io import BytesIO

    manifest = IngestManifest.read_from(BytesIO(blob.store["rxclass/2026-07/_manifest.json"][0]))
    assert manifest.resolver["name"] == "_cookbook_rxclass"
    assert manifest.resolver["config"]["crawl_config"] == {"rela_source": "ATC"}
    assert manifest.fields["snapshot_date"] == "2026-07-01"

    # --- 6. Re-run: deterministic assembly -> byte-identical -> skipped ---
    with respx.mock:
        _mock_rxclass_api()
        second_run = materialize_with_manifest(pattern, ingest, partition_spec, blob, ctx)  # type: ignore[arg-type]

    assert all(r.action == "skipped" for r in second_run)
    # --- cookbook:end ---
