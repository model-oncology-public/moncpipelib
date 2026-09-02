"""Cookbook tests for the ``api_resolver`` ingest pattern.

Demonstrates the Phase 2 dynamic-partition flow end-to-end:

- Register a custom :class:`ReleaseResolver` that returns a synthetic
  release dict instead of calling a real upstream API (so the example
  runs in CI without credentials).
- Wire a :class:`KeyVaultSecretResource` stand-in plus an in-memory
  blob.
- Construct an ``api_resolver`` :class:`IngestContract` with a
  ``partition.mode: dynamic`` block and a downstream
  :class:`FromIngestTemplate` source that hydrates
  ``effective_from`` from the manifest.
- Call :func:`materialize_with_manifest` -- the canonical entry point
  as of v0.26.0 -- and observe the per-partition manifest land.
- Resolve the landed file via
  :func:`resolve_source_for_partition` (the manifest-reader branch).
- Re-run to demonstrate ``hash_compare`` idempotency.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.
"""

from __future__ import annotations

from typing import Any

import pytest

from moncpipelib.ingest.resolvers import RESOLVERS, ResolvedDownload, register_resolver


@pytest.fixture
def cookbook_resolver_registered() -> Any:
    """Register / restore the cookbook's stub resolver.

    Yields nothing; the fixture's job is purely registry teardown.
    """
    name = "_cookbook_release"
    original = RESOLVERS.get(name)
    yield
    if original is not None:
        register_resolver(original)
    else:
        RESOLVERS.pop(name, None)


@pytest.mark.cookbook(
    title="Land an authenticated source with api_resolver + dynamic partitions",
    description=(
        "Phase 2 of the universal blob-landing ingest boundary. The "
        "``api_resolver`` pattern resolves the download URL at fetch time "
        "via an authenticated API call. This example registers a stub "
        ":class:`ReleaseResolver` so it runs in CI without credentials, "
        "builds an ``api_resolver`` contract, calls "
        "``materialize_with_manifest`` to land data + write the per-partition "
        "manifest atomically, and then resolves the landed blob via the "
        "``FromIngestTemplate`` consumer branch (the manifest-reader path). "
        "Real pipelines register a production :class:`ReleaseResolver` and "
        "configure :class:`KeyVaultSecretResource` against the vault."
    ),
    category="ingest",
)
def test_cookbook_api_resolver_roundtrip(cookbook_resolver_registered: None) -> None:
    del cookbook_resolver_registered
    # --- cookbook:start ---
    import io
    import logging
    import zipfile
    from collections.abc import Iterator
    from typing import IO, ClassVar

    import respx

    from moncpipelib.contracts.models import (
        ContractCorpus,
        DataSource,
        FromIngestTemplate,
        IngestContract,
    )
    from moncpipelib.ingest import (
        ApiResolverPattern,
        BlobRef,
        IngestContext,
        materialize_with_manifest,
        register_resolver,
        resolve_source_for_partition,
    )

    # --- 1. Register a stub resolver so the example doesn't hit a real API ---
    # In production, register a :class:`ReleaseResolver` implementation that
    # talks to your upstream (returning the current release + a download URL).
    # For documentation / testing, a tiny stub returns a fixed release without
    # any network I/O.
    class _CookbookResolver:
        name: ClassVar[str] = "_cookbook_release"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def current_release(self, api_key: str, config: dict[str, Any], ctx: Any) -> dict[str, Any]:
            del api_key, config, ctx
            return {
                "partition_key": "v1",
                "release_version": "v1",
                "release_date": "2024-01-01",
                "download_url": "https://upstream.example/release.zip",
            }

        def resolve_url(
            self,
            api_key: str,
            partition_key: str,
            config: dict[str, Any],
            ctx: Any,
        ) -> ResolvedDownload:
            del partition_key, config, ctx
            # api_key is embedded as a query param; materializers fetch via
            # the redacting client so this URL never appears in transport logs.
            # ResolvedDownload.filename is None: the cookbook stub doesn't
            # know a semantic upstream filename.  Production resolvers (e.g.
            # UtsReleaseResolver) surface the upstream basename so the
            # non-archive payload chain can land it under that name (#270).
            return ResolvedDownload(
                url=f"https://upstream.example/release.zip?apiKey={api_key}",
                filename=None,
            )

        def historical_release(
            self, api_key: str, config: dict[str, Any], ctx: Any
        ) -> list[dict[str, Any]]:
            # This stub doesn't surface historical releases; returning []
            # tells ApiResolverPattern to fall back to current_release.
            # Production resolvers (e.g. UtsReleaseResolver) implement this
            # against their upstream's history endpoint.
            del api_key, config, ctx
            return []

    register_resolver(_CookbookResolver())  # type: ignore[arg-type]

    # --- 2. Declare the api_resolver ingest contract ---
    # In production, this is loaded from a *.ingest.yaml file via
    # load_ingest_contract.  Inline-constructed here for the example.
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="cookbook-api",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="cookbook/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "_cookbook_release",
            "resolver_config": {},
            "credential": {"secret_name": "cookbook-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#cookbook-api",
    )

    # --- 3. Declare the downstream source (FromIngestTemplate) ---
    # ``periods.mode: from_ingest`` says "every partition the ingest
    # produces becomes a downstream period".  The template's
    # ``effective_from_field`` names the manifest field that hydrates
    # ``effective_from`` per partition.
    source = DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="cookbook-api-extract",
        periods=FromIngestTemplate(
            source="data/records.csv",
            effective_from_field="release_date",
        ),
        ingest_source="cookbook-api",
    )

    corpus = ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )

    # --- 4. In-memory stand-ins for blob + secrets ---
    # Real pipelines configure BlobStorageResource and
    # KeyVaultSecretResource in their code location.
    class _InMemoryBlob:
        def __init__(self) -> None:
            self.store: dict[str, tuple[bytes, str]] = {}

        def list(self, sensitivity: str, prefix: str) -> list[str]:
            del sensitivity
            return [p for p in self.store if p.startswith(prefix)]

        def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
            # Lazy iterator (#246) -- consumers prefer this for large prefixes.
            del sensitivity
            return (p for p in self.store if p.startswith(prefix))

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
            del sensitivity
            # The pattern streams uploads from a file handle for large
            # members (#239); accept either bytes or IO[bytes] in the fake.
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.store

        def download(self, sensitivity: str, path: str) -> bytes:
            del sensitivity
            return self.store[path][0]

        def stream(self, sensitivity: str, path: str) -> IO[bytes]:
            # Forward-only file-like for the streaming manifest read
            # path (#241 / #243).
            del sensitivity
            from io import BytesIO

            return BytesIO(self.store[path][0])

    class _InMemorySecrets:
        def __init__(self, value: str) -> None:
            self._value = value

        def get_secret(self, name: str) -> str:
            del name
            return self._value

    blob = _InMemoryBlob()
    secrets = _InMemorySecrets("super-secret-cookbook-key")

    # --- 5. Build the IngestContext ---
    # api_resolver requires ctx.secrets; a missing secrets resource
    # raises IngestResolutionError.
    ctx = IngestContext(
        log=logging.getLogger("api-resolver-cookbook"),
        secrets=secrets,  # type: ignore[arg-type]
    )

    # --- 6. Discover the current partition (so metadata flows into the manifest) ---
    # In production, this happens inside ``build_discovery_sensor`` --
    # the sensor calls ``discover_partitions`` at every tick and adds new
    # keys to the dynamic-partitions registry.  Here we call it directly
    # so the resolver's full output (release_date, release_version, ...)
    # ends up in ``partition_spec.metadata`` -- which the dispatcher
    # writes verbatim into ``manifest.fields`` so the
    # ``FromIngestTemplate`` consumer branch can hydrate
    # ``effective_from`` from it.
    pattern = ApiResolverPattern()
    [partition_spec] = pattern.discover_partitions(ingest, ctx)
    assert partition_spec.key == "v1"
    assert partition_spec.metadata["release_date"] == "2024-01-01"

    # --- 7. Materialize via materialize_with_manifest ---
    def _zip_bytes(files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue()

    payload = _zip_bytes({"data/records.csv": b"id,value\n1,a\n"})

    with respx.mock:
        respx.get("https://upstream.example/release.zip").respond(200, content=payload)

        results = materialize_with_manifest(
            pattern,
            ingest,
            partition_spec,
            blob,  # type: ignore[arg-type]
            ctx,
        )

    # First run uploads the data + writes the manifest.
    assert all(r.action == "uploaded" for r in results)
    assert "cookbook/v1/data/records.csv" in blob.store
    assert "cookbook/v1/_manifest.json" in blob.store

    # --- 8. Resolve via the FromIngestTemplate branch (manifest reader) ---
    # The resolver loads {prefix}/_manifest.json, validates the version,
    # checks that ``release_date`` is populated in manifest.fields, and
    # returns a BlobRef for the rendered glob.
    [ref] = resolve_source_for_partition(
        source,
        partition_key="v1",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    )
    assert isinstance(ref, BlobRef)
    assert ref.path == "cookbook/v1/data/records.csv"
    assert ref.sensitivity == "confidential"

    # --- 9. Re-materialize: idempotent skip via hash_compare ---
    with respx.mock:
        respx.get("https://upstream.example/release.zip").respond(200, content=payload)
        second_run = materialize_with_manifest(
            pattern,
            ingest,
            partition_spec,
            blob,  # type: ignore[arg-type]
            ctx,
        )

    # Data file is unchanged -> "skipped"; the manifest is rewritten
    # on every successful run so the materialized_at timestamp stays
    # current.
    data_results = [r for r in second_run if r.path.endswith(".csv")]
    assert all(r.action == "skipped" for r in data_results)
    # --- cookbook:end ---
