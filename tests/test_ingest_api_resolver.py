"""Tests for ``ApiResolverPattern``.

Network is mocked via respx; the resolver registry is stubbed to a
deterministic test resolver so the test surface focuses on the
pattern's contract -- not on UTS-specific behavior (which lives in
``test_uts_release_resolver.py``).

Covers:

- ``discover_partitions``: secret fetched via ctx.secrets, resolver
  called, single PartitionSpec emitted.
- ``materialize_partition``: secret fetched, URL resolved, payload
  streamed to tempfile, extracted via iterator, hash-compared, uploaded.
- Idempotency: a second materialization with the same content emits
  ``"skipped"``.
- Credential lifecycle: each call to ``materialize_partition`` re-fetches
  the secret (no caching across calls per the #216 decision).
- Missing ``ctx.secrets`` raises ``IngestResolutionError``.
- Missing ``key_from`` field on the resolver output raises with a
  clear error.
- api_key never appears in captured logs (via the redacting client).
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from typing import IO, Any, ClassVar, Literal
from unittest.mock import MagicMock

import pytest
import respx

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns.api_resolver import ApiResolverPattern
from moncpipelib.ingest.resolvers import (
    RESOLVERS,
    ReleaseResolver,
    ResolvedDownload,
    register_resolver,
)
from moncpipelib.ingest.types import IngestContext, PartitionSpec

# ---------------------------------------------------------------------------
# In-memory blob + secrets stand-ins
# ---------------------------------------------------------------------------


class FakeBlob:
    def __init__(self, preloaded: dict[str, str] | None = None) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}
        if preloaded:
            for path, sha in preloaded.items():
                self.blobs[path] = (b"<existing>", sha)

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
        # Per #239 the pattern streams uploads from a file handle, not bytes.
        body = data if isinstance(data, bytes) else data.read()
        self.blobs[path] = (body, sha256)


class FakeSecrets:
    """KV stand-in: returns a fixed value, counts get_secret calls."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.calls: list[str] = []

    def get_secret(self, name: str) -> str:
        self.calls.append(name)
        return self.value


# ---------------------------------------------------------------------------
# Stub resolver -- registered/restored per test
# ---------------------------------------------------------------------------


class _StubResolver:
    """Test resolver: returns a fixed release dict and the request URL.

    Captures the ``api_key`` it was called with so tests can assert it
    was forwarded correctly.
    """

    name: ClassVar[str] = "stub_resolver"
    discovery_requires_auth: ClassVar[bool] = True

    def __init__(self) -> None:
        self.api_keys_seen: list[str | None] = []

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del config, ctx
        self.api_keys_seen.append(api_key)
        return {
            "partition_key": "rel-2026-01",
            "release_version": "rel-2026-01",
            "download_url": "https://upstream.test/release.zip",
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del config, ctx
        self.api_keys_seen.append(api_key)
        # Embed the api_key as a query param to mirror the UTS
        # download endpoint shape (credential-less contracts pass None).
        suffix = f"&apiKey={api_key}" if api_key is not None else ""
        return ResolvedDownload(
            url=f"https://upstream.test/download?partition={partition_key}{suffix}",
            filename=None,
        )

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        # Default opt-out: tests that exercise historical override this.
        del api_key, config, ctx
        return []


@pytest.fixture
def stub_resolver() -> Any:
    """Register a fresh ``_StubResolver`` and restore the original on
    cleanup.  Mirrors the resolver-registry test pattern from PR 3."""
    original = RESOLVERS.get("stub_resolver")
    stub = _StubResolver()
    register_resolver(stub)  # type: ignore[arg-type]
    yield stub
    if original is not None:
        register_resolver(original)
    else:
        RESOLVERS.pop("stub_resolver", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _contract(
    *,
    sensitivity: Literal["public", "confidential", "phi"] = "confidential",
    extract: tuple[str, ...] = ("zip",),
    extract_filter: tuple[str, ...] = (),
    key_from: str = "release_version",
    with_credential: bool = True,
) -> IngestContract:
    pattern_config: dict[str, Any] = {
        "resolver": "stub_resolver",
        "resolver_config": {},
        "partition": {"mode": "dynamic", "key_from": key_from},
        "idempotency": "hash_compare",
        "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
    }
    if with_credential:
        pattern_config["credential"] = {"secret_name": "uts-api-key"}
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="umls-meta",
        sensitivity=sensitivity,
        pattern="api_resolver",
        prefix_template="umls/{partition_key}",
        extract=extract,
        strip_extensions=(),
        extract_filter=extract_filter,
        pattern_config=pattern_config,
        data_owner="data-platform",
        compliance_review="SECURITY.md#umls",
    )


def _ctx(
    secrets: FakeSecrets | None,
) -> IngestContext:
    return IngestContext(
        log=MagicMock(name="LoggingContext"),
        secrets=secrets,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# discover_partitions
# ---------------------------------------------------------------------------


def test_discover_partitions_emits_one_spec_for_current_release(
    stub_resolver: _StubResolver,
) -> None:
    secrets = FakeSecrets("api-key-value")
    contract = _contract()

    specs = ApiResolverPattern().discover_partitions(contract, _ctx(secrets))

    assert len(specs) == 1
    [spec] = specs
    assert spec.key == "rel-2026-01"
    assert spec.metadata["release_version"] == "rel-2026-01"
    assert spec.metadata["download_url"] == "https://upstream.test/release.zip"
    assert stub_resolver.api_keys_seen == ["api-key-value"]


def test_discover_partitions_uses_partition_key_from_field(
    stub_resolver: _StubResolver,
) -> None:
    """The ``partition.key_from`` field of the contract picks which
    field of the resolver's output becomes the partition key.  Here we
    use ``partition_key`` explicitly to verify the indirection."""
    del stub_resolver  # registered via fixture; unused in this test
    secrets = FakeSecrets("k")
    contract = _contract(key_from="partition_key")

    [spec] = ApiResolverPattern().discover_partitions(contract, _ctx(secrets))

    assert spec.key == "rel-2026-01"


def test_discover_partitions_missing_key_from_field_raises(
    stub_resolver: _StubResolver,
) -> None:
    del stub_resolver  # registered via fixture; unused in this test
    secrets = FakeSecrets("k")
    contract = _contract(key_from="release_date")  # not in stub output

    with pytest.raises(IngestResolutionError, match="release_date"):
        ApiResolverPattern().discover_partitions(contract, _ctx(secrets))


def test_discover_partitions_without_secrets_raises(
    stub_resolver: _StubResolver,
) -> None:
    del stub_resolver  # registered via fixture; unused in this test
    contract = _contract()

    with pytest.raises(IngestResolutionError, match="ctx.secrets"):
        ApiResolverPattern().discover_partitions(contract, _ctx(secrets=None))


def test_discover_partitions_no_credential_no_secrets_required(
    stub_resolver: _StubResolver,
) -> None:
    """Per #218: contracts that omit the ``credential`` block (e.g.
    calendar resolver) materialize without a ``KeyVaultSecretResource``;
    the resolver receives ``api_key=None``."""
    contract = _contract(with_credential=False)

    [spec] = ApiResolverPattern().discover_partitions(contract, _ctx(secrets=None))

    assert spec.key == "rel-2026-01"
    assert stub_resolver.api_keys_seen == [None]


# ---------------------------------------------------------------------------
# materialize_partition
# ---------------------------------------------------------------------------


@respx.mock
def test_materialize_streams_and_uploads(stub_resolver: _StubResolver) -> None:
    del stub_resolver  # registered via fixture; unused in this test
    payload = _zip_bytes({"data/MRCONSO.RRF": b"row\n"})
    respx.get("https://upstream.test/download").respond(200, content=payload)

    secrets = FakeSecrets("api-key-value")
    contract = _contract()
    blob = FakeBlob()
    spec = PartitionSpec(key="rel-2026-01")

    [result] = ApiResolverPattern().materialize_partition(
        contract,
        spec,
        blob,
        _ctx(secrets),  # type: ignore[arg-type]
    )

    assert result.action == "uploaded"
    assert result.path == "umls/rel-2026-01/data/MRCONSO.RRF"
    assert result.sha256 == hashlib.sha256(b"row\n").hexdigest()


@respx.mock
def test_materialize_sends_configured_user_agent(stub_resolver: _StubResolver) -> None:
    """``fetch.user_agent`` applies to the resolved-URL download (#413)."""
    del stub_resolver  # registered via fixture; unused in this test
    payload = _zip_bytes({"data/MRCONSO.RRF": b"row\n"})
    route = respx.get("https://upstream.test/download").respond(200, content=payload)

    contract = _contract()
    contract.pattern_config["fetch"]["user_agent"] = "ExampleOrgDataPlatform/1.0"

    ApiResolverPattern().materialize_partition(
        contract,
        PartitionSpec(key="rel-2026-01"),
        FakeBlob(),
        _ctx(FakeSecrets("api-key-value")),  # type: ignore[arg-type]
    )

    assert route.calls.last.request.headers["User-Agent"] == "ExampleOrgDataPlatform/1.0"


@respx.mock
def test_materialize_idempotent_skip_via_hash_compare(
    stub_resolver: _StubResolver,
) -> None:
    """A second materialization with unchanged content emits ``"skipped"``."""
    del stub_resolver
    file_data = b"unchanged\n"
    payload = _zip_bytes({"a.csv": file_data})
    respx.get("https://upstream.test/download").respond(200, content=payload)

    contract = _contract()
    secrets = FakeSecrets("k")
    known_sha = hashlib.sha256(file_data).hexdigest()
    blob = FakeBlob(preloaded={"umls/rel-2026-01/a.csv": known_sha})

    [result] = ApiResolverPattern().materialize_partition(
        contract,
        PartitionSpec(key="rel-2026-01"),
        blob,  # type: ignore[arg-type]
        _ctx(secrets),
    )

    assert result.action == "skipped"
    # The pre-existing payload bytes must not be replaced.
    assert blob.blobs["umls/rel-2026-01/a.csv"][0] == b"<existing>"


@respx.mock
def test_materialize_re_fetches_secret_per_call(
    stub_resolver: _StubResolver,
) -> None:
    """Per the #216 credential-lifecycle decision: every
    ``materialize_partition`` call hits ctx.secrets.get_secret again
    so a rotated KV value is picked up on the next tick."""
    del stub_resolver
    payload = _zip_bytes({"x.csv": b"x"})
    respx.get("https://upstream.test/download").respond(200, content=payload)

    contract = _contract()
    secrets = FakeSecrets("k")
    blob = FakeBlob()

    pattern = ApiResolverPattern()
    pattern.materialize_partition(contract, PartitionSpec(key="rel-2026-01"), blob, _ctx(secrets))  # type: ignore[arg-type]
    pattern.materialize_partition(contract, PartitionSpec(key="rel-2026-01"), blob, _ctx(secrets))  # type: ignore[arg-type]

    # 2 calls -> 2 fetches (no caching by the pattern).  Multiple
    # internal calls per materialize_partition (resolve_url may also
    # fetch) are also acceptable; the contract pinned by #216 is
    # "fetched per materialize_partition", not "exactly once".
    assert len(secrets.calls) >= 2
    assert all(name == "uts-api-key" for name in secrets.calls)


@respx.mock
def test_materialize_without_secrets_raises(
    stub_resolver: _StubResolver,
) -> None:
    del stub_resolver
    contract = _contract()
    blob = FakeBlob()

    with pytest.raises(IngestResolutionError, match="ctx.secrets"):
        ApiResolverPattern().materialize_partition(
            contract,
            PartitionSpec(key="rel-2026-01"),
            blob,  # type: ignore[arg-type]
            _ctx(secrets=None),
        )


@respx.mock
def test_materialize_no_credential_no_secrets_required(
    stub_resolver: _StubResolver,
) -> None:
    """Per #218: end-to-end materialize for a contract with no credential
    block succeeds without a ``KeyVaultSecretResource``."""
    payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://upstream.test/download").respond(200, content=payload)
    contract = _contract(with_credential=False)
    blob = FakeBlob()

    [result] = ApiResolverPattern().materialize_partition(
        contract,
        PartitionSpec(key="rel-2026-01"),
        blob,  # type: ignore[arg-type]
        _ctx(secrets=None),
    )

    assert result.action == "uploaded"
    assert result.path == "umls/rel-2026-01/data.csv"
    # api_key is None throughout for credential-less contracts.
    assert all(k is None for k in stub_resolver.api_keys_seen)


@respx.mock
def test_materialize_extracts_with_filter_recursively(
    stub_resolver: _StubResolver,
) -> None:
    """ADR-1 nested-filter behavior pipes through to ApiResolverPattern.

    Outer zip with one inner zip containing meta/ + otherks/ files;
    contract.extract_filter = ('meta/**',) keeps only the meta tree."""
    del stub_resolver
    inner = _zip_bytes(
        {
            "meta/MRCONSO.RRF": b"x",
            "otherks/SKIP.RRF": b"skip",
        }
    )
    outer = _zip_bytes({"2026AB-meta.nlm.zip": inner})
    respx.get("https://upstream.test/download").respond(200, content=outer)

    contract = _contract(extract=("zip", "zip"), extract_filter=("meta/**",))
    blob = FakeBlob()

    results = ApiResolverPattern().materialize_partition(
        contract,
        PartitionSpec(key="rel-2026-01"),
        blob,  # type: ignore[arg-type]
        _ctx(FakeSecrets("k")),
    )

    paths = {r.path for r in results}
    assert paths == {"umls/rel-2026-01/meta/MRCONSO.RRF"}


@respx.mock
def test_api_key_never_appears_in_captured_logs(
    stub_resolver: _StubResolver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The resolver embeds the api_key in the resolved download URL.
    The redacting client's transport hooks must ensure it does NOT
    appear in the audit log even though the URL contains it."""
    del stub_resolver
    payload = _zip_bytes({"x.csv": b"x"})
    respx.get("https://upstream.test/download").respond(200, content=payload)

    secret = "redact-me-leak-canary"
    contract = _contract()

    with caplog.at_level(logging.INFO, logger="moncpipelib.ingest._http"):
        ApiResolverPattern().materialize_partition(
            contract,
            PartitionSpec(key="rel-2026-01"),
            FakeBlob(),  # type: ignore[arg-type]
            _ctx(FakeSecrets(secret)),
        )

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in captured


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_api_resolver_pattern_satisfies_ingest_pattern_protocol() -> None:
    from moncpipelib.ingest.patterns import IngestPattern

    assert isinstance(ApiResolverPattern(), IngestPattern)


def test_stub_resolver_satisfies_release_resolver_protocol() -> None:
    """Sanity: the test-only stub also satisfies the Protocol so we know
    the assertion-style checks elsewhere stay valid."""
    assert isinstance(_StubResolver(), ReleaseResolver)


# ---------------------------------------------------------------------------
# discover_partitions historical-first dispatch (per #228)
# ---------------------------------------------------------------------------


class _HistoricalStubResolver:
    """Stub resolver that returns N historical releases.

    Used to exercise the discover_partitions historical-first dispatch
    path independent of UTS-specific behavior.
    """

    name: ClassVar[str] = "historical_stub_resolver"
    discovery_requires_auth: ClassVar[bool] = True

    def __init__(self, historical: list[dict[str, Any]]) -> None:
        self._historical = historical
        self.current_release_calls = 0
        self.historical_release_calls = 0

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del api_key, config, ctx
        self.current_release_calls += 1
        return {
            "partition_key": "current-only",
            "release_version": "current-only",
            "download_url": "https://upstream.test/current.zip",
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del api_key, config, ctx
        return ResolvedDownload(
            url=f"https://upstream.test/download?partition={partition_key}",
            filename=None,
        )

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del api_key, config, ctx
        self.historical_release_calls += 1
        return list(self._historical)


@pytest.fixture
def historical_resolver() -> Any:
    """Register/restore a historical-aware stub resolver."""
    original = RESOLVERS.get("historical_stub_resolver")
    yield  # tests construct + register their own instance
    if original is not None:
        register_resolver(original)
    else:
        RESOLVERS.pop("historical_stub_resolver", None)


def _historical_contract() -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="hist-source",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="hist/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "historical_stub_resolver",
            "resolver_config": {},
            "credential": {"secret_name": "uts-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#hist",
    )


def test_discover_partitions_emits_one_spec_per_historical_release(
    historical_resolver: None,
) -> None:
    del historical_resolver
    stub = _HistoricalStubResolver(
        [
            {
                "partition_key": "2024AA",
                "release_version": "2024AA",
                "download_url": "https://upstream.test/2024AA.zip",
            },
            {
                "partition_key": "2024AB",
                "release_version": "2024AB",
                "download_url": "https://upstream.test/2024AB.zip",
            },
        ]
    )
    register_resolver(stub)  # type: ignore[arg-type]

    specs = ApiResolverPattern().discover_partitions(_historical_contract(), _ctx(FakeSecrets("k")))

    assert [s.key for s in specs] == ["2024AA", "2024AB"]
    assert stub.historical_release_calls == 1
    # Pattern should NOT call current_release when historical returns >=1 spec.
    assert stub.current_release_calls == 0


def test_discover_partitions_falls_back_to_current_when_historical_empty(
    historical_resolver: None,
) -> None:
    """Per #228: when historical_release returns [], pattern uses
    current_release.  Preserves v0.27 behavior for resolvers that opt
    out of historical (e.g. CalendarReleaseResolver)."""
    del historical_resolver
    stub = _HistoricalStubResolver([])
    register_resolver(stub)  # type: ignore[arg-type]

    [spec] = ApiResolverPattern().discover_partitions(
        _historical_contract(), _ctx(FakeSecrets("k"))
    )

    assert spec.key == "current-only"
    assert stub.historical_release_calls == 1
    assert stub.current_release_calls == 1


# ---------------------------------------------------------------------------
# discover_requires_auth opt-out (per #253)
# ---------------------------------------------------------------------------


class _UnauthDiscoveryResolver:
    """Stub that opts out of discovery-time auth via the #253 flag.

    Records the ``api_key`` value the pattern hands to discovery
    methods so tests can assert ``None`` is forwarded even when the
    contract declares a ``credential`` block.
    """

    name: ClassVar[str] = "unauth_discovery_resolver"
    discovery_requires_auth: ClassVar[bool] = False

    def __init__(self) -> None:
        self.discovery_api_keys: list[str | None] = []
        self.materialize_api_keys: list[str | None] = []

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del config, ctx
        self.discovery_api_keys.append(api_key)
        return {
            "partition_key": "rel-2026-01",
            "release_version": "rel-2026-01",
            "download_url": "https://upstream.test/release.zip",
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del config, ctx
        self.materialize_api_keys.append(api_key)
        suffix = f"&apiKey={api_key}" if api_key is not None else ""
        return ResolvedDownload(
            url=f"https://upstream.test/download?partition={partition_key}{suffix}",
            filename=None,
        )

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del config, ctx
        self.discovery_api_keys.append(api_key)
        return []


@pytest.fixture
def unauth_discovery_resolver() -> Any:
    original = RESOLVERS.get("unauth_discovery_resolver")
    stub = _UnauthDiscoveryResolver()
    register_resolver(stub)  # type: ignore[arg-type]
    yield stub
    if original is not None:
        register_resolver(original)
    else:
        RESOLVERS.pop("unauth_discovery_resolver", None)


def _unauth_contract() -> IngestContract:
    """Contract that declares a credential block but points at the
    unauth-discovery stub.  Mirrors the UTS shape: KV-backed credential
    needed at materialize time, but discovery does not need it."""
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="unauth-disc-source",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="src/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "unauth_discovery_resolver",
            "resolver_config": {},
            "credential": {"secret_name": "uts-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#unauth",
    )


def test_discover_skips_secret_fetch_when_resolver_opts_out(
    unauth_discovery_resolver: _UnauthDiscoveryResolver,
) -> None:
    """A resolver with ``discovery_requires_auth = False`` causes the
    pattern to skip ``ctx.secrets.get_secret`` during ``discover_partitions``
    even though the contract declares a ``credential`` block (#253)."""
    secrets = FakeSecrets("api-key-value")

    [spec] = ApiResolverPattern().discover_partitions(_unauth_contract(), _ctx(secrets))

    assert spec.key == "rel-2026-01"
    # Pattern must not have consulted ctx.secrets at all.
    assert secrets.calls == []
    # Resolver received api_key=None for both historical_release (which
    # returned []) and the current_release fallback.
    assert unauth_discovery_resolver.discovery_api_keys == [None, None]


def test_discover_skips_secret_fetch_works_without_secrets_resource(
    unauth_discovery_resolver: _UnauthDiscoveryResolver,
) -> None:
    """The whole point of the opt-out: a daemon with no
    ``KeyVaultSecretResource`` wired on ``ctx.secrets`` can still run
    discovery for an opt-out resolver, even when the contract declares
    a credential block."""
    del unauth_discovery_resolver

    [spec] = ApiResolverPattern().discover_partitions(_unauth_contract(), _ctx(secrets=None))

    assert spec.key == "rel-2026-01"


def test_default_resolver_still_fetches_secret_during_discovery(
    stub_resolver: _StubResolver,
) -> None:
    """Resolvers that don't declare ``discovery_requires_auth`` keep
    the prior behavior: pattern calls ``_fetch_api_key`` and forwards
    the value to discovery methods.  Guards against regressions in the
    default (#253)."""
    secrets = FakeSecrets("api-key-value")
    contract = _contract()

    ApiResolverPattern().discover_partitions(contract, _ctx(secrets))

    # Default-True path: ctx.secrets IS consulted, value reaches resolver.
    assert secrets.calls == ["uts-api-key"]
    assert stub_resolver.api_keys_seen == ["api-key-value"]


class _MisdeclaredOptOutResolver:
    """Resolver that wrongly declares ``discovery_requires_auth = False``
    but whose ``historical_release`` actually requires the api_key.
    Used to assert that misdeclaration fails fast and visibly."""

    name: ClassVar[str] = "misdeclared_optout_resolver"
    discovery_requires_auth: ClassVar[bool] = False

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del config, ctx
        if api_key is None:
            raise IngestResolutionError("misdeclared: current_release needs api_key")
        return {
            "partition_key": "x",
            "release_version": "x",
            "download_url": "https://upstream.test/x.zip",
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del api_key, config, ctx
        return ResolvedDownload(url=f"https://upstream.test/{partition_key}", filename=None)

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del config, ctx
        if api_key is None:
            raise IngestResolutionError("misdeclared: historical_release needs api_key")
        return []


def test_misdeclared_optout_surfaces_resolver_failure() -> None:
    """If a resolver lies about not needing auth at discovery, the
    failure must come from the resolver -- not be silently swallowed
    by the pattern.  Locks in the "fail loud, fix the declaration"
    posture from the #253 review."""
    original = RESOLVERS.get("misdeclared_optout_resolver")
    register_resolver(_MisdeclaredOptOutResolver())  # type: ignore[arg-type]
    try:
        contract = IngestContract(
            source_id="11111111-1111-1111-1111-111111111111",
            source_name="misdeclared",
            sensitivity="confidential",
            pattern="api_resolver",
            prefix_template="m/{partition_key}",
            extract=("zip",),
            strip_extensions=(),
            pattern_config={
                "resolver": "misdeclared_optout_resolver",
                "resolver_config": {},
                "credential": {"secret_name": "uts-api-key"},
                "partition": {"mode": "dynamic", "key_from": "release_version"},
                "idempotency": "hash_compare",
                "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
            },
            data_owner="data-platform",
            compliance_review="SECURITY.md#misdeclared",
        )
        # ctx.secrets is populated -- proves the failure is from the
        # resolver, not from the pattern's missing-secrets guard.
        with pytest.raises(IngestResolutionError, match="historical_release needs api_key"):
            ApiResolverPattern().discover_partitions(contract, _ctx(FakeSecrets("k")))
    finally:
        if original is not None:
            register_resolver(original)
        else:
            RESOLVERS.pop("misdeclared_optout_resolver", None)


@respx.mock
def test_materialize_still_fetches_api_key_for_optout_resolver(
    unauth_discovery_resolver: _UnauthDiscoveryResolver,
) -> None:
    """The opt-out gates discovery only.  ``materialize_partition``
    must keep fetching the api_key because :meth:`resolve_url` typically
    needs it (e.g. UTS ``/download``).  Regression guard for #253."""
    payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://upstream.test/download").respond(200, content=payload)
    secrets = FakeSecrets("api-key-value")

    [result] = ApiResolverPattern().materialize_partition(
        _unauth_contract(),
        PartitionSpec(key="rel-2026-01"),
        FakeBlob(),  # type: ignore[arg-type]
        _ctx(secrets),
    )

    assert result.action == "uploaded"
    # ctx.secrets IS consulted at materialize time.
    assert secrets.calls == ["uts-api-key"]
    # The fetched value reaches resolve_url.
    assert unauth_discovery_resolver.materialize_api_keys == ["api-key-value"]


def test_uts_resolver_declares_unauth_discovery() -> None:
    """Pin the UTS resolver's opt-out so a future refactor that drops
    the flag is caught here, not by a daemon-pod sensor failure."""
    from moncpipelib.ingest.resolvers.uts import UtsReleaseResolver

    assert UtsReleaseResolver.discovery_requires_auth is False


def test_protocol_default_is_unauth_required() -> None:
    """The Protocol's default for ``discovery_requires_auth`` is True --
    safe-by-default so a resolver that doesn't explicitly opt out keeps
    the prior 'fetch the api_key at discovery' behavior."""
    assert ReleaseResolver.discovery_requires_auth is True


class _UndeclaredFlagResolver:
    """Resolver that omits ``discovery_requires_auth`` entirely.

    Models a third-party / pre-#253 Protocol implementation.  The
    pattern's ``getattr(..., True)`` fallback should still treat it as
    'needs auth at discovery' so prior behavior is preserved.
    """

    name: ClassVar[str] = "undeclared_flag_resolver"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del config, ctx
        return {
            "partition_key": "x",
            "release_version": "x",
            "download_url": "https://upstream.test/x.zip",
            "api_key_seen": api_key,
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del api_key, config, ctx
        return ResolvedDownload(url=f"https://upstream.test/{partition_key}", filename=None)

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del api_key, config, ctx
        return []


def test_undeclared_flag_falls_back_to_requiring_auth() -> None:
    """A resolver that does not declare ``discovery_requires_auth`` is
    treated as if it had declared ``True``: the pattern fetches the
    api_key from ctx.secrets and forwards it to discovery methods.

    Locks in the ``getattr(resolver, 'discovery_requires_auth', True)``
    safety net for resolvers predating #253."""
    original = RESOLVERS.get("undeclared_flag_resolver")
    register_resolver(_UndeclaredFlagResolver())  # type: ignore[arg-type]
    try:
        contract = IngestContract(
            source_id="11111111-1111-1111-1111-111111111111",
            source_name="undeclared",
            sensitivity="confidential",
            pattern="api_resolver",
            prefix_template="u/{partition_key}",
            extract=("zip",),
            strip_extensions=(),
            pattern_config={
                "resolver": "undeclared_flag_resolver",
                "resolver_config": {},
                "credential": {"secret_name": "uts-api-key"},
                "partition": {"mode": "dynamic", "key_from": "release_version"},
                "idempotency": "hash_compare",
                "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
            },
            data_owner="data-platform",
            compliance_review="SECURITY.md#undeclared",
        )
        secrets = FakeSecrets("api-key-value")
        [spec] = ApiResolverPattern().discover_partitions(contract, _ctx(secrets))
        assert spec.metadata["api_key_seen"] == "api-key-value"
        assert secrets.calls == ["uts-api-key"]
    finally:
        if original is not None:
            register_resolver(original)
        else:
            RESOLVERS.pop("undeclared_flag_resolver", None)


# ---------------------------------------------------------------------------
# partition_metadata (per #256)
# ---------------------------------------------------------------------------


class _PartitionMetadataStubResolver:
    """Stub resolver that returns a multi-entry ``historical_release``.

    Used to verify ``ApiResolverPattern.partition_metadata`` finds the
    matching release dict across historical entries -- the bug from
    #256 that surfaced when RxNorm materialized a non-current
    partition.
    """

    name: ClassVar[str] = "partition_metadata_stub"
    discovery_requires_auth: ClassVar[bool] = False

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del api_key, config, ctx
        return {
            "partition_key": "2026-04-06",
            "release_version": "2026-04-06",
            "release_date": "2026-04-06",
            "download_url": "https://upstream.test/2026-04-06.zip",
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del api_key, config, ctx
        return ResolvedDownload(url=f"https://upstream.test/{partition_key}.zip", filename=None)

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del api_key, config, ctx
        return [
            {
                "partition_key": "2025-12-01",
                "release_version": "2025-12-01",
                "release_date": "2025-12-01",
                "download_url": "https://upstream.test/2025-12-01.zip",
            },
            {
                "partition_key": "2026-04-06",
                "release_version": "2026-04-06",
                "release_date": "2026-04-06",
                "download_url": "https://upstream.test/2026-04-06.zip",
            },
        ]


@pytest.fixture
def partition_metadata_stub() -> Any:
    original = RESOLVERS.get("partition_metadata_stub")
    register_resolver(_PartitionMetadataStubResolver())  # type: ignore[arg-type]
    yield
    if original is not None:
        register_resolver(original)
    else:
        RESOLVERS.pop("partition_metadata_stub", None)


def _partition_metadata_contract() -> IngestContract:
    return IngestContract(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="partition-metadata-stub-source",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="hist/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "partition_metadata_stub",
            "resolver_config": {},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#partition-metadata-stub",
    )


def test_partition_metadata_returns_historical_match(
    partition_metadata_stub: None,
) -> None:
    """Per #256: ``partition_metadata`` must find the release dict for a
    historical (non-current) partition_key, mirroring the RxNorm
    Phase 2a regression in data-platform#613."""
    del partition_metadata_stub
    contract = _partition_metadata_contract()

    fields = ApiResolverPattern().partition_metadata(contract, "2025-12-01", _ctx(secrets=None))

    assert fields == {
        "partition_key": "2025-12-01",
        "release_version": "2025-12-01",
        "release_date": "2025-12-01",
        "download_url": "https://upstream.test/2025-12-01.zip",
    }


def test_partition_metadata_returns_current_release_when_historical_empty(
    stub_resolver: _StubResolver,
) -> None:
    """Calendar-shaped resolvers return ``[]`` from ``historical_release``;
    ``partition_metadata`` must fall back to ``current_release`` so the
    manifest's ``fields`` still hydrates a Calendar consumer's
    ``effective_from_field`` (snapshot_date)."""
    del stub_resolver
    contract = _contract(with_credential=False)

    fields = ApiResolverPattern().partition_metadata(contract, "rel-2026-01", _ctx(secrets=None))

    assert fields["release_version"] == "rel-2026-01"
    assert fields["download_url"] == "https://upstream.test/release.zip"


def test_partition_metadata_returns_empty_when_partition_not_found(
    partition_metadata_stub: None,
) -> None:
    """Defensive: a partition_key not present in either historical or
    current -> ``{}``.  The dispatcher then falls back to
    ``partition_spec.metadata`` (preserving back-compat)."""
    del partition_metadata_stub
    contract = _partition_metadata_contract()

    fields = ApiResolverPattern().partition_metadata(contract, "1999-01-01", _ctx(secrets=None))

    assert fields == {}


def test_partition_metadata_skips_kv_when_discovery_requires_auth_false(
    partition_metadata_stub: None,
) -> None:
    """Mirror the #253 opt-out at the partition_metadata path: a resolver
    with ``discovery_requires_auth = False`` should not force a KV fetch
    here either, since it shares the same release-listing endpoint as
    discovery."""
    del partition_metadata_stub
    contract = _partition_metadata_contract()
    secrets = FakeSecrets("should-not-be-fetched")

    ApiResolverPattern().partition_metadata(contract, "2025-12-01", _ctx(secrets=secrets))

    assert secrets.calls == []


# ---------------------------------------------------------------------------
# Non-archive payload filename precedence chain (#270)
# ---------------------------------------------------------------------------


class _NonArchiveStubResolver:
    """Resolver stub for non-archive precedence tests.

    Configurable per-instance so a single test can dial the
    ``ResolvedDownload.filename`` hint independently of the URL.
    """

    name: ClassVar[str] = "non_archive_stub_resolver"
    discovery_requires_auth: ClassVar[bool] = False

    def __init__(self, *, url: str, filename: str | None) -> None:
        self._url = url
        self._filename = filename

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def current_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        del api_key, config, ctx
        return {
            "partition_key": "rel-2026-01",
            "release_version": "rel-2026-01",
            "download_url": self._url,
        }

    def resolve_url(
        self,
        api_key: str | None,
        partition_key: str,
        config: dict[str, Any],
        ctx: Any,
    ) -> ResolvedDownload:
        del api_key, partition_key, config, ctx
        return ResolvedDownload(url=self._url, filename=self._filename)

    def historical_release(
        self, api_key: str | None, config: dict[str, Any], ctx: Any
    ) -> list[dict[str, Any]]:
        del api_key, config, ctx
        return []


def _non_archive_api_resolver_contract(
    *, payload_filename_template: str | None = None
) -> IngestContract:
    """Contract using the non-archive stub resolver."""
    return IngestContract(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="non-archive-source",
        sensitivity="public",
        pattern="api_resolver",
        prefix_template="src/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "resolver": "non_archive_stub_resolver",
            "resolver_config": {},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
        },
        payload_filename_template=payload_filename_template,
    )


@pytest.fixture
def _non_archive_resolver() -> Any:
    """Register/restore the non-archive stub resolver per test."""
    original = RESOLVERS.get("non_archive_stub_resolver")
    yield
    RESOLVERS.pop("non_archive_stub_resolver", None)
    if original is not None:
        register_resolver(original)


@respx.mock
def test_api_resolver_non_archive_uses_resolver_filename_hint(
    _non_archive_resolver: None,
) -> None:
    """Resolver hint wins over Content-Disposition + URL basename
    (no template set).  Mirrors the UTS shape where the resolver knows
    the upstream basename even though the proxied download URL hides
    it behind ``apiKey=...&url=...``."""
    register_resolver(
        _NonArchiveStubResolver(
            url="https://upstream.test/proxied?token=abc",
            filename="2026AA-rxnorm-full.zip",
        )
    )
    respx.get("https://upstream.test/proxied").respond(
        200,
        content=b"x",
        headers={"Content-Disposition": 'attachment; filename="proxied.bin"'},
    )

    contract = _non_archive_api_resolver_contract()
    blob: Any = MagicMock()
    blob.read_sha256_metadata.return_value = None
    blob.upload.return_value = None

    spec = PartitionSpec(key="rel-2026-01", metadata={})
    ctx = _ctx(secrets=None)

    ApiResolverPattern().materialize_partition(contract, spec, blob, ctx)

    upload_paths = [call.args[1] for call in blob.upload.call_args_list]
    assert upload_paths == ["src/rel-2026-01/2026AA-rxnorm-full.zip"]


@respx.mock
def test_api_resolver_non_archive_template_overrides_resolver_hint(
    _non_archive_resolver: None,
) -> None:
    """``payload_filename_template`` wins over a resolver-supplied hint;
    contract author intent dominates."""
    register_resolver(
        _NonArchiveStubResolver(url="https://upstream.test/u", filename="upstream-name.zip")
    )
    respx.get("https://upstream.test/u").respond(200, content=b"x")

    contract = _non_archive_api_resolver_contract(
        payload_filename_template="{source_name}_{partition_key}.zip"
    )
    blob: Any = MagicMock()
    blob.read_sha256_metadata.return_value = None
    blob.upload.return_value = None

    spec = PartitionSpec(key="rel-2026-01", metadata={})
    ApiResolverPattern().materialize_partition(contract, spec, blob, _ctx(secrets=None))

    upload_paths = [call.args[1] for call in blob.upload.call_args_list]
    assert upload_paths == ["src/rel-2026-01/non-archive-source_rel-2026-01.zip"]


@respx.mock
def test_api_resolver_archive_unaffected_by_precedence_chain(
    stub_resolver: _StubResolver,
) -> None:
    """``extract: [zip]`` ignores the precedence chain entirely; archive
    members keep their in-archive filenames."""
    del stub_resolver
    payload = _zip_bytes({"meta/data.csv": b"hello"})
    respx.get("https://upstream.test/download").respond(200, content=payload)

    contract = IngestContract(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="archive-src",
        sensitivity="public",
        pattern="api_resolver",
        prefix_template="src/{partition_key}",
        extract=("zip",),  # archive contract
        strip_extensions=(),
        pattern_config={
            "resolver": "stub_resolver",
            "resolver_config": {},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
        },
        payload_filename_template="OVERRIDE.zip",  # MUST be ignored
    )
    blob: Any = MagicMock()
    blob.read_sha256_metadata.return_value = None
    blob.upload.return_value = None

    spec = PartitionSpec(key="rel-2026-01", metadata={})
    ApiResolverPattern().materialize_partition(contract, spec, blob, _ctx(secrets=None))

    upload_paths = [call.args[1] for call in blob.upload.call_args_list]
    assert upload_paths == ["src/rel-2026-01/meta/data.csv"]
