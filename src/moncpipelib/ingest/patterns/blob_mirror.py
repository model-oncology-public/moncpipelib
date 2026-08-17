"""``blob_mirror`` ingest pattern (#437).

Mirrors objects from a foreign Azure Storage / ADLS Gen2 source (a
partner-owned account in another tenant) into our sensitivity-scoped
landing boundary, so the single-ingress audit trail and per-partition
``_manifest.json`` hold for blob-sourced feeds -- the Azure-blob analogue
of :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`.

Driving use case: the Trilliant Health ``visits_oncology`` feed
(``docs/migrations/20260717_436-439-foreign-blob-parquet-ingest.md``),
delivered as immutable monthly ``YYYYMM`` snapshots of N snappy-parquet
parts per asset.

Flow per partition:

1. Build a :class:`~moncpipelib.resources.foreign_blob.ForeignBlobSource`
   from the contract's ``blob_mirror`` config plus an SP secret pulled
   via ``ctx.secrets`` (design decision D1: the pattern constructs the
   source at materialize time -- there is no injection slot for a
   foreign read source on ``IngestContext``).
2. List the source objects under the rendered ``object_prefix``, keep
   those matching ``object_glob`` and not matching any ``exclude_globs``
   (the writer meta-files ``_committed_*`` / ``_started_*`` / ``_SUCCESS``
   are excluded this way).
3. Mirror each object into our container under the contract prefix.

Idempotency is **etag-compare** (design decision D3), not sha256: a
sha256 skip would have to download the whole object every run just to
hash it (their parts run to 100M-13B rows).  On upload we stamp the
source object's etag into our blob metadata (``source_etag``); on a
re-run we HEAD-compare the source etag against the stored value and skip
without downloading.  Snapshots are immutable per Trilliant's contract,
so the etag is a stable identity.  The sha256 integrity stamp is still
written (computed in the same streaming pass as the upload) for the
manifest and downstream consumers.

Peak memory is bounded by the copy chunk size regardless of object size
(CLAUDE.md I/O-at-boundaries): the object streams source -> hashing
tempfile -> our upload; it is never materialized whole.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import IO, TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from moncpipelib.ingest._hashing import hashing_tempfile
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.types import IngestResult, PartitionSpec
from moncpipelib.resources.foreign_blob import ForeignBlobSource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.types import IngestContext
    from moncpipelib.resources.blob import BlobStorageResource


_SOURCE_ETAG_KEY: str = "source_etag"
"""Blob metadata key holding the mirrored object's foreign-source etag.

Written alongside ``sha256`` on upload; read HEAD-only on re-run to skip
an unchanged object without downloading it (design decision D3)."""

_COPY_CHUNK_BYTES: int = 8 * 1024 * 1024
"""Read size when copying source -> tempfile.  Peak heap tracks this, not
the object size."""


@runtime_checkable
class ForeignBlobReader(Protocol):
    """The read surface ``blob_mirror`` needs from a foreign source.

    Structurally satisfied by
    :class:`~moncpipelib.resources.foreign_blob.ForeignBlobSource`; a
    narrow Protocol so tests can inject an in-memory fake without the
    Azure SDK.
    """

    def iter_list(self, prefix: str) -> Iterator[str]: ...
    def iter_child_prefixes(self, prefix: str) -> Iterator[str]: ...
    def stream(self, path: str) -> IO[bytes]: ...
    def get_properties(self, path: str) -> Any: ...


def open_foreign_source(
    source_cfg: dict[str, Any],
    credential_cfg: dict[str, Any],
    ctx: IngestContext,
) -> ForeignBlobReader:
    """Build a :class:`ForeignBlobSource` from config + ``ctx.secrets``.

    When ``credential.secret_name`` is set, resolves the SP client secret
    from Key Vault (via ``ctx.secrets``) and builds a
    :class:`~azure.identity.ClientSecretCredential`.  Otherwise falls back
    to ``DefaultAzureCredential`` (local dev).  The secret value is never
    logged and lives only for the constructed credential's lifetime.
    """
    account_url = str(source_cfg["account_url"])
    container = str(source_cfg["container"])
    secret_name = credential_cfg.get("secret_name")
    if secret_name:
        if ctx.secrets is None:
            raise IngestResolutionError(
                "blob_mirror credential.secret_name is set but no secrets "
                "resource is available on the IngestContext"
            )
        client_secret = ctx.secrets.get_secret(str(secret_name))
        return ForeignBlobSource.from_client_secret(
            account_url=account_url,
            container=container,
            tenant_id=str(credential_cfg["tenant_id"]),
            client_id=str(credential_cfg["client_id"]),
            client_secret=client_secret,
        )
    return ForeignBlobSource.with_default_credential(
        account_url=account_url,
        container=container,
    )


class BlobMirrorPattern:
    """Foreign-blob-mirror ingest pattern.

    The ``source_factory`` seam exists for tests: production uses the
    default (:func:`open_foreign_source`), which reaches Key Vault + Azure;
    tests pass a factory returning an in-memory :class:`ForeignBlobReader`.
    """

    name: ClassVar[str] = "blob_mirror"

    def __init__(
        self,
        source_factory: Callable[
            [dict[str, Any], dict[str, Any], IngestContext], ForeignBlobReader
        ] = open_foreign_source,
    ) -> None:
        self._source_factory = source_factory

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_partitions(
        self,
        contract: IngestContract,
        ctx: IngestContext,
    ) -> list[PartitionSpec]:
        """Discover partitions by walking the foreign source's cycle folders.

        When ``source.discovery_prefix`` is declared, walks its immediate
        child "folders" in the foreign container (the partner's
        folder-derived cycle discovery -- "which ``YYYYMM`` snapshots are
        present?") and emits one :class:`PartitionSpec` per child whose
        key matches ``source.partition_pattern`` (when set).  The
        presence-poll discovery sensor calls this each tick; keys are
        returned deterministically (sorted).

        When ``discovery_prefix`` is absent, returns ``[]`` -- discovery
        is then driven externally (e.g. a resolver / explicit partition
        add).  A key must not span ``/``; only the immediate folder level
        is considered.
        """
        source_cfg, credential_cfg = self._config(contract)
        if "discovery_prefix" not in source_cfg:
            return []

        import re

        discovery_prefix = render_prefix(str(source_cfg["discovery_prefix"]), "", contract)
        pattern_re = source_cfg.get("partition_pattern")
        compiled = re.compile(str(pattern_re)) if pattern_re else None

        source = self._source_factory(source_cfg, credential_cfg, ctx)
        keys: set[str] = set()
        for child in source.iter_child_prefixes(discovery_prefix):
            # child is the full prefix incl. trailing "/", e.g.
            # "<discovery_prefix>202501/".  Strip the prefix + slash to
            # get the immediate folder name.
            relative = child[len(discovery_prefix) :].strip("/")
            if not relative or "/" in relative:
                continue
            if compiled is not None and not compiled.fullmatch(relative):
                continue
            keys.add(relative)
        return [PartitionSpec(key=k, metadata={"partition_key": k}) for k in sorted(keys)]

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: BlobStorageResource,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        """Mirror one partition's foreign objects into our boundary."""
        source_cfg, credential_cfg = self._config(contract)
        object_glob = str(contract.pattern_config.get("object_glob", "*"))
        exclude_globs = [str(g) for g in (contract.pattern_config.get("exclude_globs") or [])]

        prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)
        object_prefix = render_prefix(
            str(source_cfg["object_prefix"]), partition_spec.key, contract
        )

        source = self._source_factory(source_cfg, credential_cfg, ctx)
        matched = self._match_objects(source, object_prefix, object_glob, exclude_globs)
        if not matched:
            raise IngestResolutionError(
                f"blob_mirror found no objects matching {object_glob!r} under foreign "
                f"prefix {object_prefix!r} for partition {partition_spec.key!r} "
                f"(exclude_globs={exclude_globs})"
            )

        results: list[IngestResult] = []
        seen_filenames: set[str] = set()
        for src_path in matched:
            filename = PurePosixPath(src_path).name
            if filename in seen_filenames:
                raise IngestResolutionError(
                    f"blob_mirror: two source objects map to the same landing "
                    f"filename {filename!r} under prefix {object_prefix!r}; source "
                    f"paths must have distinct basenames"
                )
            seen_filenames.add(filename)
            results.append(
                self._mirror_object(
                    source, blob, contract.sensitivity, prefix, src_path, filename, ctx
                )
            )
        return results

    def partition_metadata(
        self,
        contract: IngestContract,
        partition_key: str,
        ctx: IngestContext,
    ) -> dict[str, Any]:
        """Return ``{"partition_key": ...}`` for the manifest ``fields`` block.

        Symmetric with the other patterns' ``partition_metadata`` so a
        ``from_ingest`` consumer can hydrate ``effective_from`` from
        ``partition_key`` (the ``YYYYMM`` cycle == ``load_period``).
        """
        del contract, ctx
        return {"partition_key": partition_key}

    def manifest_resolver_block(self, contract: IngestContract) -> dict[str, Any]:
        """Return the manifest ``resolver`` audit block (foreign source location).

        Names where the bytes came from -- account, container, object
        glob -- for the durable audit trail.  Redaction contract: NO
        secrets (the client secret is never in config; ``secret_name`` is
        a Key Vault reference, not a value, and is intentionally omitted
        to keep the block purely descriptive of the byte source).
        """
        source_cfg, _ = self._config(contract)
        return {
            "name": self.name,
            "config": {
                "account_url": source_cfg.get("account_url"),
                "container": source_cfg.get("container"),
                "object_glob": contract.pattern_config.get("object_glob", "*"),
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _config(contract: IngestContract) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = contract.pattern_config
        if not cfg:
            raise IngestResolutionError(
                f"Contract {contract.source_name!r} has empty blob_mirror config"
            )
        source_cfg = cfg.get("source")
        if not isinstance(source_cfg, dict):
            raise IngestResolutionError(
                f"Contract {contract.source_name!r} blob_mirror config is missing a 'source' block"
            )
        credential_cfg = cfg.get("credential") or {}
        return source_cfg, dict(credential_cfg)

    @staticmethod
    def _match_objects(
        source: ForeignBlobReader,
        object_prefix: str,
        object_glob: str,
        exclude_globs: list[str],
    ) -> list[str]:
        """Return matching source object paths, sorted for determinism."""
        matched: list[str] = []
        for path in source.iter_list(object_prefix):
            if path.endswith("/"):  # virtual directory placeholder
                continue
            basename = PurePosixPath(path).name
            if not fnmatch(basename, object_glob):
                continue
            if any(fnmatch(basename, ex) for ex in exclude_globs):
                continue
            matched.append(path)
        return sorted(matched)

    def _mirror_object(
        self,
        source: ForeignBlobReader,
        blob: BlobStorageResource,
        sensitivity: Any,
        prefix: str,
        src_path: str,
        filename: str,
        ctx: IngestContext,
    ) -> IngestResult:
        """Mirror one object; skip via etag-compare when already landed."""
        target_path = f"{prefix}/{filename}"
        props = source.get_properties(src_path)
        src_etag = _normalize_etag(getattr(props, "etag", None))
        src_size = int(getattr(props, "size", 0) or 0)

        if src_etag:
            existing_etag = blob.read_metadata_value(sensitivity, target_path, _SOURCE_ETAG_KEY)
            if existing_etag == src_etag:
                existing_sha = blob.read_sha256_metadata(sensitivity, target_path)
                if existing_sha is not None:
                    _log_action(ctx, target_path, "skipped")
                    return IngestResult(
                        path=target_path,
                        sha256=existing_sha,
                        action="skipped",
                        size_bytes=src_size,
                    )
                # etag matched but no sha stamp (blob predates the scheme):
                # fall through and re-upload to restore the integrity stamp.

        with hashing_tempfile(suffix=".mirror") as writer:
            with source.stream(src_path) as fp:
                while True:
                    chunk = fp.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    writer.write(chunk)
            writer.close()
            sha256 = writer.sha256_hexdigest()
            size_bytes = writer.size_bytes
            extra = {_SOURCE_ETAG_KEY: src_etag} if src_etag else None
            with writer.path.open("rb") as up:
                blob.upload(sensitivity, target_path, up, sha256=sha256, extra_metadata=extra)

        _log_action(ctx, target_path, "uploaded")
        return IngestResult(
            path=target_path, sha256=sha256, action="uploaded", size_bytes=size_bytes
        )


def _normalize_etag(etag: str | None) -> str:
    """Strip surrounding quotes / whitespace so etag comparison is stable.

    Azure returns etags as quoted strings (``'"0x8D..."'``); metadata
    round-trips can vary the quoting, so we compare on the unquoted form.
    """
    if not etag:
        return ""
    return etag.strip().strip('"')


def _log_action(ctx: IngestContext, path: str, action: str) -> None:
    log: Any = ctx.log
    log.info("blob_mirror %s path=%s", action, path)
