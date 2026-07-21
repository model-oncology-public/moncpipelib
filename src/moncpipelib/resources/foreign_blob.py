"""Foreign-account (cross-tenant) ADLS Gen2 read source (#436).

:class:`ForeignBlobSource` reads and lists blobs in an **arbitrary**
Azure Storage / ADLS Gen2 account -- typically a partner-owned account
in a different Entra tenant -- authenticated by a service-principal
credential.  It is deliberately distinct from
:class:`~moncpipelib.resources.blob.BlobStorageResource`, which is
workload-identity-federated, single-account, and container-by-sensitivity
for *our own* landing boundary.

Why a separate type (and why an SP at all):

- A managed identity (system- or user-assigned) is strictly
  single-tenant -- it can only obtain tokens for its home tenant, so it
  **cannot** read a resource in a partner's tenant.  Cross-tenant
  data-plane access requires an app registration / service principal the
  partner tenant can grant an RBAC role to.  This is the partner's
  documented default (they grant our SP ``Storage Blob Data Reader`` on
  their container).  See ``docs/migrations/20260717_436-439-foreign-blob-parquet-ingest.md``
  (design decisions D1/D2) and ``SECURITY.md``.
- The credential is injected (a :class:`~azure.core.credentials.TokenCredential`),
  not hard-coded to one auth mechanism: the default factory
  (:meth:`from_client_secret`) builds a
  :class:`~azure.identity.ClientSecretCredential` from an SP secret
  resolved out of Key Vault, but a certificate or federated (WIF)
  credential can be substituted config-only, and
  :meth:`with_default_credential` supports local dev.

Read/list only.  This type has no upload path by design -- foreign
accounts are sources, never sinks (issue #436 non-goal).  The
``blob_mirror`` pattern (#437) is what lands foreign bytes into our
sensitivity-scoped boundary.

Streaming posture mirrors ``BlobStorageResource``: :meth:`stream` yields
a forward-only, chunk-bounded reader; :meth:`download_to_path` streams to
disk; peak memory stays bounded by ``max_chunk_get_size`` regardless of
blob size (#241).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

from azure.storage.blob import BlobServiceClient

from moncpipelib.resources._blob_reader import (
    _DEFAULT_MAX_CHUNK_GET_SIZE,
    _ChunkedBlobReader,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from azure.core.credentials import TokenCredential
    from azure.storage.blob import BlobProperties, ContainerClient


class ForeignBlobSource:
    """SP-credentialed read/list surface over one foreign account+container.

    Construct via a factory (:meth:`from_client_secret` /
    :meth:`with_default_credential`) in production, or directly with an
    injected ``credential`` (tests pass a fake credential + patch
    :class:`~azure.storage.blob.BlobServiceClient`).

    Unlike :class:`~moncpipelib.resources.blob.BlobStorageResource`, the
    read methods take **no** ``sensitivity`` argument: a foreign source
    is pinned to exactly one ``(account_url, container)`` for its
    lifetime.

    Args:
        account_url: Full blob-service endpoint of the foreign account,
            e.g. ``"https://examplestorageacct.blob.core.windows.net"``.  ADLS
            Gen2 accounts also expose a ``dfs`` endpoint; the blob
            endpoint is what the ``azure-storage-blob`` SDK expects.
        container: Container (filesystem) name within that account.
        credential: Any :class:`~azure.core.credentials.TokenCredential`
            (typically a :class:`~azure.identity.ClientSecretCredential`
            for the cross-tenant SP).
        max_chunk_get_size: Per-chunk download size; bounds streaming
            memory.  Defaults to 8 MiB (matches ``BlobStorageResource``).
    """

    def __init__(
        self,
        *,
        account_url: str,
        container: str,
        credential: TokenCredential,
        max_chunk_get_size: int = _DEFAULT_MAX_CHUNK_GET_SIZE,
    ) -> None:
        self.account_url = account_url
        self.container = container
        self.max_chunk_get_size = max_chunk_get_size
        self._service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
            max_chunk_get_size=max_chunk_get_size,
            max_single_get_size=max_chunk_get_size,
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_client_secret(
        cls,
        *,
        account_url: str,
        container: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        max_chunk_get_size: int = _DEFAULT_MAX_CHUNK_GET_SIZE,
    ) -> ForeignBlobSource:
        """Build a source authenticated by an SP client secret.

        ``tenant_id`` is the **partner's** tenant (the token audience),
        ``client_id`` our SP's app-registration id, and ``client_secret``
        the secret value resolved from Key Vault at call time.  The
        secret is never logged and is held only for the lifetime of the
        constructed credential.
        """
        # Imported here (not at module top) so the module import stays
        # light for callers that inject their own credential in tests.
        from azure.identity import ClientSecretCredential

        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        return cls(
            account_url=account_url,
            container=container,
            credential=credential,
            max_chunk_get_size=max_chunk_get_size,
        )

    @classmethod
    def with_default_credential(
        cls,
        *,
        account_url: str,
        container: str,
        max_chunk_get_size: int = _DEFAULT_MAX_CHUNK_GET_SIZE,
    ) -> ForeignBlobSource:
        """Build a source using ``DefaultAzureCredential`` (local dev only).

        Intended for a developer who has an ``az login`` session with
        reader access to the foreign container; production uses
        :meth:`from_client_secret`.
        """
        from azure.identity import DefaultAzureCredential

        return cls(
            account_url=account_url,
            container=container,
            credential=DefaultAzureCredential(),
            max_chunk_get_size=max_chunk_get_size,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _container_client(self) -> ContainerClient:
        return self._service_client.get_container_client(self.container)

    # ------------------------------------------------------------------
    # List surface
    # ------------------------------------------------------------------

    def iter_list(self, prefix: str) -> Iterator[str]:
        """Yield blob paths under ``prefix`` (lazy, SDK-paginated).

        Preserves the SDK's pagination laziness end-to-end so a
        high-cardinality prefix does not materialize every name on the
        heap before the first is seen (mirrors
        :meth:`BlobStorageResource.iter_list`).
        """
        container = self._container_client()
        for blob in container.list_blobs(name_starts_with=prefix):
            yield blob.name

    def list(self, prefix: str) -> list[str]:
        """Materialize :meth:`iter_list` into a ``list[str]``."""
        return list(self.iter_list(prefix))

    def iter_child_prefixes(self, prefix: str) -> Iterator[str]:
        """Yield the immediate child "folder" prefixes under ``prefix``.

        Uses the SDK's delimiter walk (``walk_blobs(..., delimiter="/")``)
        so folder discovery does not enumerate every leaf blob -- the
        cheap primitive behind the ``blob_mirror`` presence-poll
        discovery ("which ``YYYYMM`` cycle folders exist?", #437).

        Each yielded value is the full prefix **including** the trailing
        ``/`` (e.g. ``"202501/"``), matching the SDK's ``BlobPrefix.name``.
        ``prefix`` may be ``""`` to walk the container root.
        """
        container = self._container_client()
        for item in container.walk_blobs(name_starts_with=prefix, delimiter="/"):
            # ``walk_blobs`` yields ``BlobPrefix`` (virtual dir) and
            # ``BlobProperties`` (leaf) items intermixed; only the former
            # carry a folder name.  ``BlobPrefix`` has no ``size``.
            name = getattr(item, "name", None)
            if name is not None and name.endswith("/"):
                yield name

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    def stream(self, path: str) -> IO[bytes]:
        """Return a forward-only, chunk-bounded ``IO[bytes]`` over ``path``.

        Peak memory is bounded by ``max_chunk_get_size``.  Bounded reads
        only (``read(n)`` / ``readinto``); an unbounded ``read()`` raises
        (see :class:`~moncpipelib.resources._blob_reader._ChunkedBlobReader`).
        Wrap in ``with`` so the underlying HTTP response is released
        deterministically.
        """
        downloader = self._container_client().get_blob_client(path).download_blob()
        return cast("IO[bytes]", _ChunkedBlobReader(downloader))

    def download_to_path(self, src: str, dest: Path | str) -> None:
        """Stream-download a blob to a local file.

        Peak memory is bounded by ``max_chunk_get_size`` (the SDK's
        ``readinto`` path).  On failure the destination is unlinked so
        callers never observe a half-written file.
        """
        dest_path = Path(dest)
        blob_client = self._container_client().get_blob_client(src)
        try:
            with dest_path.open("wb") as fp:
                blob_client.download_blob().readinto(fp)
        except Exception:
            dest_path.unlink(missing_ok=True)
            raise

    def exists(self, path: str) -> bool:
        """Return ``True`` iff a blob exists at ``path``."""
        return self._container_client().get_blob_client(path).exists()

    def get_properties(self, path: str) -> BlobProperties:
        """Return the SDK's ``BlobProperties`` for ``path``.

        Carries ``etag`` and ``size`` -- the inputs to the ``blob_mirror``
        etag-compare idempotency check (#437).
        """
        return self._container_client().get_blob_client(path).get_blob_properties()
