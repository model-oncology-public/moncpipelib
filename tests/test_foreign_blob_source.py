"""Tests for ForeignBlobSource (#436).

The Azure SDK is patched at module level -- no network calls.  We
exercise what this type actually owns:

- Construction wires ``BlobServiceClient`` with the foreign account URL,
  the injected credential, and uniform chunk-size bounds.
- The credential factories build the right ``azure.identity`` credential.
- The read/list surface (``iter_list``, ``iter_child_prefixes``,
  ``stream``, ``download_to_path``, ``get_properties``, ``exists``) with
  no ``sensitivity`` argument, against one pinned container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from moncpipelib.resources.foreign_blob import ForeignBlobSource


def _build_source(**overrides: Any) -> tuple[ForeignBlobSource, MagicMock]:
    """Construct a ForeignBlobSource with BlobServiceClient patched out.

    Returns the source and the mock service-client instance so tests can
    inspect the ``.get_container_client(...).get_blob_client(...)`` chain.
    """
    defaults: dict[str, Any] = {
        "account_url": "https://examplestorageacct.blob.core.windows.net",
        "container": "delivery",
        "credential": MagicMock(name="TokenCredential"),
    }
    defaults.update(overrides)
    with patch("moncpipelib.resources.foreign_blob.BlobServiceClient") as mock_cls:
        service_instance = MagicMock(name="BlobServiceClient")
        mock_cls.return_value = service_instance
        source = ForeignBlobSource(**defaults)
    return source, service_instance


def test_construction_wires_service_client() -> None:
    cred = MagicMock(name="TokenCredential")
    with patch("moncpipelib.resources.foreign_blob.BlobServiceClient") as mock_cls:
        ForeignBlobSource(
            account_url="https://examplestorageacct.blob.core.windows.net",
            container="delivery",
            credential=cred,
        )
    mock_cls.assert_called_once_with(
        account_url="https://examplestorageacct.blob.core.windows.net",
        credential=cred,
        max_chunk_get_size=8 * 1024 * 1024,
        max_single_get_size=8 * 1024 * 1024,
    )


def test_from_client_secret_builds_client_secret_credential() -> None:
    with (
        patch("moncpipelib.resources.foreign_blob.BlobServiceClient") as mock_cls,
        patch("azure.identity.ClientSecretCredential") as mock_cred_cls,
    ):
        cred_instance = MagicMock(name="ClientSecretCredential")
        mock_cred_cls.return_value = cred_instance
        ForeignBlobSource.from_client_secret(
            account_url="https://examplestorageacct.blob.core.windows.net",
            container="delivery",
            tenant_id="partner-tenant",
            client_id="our-sp-client",
            client_secret="s3cr3t",
        )
    mock_cred_cls.assert_called_once_with(
        tenant_id="partner-tenant",
        client_id="our-sp-client",
        client_secret="s3cr3t",
    )
    # The credential built above is the one handed to BlobServiceClient.
    assert mock_cls.call_args.kwargs["credential"] is cred_instance


def test_with_default_credential_uses_default_azure_credential() -> None:
    with (
        patch("moncpipelib.resources.foreign_blob.BlobServiceClient") as mock_cls,
        patch("azure.identity.DefaultAzureCredential") as mock_cred_cls,
    ):
        cred_instance = MagicMock(name="DefaultAzureCredential")
        mock_cred_cls.return_value = cred_instance
        ForeignBlobSource.with_default_credential(
            account_url="https://examplestorageacct.blob.core.windows.net",
            container="delivery",
        )
    mock_cred_cls.assert_called_once_with()
    assert mock_cls.call_args.kwargs["credential"] is cred_instance


def test_iter_list_yields_blob_names() -> None:
    source, service = _build_source()
    container = service.get_container_client.return_value
    # ``name`` is a reserved MagicMock kwarg, so set it as an attribute.
    b1 = MagicMock(spec=["name"])
    b1.name = "202501/visits_oncology/part-00001.parquet"
    b2 = MagicMock(spec=["name"])
    b2.name = "202501/visits_oncology/part-00002.parquet"
    container.list_blobs.return_value = [b1, b2]

    names = list(source.iter_list("202501/visits_oncology"))

    service.get_container_client.assert_called_with("delivery")
    container.list_blobs.assert_called_once_with(name_starts_with="202501/visits_oncology")
    assert names == [
        "202501/visits_oncology/part-00001.parquet",
        "202501/visits_oncology/part-00002.parquet",
    ]


def test_iter_child_prefixes_returns_only_folders() -> None:
    source, service = _build_source()
    container = service.get_container_client.return_value
    # walk_blobs intermixes BlobPrefix (folder, name ends with "/") and
    # leaf BlobProperties (no trailing slash).  Only folders are yielded.
    folder_a = MagicMock(spec=["name"])
    folder_a.name = "202501/"
    folder_b = MagicMock(spec=["name"])
    folder_b.name = "202502/"
    leaf = MagicMock(spec=["name"])
    leaf.name = "current.json"
    container.walk_blobs.return_value = [folder_a, leaf, folder_b]

    prefixes = list(source.iter_child_prefixes(""))

    container.walk_blobs.assert_called_once_with(name_starts_with="", delimiter="/")
    assert prefixes == ["202501/", "202502/"]


def test_stream_reads_chunks_forward_only() -> None:
    source, service = _build_source()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    downloader = MagicMock(name="StorageStreamDownloader")
    downloader.chunks.return_value = iter([b"hello ", b"world"])
    downloader._response = None
    blob_client.download_blob.return_value = downloader

    out = bytearray()
    with source.stream("202501/visits_oncology/part-00001.parquet") as fp:
        while True:
            chunk = fp.read(4)
            if not chunk:
                break
            out.extend(chunk)

    blob_client.download_blob.assert_called_once_with()
    assert bytes(out) == b"hello world"


def test_download_to_path_streams_to_disk(tmp_path: Path) -> None:
    source, service = _build_source()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    downloader = MagicMock(name="StorageStreamDownloader")

    def _readinto(fp: Any) -> int:
        return fp.write(b"parquet-bytes")

    downloader.readinto.side_effect = _readinto
    blob_client.download_blob.return_value = downloader

    dest = tmp_path / "part.parquet"
    source.download_to_path("202501/visits_oncology/part-00001.parquet", dest)

    assert dest.read_bytes() == b"parquet-bytes"


def test_download_to_path_unlinks_partial_on_failure(tmp_path: Path) -> None:
    source, service = _build_source()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    downloader = MagicMock(name="StorageStreamDownloader")
    downloader.readinto.side_effect = RuntimeError("mid-stream failure")
    blob_client.download_blob.return_value = downloader

    dest = tmp_path / "part.parquet"
    with pytest.raises(RuntimeError):
        source.download_to_path("x", dest)
    assert not dest.exists()


def test_get_properties_returns_etag_and_size() -> None:
    source, service = _build_source()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    props = MagicMock(name="BlobProperties")
    props.etag = '"0x8DABCDEF"'
    props.size = 4096
    blob_client.get_blob_properties.return_value = props

    result = source.get_properties("202501/visits_oncology/part-00001.parquet")

    assert result.etag == '"0x8DABCDEF"'
    assert result.size == 4096


def test_exists_delegates_to_blob_client() -> None:
    source, service = _build_source()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.exists.return_value = True

    assert source.exists("202501/visits_oncology/part-00001.parquet") is True
