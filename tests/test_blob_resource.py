"""Tests for BlobStorageResource.

The Azure SDK clients are patched at module level -- no network calls.
We exercise the three things this resource actually owns:

- Container selection by sensitivity (and the missing-sensitivity error).
- sha256 metadata handling on upload + read.
- The ``read_sha256_metadata`` contract: None on missing blob OR missing header.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceNotFoundError

from moncpipelib.resources.blob import BlobStorageResource


def _build_resource(**overrides: object) -> tuple[BlobStorageResource, MagicMock]:
    """Construct a BlobStorageResource with the Azure SDK patched out.

    Returns the resource and the mock ``BlobServiceClient`` so tests can
    inspect .get_container_client(...).get_blob_client(...) call chains.
    """
    defaults = {
        "storage_account": "examplestorageacct",
        "container_public": "landing-reference",
    }
    defaults.update(overrides)

    with (
        patch("moncpipelib.resources.blob.DefaultAzureCredential") as _cred,
        patch("moncpipelib.resources.blob.BlobServiceClient") as mock_service_cls,
    ):
        del _cred  # silence ARG on the unused binding
        service_instance = MagicMock(name="BlobServiceClient")
        mock_service_cls.return_value = service_instance
        resource = BlobStorageResource(**defaults)  # type: ignore[arg-type]
        resource.setup_for_execution(MagicMock(name="InitResourceContext"))
    return resource, service_instance


def test_missing_sensitivity_raises() -> None:
    resource, _ = _build_resource()
    # container_confidential was left unset
    with pytest.raises(ValueError, match="sensitivity='confidential'"):
        resource._container_name("confidential")


def test_upload_sets_sha256_metadata() -> None:
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value

    resource.upload("public", "cms_asp/2024-01-01/foo.csv", b"payload", sha256="abc123")

    service.get_container_client.assert_called_with("landing-reference")
    service.get_container_client.return_value.get_blob_client.assert_called_with(
        "cms_asp/2024-01-01/foo.csv"
    )
    blob_client.upload_blob.assert_called_once_with(
        b"payload", overwrite=True, metadata={"sha256": "abc123"}
    )


def test_read_sha256_metadata_returns_header_when_present() -> None:
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    props = MagicMock()
    props.metadata = {"sha256": "deadbeef"}
    blob_client.get_blob_properties.return_value = props

    assert resource.read_sha256_metadata("public", "foo") == "deadbeef"


def test_read_sha256_metadata_returns_none_when_blob_missing() -> None:
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.get_blob_properties.side_effect = ResourceNotFoundError("not found")

    assert resource.read_sha256_metadata("public", "foo") is None


def test_read_sha256_metadata_returns_none_when_header_absent() -> None:
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    props = MagicMock()
    props.metadata = {}  # blob exists but was uploaded before we tracked sha256
    blob_client.get_blob_properties.return_value = props

    assert resource.read_sha256_metadata("public", "foo") is None


@pytest.mark.parametrize(
    "server_key",
    ["sha256", "Sha256", "SHA256", "sHa256"],
)
def test_read_sha256_metadata_is_case_insensitive(server_key: str) -> None:
    """Regression for #214: Azure Storage may return ``x-ms-meta-sha256``
    with arbitrary header casing (observed in prod as ``Sha256``).
    """
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    props = MagicMock()
    props.metadata = {server_key: "deadbeef"}
    blob_client.get_blob_properties.return_value = props

    assert resource.read_sha256_metadata("public", "foo") == "deadbeef"


def test_read_sha256_metadata_ignores_unrelated_keys() -> None:
    resource, service = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    props = MagicMock()
    props.metadata = {"OtherKey": "x", "ContentType": "text/csv"}
    blob_client.get_blob_properties.return_value = props

    assert resource.read_sha256_metadata("public", "foo") is None


def test_container_selected_by_sensitivity() -> None:
    resource, service = _build_resource(
        container_confidential="landing-confidential",
        container_phi="landing-phi",
    )

    resource.upload("phi", "rx/part/foo.rrf", b"x", sha256="s")

    service.get_container_client.assert_called_with("landing-phi")


def test_list_delegates_to_container_client() -> None:
    resource, service = _build_resource()
    container = service.get_container_client.return_value
    blob_a = MagicMock()
    blob_a.name = "cms_asp/2024-01-01/a.csv"
    blob_b = MagicMock()
    blob_b.name = "cms_asp/2024-01-01/b.csv"
    container.list_blobs.return_value = [blob_a, blob_b]

    out = resource.list("public", "cms_asp/2024-01-01")

    container.list_blobs.assert_called_once_with(name_starts_with="cms_asp/2024-01-01")
    assert out == ["cms_asp/2024-01-01/a.csv", "cms_asp/2024-01-01/b.csv"]


# ---------------------------------------------------------------------------
# iter_list lazy iterator (Migration 012 Phase E / #246)
# ---------------------------------------------------------------------------


def test_iter_list_returns_generator_not_list() -> None:
    """``iter_list`` must yield, not return a fully-materialized list.

    Pre-fix the only ``list``-style API on ``BlobStorageResource``
    materialized every blob name eagerly; a 100k-name prefix would
    produce ~10 MiB of strings on the heap before the consumer saw
    the first name.  ``iter_list`` preserves the SDK's pagination
    laziness all the way to the caller.
    """
    import types

    resource, service = _build_resource()
    container = service.get_container_client.return_value
    container.list_blobs.return_value = iter([])  # empty SDK iterator

    out = resource.iter_list("public", "cms_asp/2024-01-01")

    assert isinstance(out, types.GeneratorType)
    container.list_blobs.assert_not_called()  # not yet -- iter_list is lazy


def test_iter_list_does_not_consume_sdk_iterator_until_pulled() -> None:
    """A consumer that takes only the first N names should not pull
    every page from the SDK -- this is the whole point of streaming
    the listing for large prefixes."""
    resource, service = _build_resource()
    container = service.get_container_client.return_value

    # Build an SDK iterator that records each pull so we can assert
    # the consumer pulled exactly what it asked for.
    pulled: list[str] = []

    def _gen() -> Any:  # type: ignore[no-untyped-def]
        for i in range(10):
            blob = MagicMock()
            blob.name = f"prefix/{i:03d}.bin"
            pulled.append(blob.name)
            yield blob

    container.list_blobs.return_value = _gen()

    iterator = resource.iter_list("public", "prefix")
    first = next(iterator)
    assert first == "prefix/000.bin"
    # Only the first name has been pulled from the SDK so far.
    assert pulled == ["prefix/000.bin"]
    # Pull two more.
    assert next(iterator) == "prefix/001.bin"
    assert next(iterator) == "prefix/002.bin"
    assert pulled == ["prefix/000.bin", "prefix/001.bin", "prefix/002.bin"]


def test_iter_list_yields_same_results_as_list() -> None:
    """Round-trip equivalence: ``list(blob.iter_list(...))`` matches
    ``blob.list(...)`` for the same prefix.  Pins that the streaming
    rewrite of ``list()`` (now a thin wrapper around ``iter_list``)
    preserves order + completeness."""
    resource, service = _build_resource()
    container = service.get_container_client.return_value
    blob_a = MagicMock()
    blob_a.name = "x/a.csv"
    blob_b = MagicMock()
    blob_b.name = "x/b.csv"
    blob_c = MagicMock()
    blob_c.name = "x/c.csv"
    # The SDK iterator can be consumed once, so build it twice.
    container.list_blobs.side_effect = [
        iter([blob_a, blob_b, blob_c]),
        iter([blob_a, blob_b, blob_c]),
    ]

    streamed = list(resource.iter_list("public", "x"))
    materialized = resource.list("public", "x")

    assert streamed == materialized == ["x/a.csv", "x/b.csv", "x/c.csv"]


def test_list_is_a_thin_wrapper_around_iter_list() -> None:
    """``list()`` materializes ``iter_list()`` -- pin that the wrapper
    delegation holds so that future changes to ``iter_list`` propagate
    to ``list()`` automatically (e.g. a docstring or filtering change
    in one place rather than two).
    """
    resource, service = _build_resource()
    container = service.get_container_client.return_value
    blob = MagicMock()
    blob.name = "p/single.bin"
    container.list_blobs.return_value = iter([blob])

    out = resource.list("public", "p")

    assert out == ["p/single.bin"]
    container.list_blobs.assert_called_once_with(name_starts_with="p")
