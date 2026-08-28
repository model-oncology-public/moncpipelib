"""Tests for KeyVaultSecretResource.

The Azure SDK clients are patched at module level -- no network calls.
We exercise the three things this resource actually owns:

- Lazy credential construction (no auth at import time).
- ``get_secret`` happy path.
- ``get_secret`` error contracts: missing secret -> ``KeyError``;
  empty-value secret -> ``ValueError``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceNotFoundError

from moncpipelib.resources.keyvault import KeyVaultSecretResource


def _build_resource(
    vault_url: str = "https://kv-monc-data-npe.vault.azure.net/",
) -> tuple[KeyVaultSecretResource, MagicMock]:
    """Construct a KeyVaultSecretResource with the Azure SDK patched out.

    Returns the resource and the mock ``SecretClient`` so tests can
    drive its ``get_secret`` return value.
    """
    with (
        patch("moncpipelib.resources.keyvault.DefaultAzureCredential") as _cred,
        patch("moncpipelib.resources.keyvault.SecretClient") as mock_client_cls,
    ):
        del _cred  # silence ARG on the unused binding
        client_instance = MagicMock(name="SecretClient")
        mock_client_cls.return_value = client_instance
        resource = KeyVaultSecretResource(vault_url=vault_url)
        resource.setup_for_execution(MagicMock(name="InitResourceContext"))
    return resource, client_instance


def test_resource_construction_does_not_authenticate_eagerly() -> None:
    """Constructing the resource (without ``setup_for_execution``) must not
    instantiate ``DefaultAzureCredential`` -- otherwise importing a code
    location would attempt Azure auth at definition-load time."""
    with (
        patch("moncpipelib.resources.keyvault.DefaultAzureCredential") as cred_cls,
        patch("moncpipelib.resources.keyvault.SecretClient") as client_cls,
    ):
        KeyVaultSecretResource(vault_url="https://example.vault.azure.net/")
        cred_cls.assert_not_called()
        client_cls.assert_not_called()


def test_get_secret_happy_path() -> None:
    resource, mock_client = _build_resource()
    secret_obj = MagicMock(name="KeyVaultSecret")
    secret_obj.value = "super-secret-value"
    mock_client.get_secret.return_value = secret_obj

    result = resource.get_secret("uts-api-key")

    assert result == "super-secret-value"
    mock_client.get_secret.assert_called_once_with("uts-api-key")


def test_get_secret_missing_raises_key_error_with_context() -> None:
    """Missing-secret error must name the secret and the vault for
    operator diagnosis without leaking any value (none exists)."""
    resource, mock_client = _build_resource(vault_url="https://kv-prd.vault.azure.net/")
    mock_client.get_secret.side_effect = ResourceNotFoundError("404 from KV")

    with pytest.raises(KeyError) as exc:
        resource.get_secret("missing-secret")

    msg = str(exc.value)
    assert "missing-secret" in msg
    assert "kv-prd.vault.azure.net" in msg


def test_get_secret_empty_value_raises_value_error() -> None:
    """A secret whose ``value`` is None (manually-purged but audit-retained)
    must surface as a clear error rather than returning ``None`` to
    callers expecting a string."""
    resource, mock_client = _build_resource()
    secret_obj = MagicMock(name="KeyVaultSecret")
    secret_obj.value = None
    mock_client.get_secret.return_value = secret_obj

    with pytest.raises(ValueError, match="has no value"):
        resource.get_secret("purged-secret")


def test_get_secret_no_caching_across_calls() -> None:
    """Per the credential-lifecycle decision in #216, the resource MUST
    NOT cache values across calls -- every ``get_secret`` invocation
    hits Key Vault so a rotated secret is picked up on the next call."""
    resource, mock_client = _build_resource()
    secret_obj = MagicMock(name="KeyVaultSecret")
    secret_obj.value = "v1"
    mock_client.get_secret.return_value = secret_obj

    resource.get_secret("uts-api-key")
    resource.get_secret("uts-api-key")
    resource.get_secret("uts-api-key")

    assert mock_client.get_secret.call_count == 3
