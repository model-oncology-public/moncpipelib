"""Azure Key Vault secret resource for the universal ingest boundary.

This module provides :class:`KeyVaultSecretResource`, a Dagster
:class:`~dagster.ConfigurableResource` that mediates access to secrets
required by ingest patterns -- most notably the API keys used by
authenticated :mod:`moncpipelib.ingest.resolvers` (e.g. the UTS API key
for UMLS Metathesaurus and RxNorm release discovery).

Security / compliance context:

- Workload identity federation only -- no shared keys or static
  credentials in YAML / env vars.  Auth flows through
  :class:`~azure.identity.DefaultAzureCredential`.
- Secrets are fetched per call; this resource does NOT cache secret
  values across :meth:`get_secret` invocations.  Per the
  credential-lifecycle decision in moncpipelib#216 the dispatcher
  resolves the secret per ``materialize_partition`` call so a rotated
  Key Vault value is picked up on the next tick rather than stale-cached
  on a resolver instance.
- Secret values must never appear in logs.  This resource never logs
  the value itself; callers are responsible for redaction at use sites
  (see ``src/moncpipelib/ingest/_http.py`` for the redacting httpx
  client factory used by ``api_resolver``-flow callers).
- Supports HIPAA 164.312(a)(2)(i) (unique user identification via
  workload identity) and SOC 2 CC6.1 (logical access via a centralized
  credential broker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dagster import ConfigurableResource
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from dagster import InitResourceContext


class KeyVaultSecretResource(ConfigurableResource):
    """Dagster resource for resolving secrets from Azure Key Vault.

    Used by ingest patterns that require API credentials at
    materialization time (e.g. ``api_resolver`` calling UTS for the
    current UMLS release URL).  Workload-identity federated; no static
    credentials.

    Attributes:
        vault_url: Full URL of the Key Vault, e.g.
            ``"https://kv-monc-data-npe.vault.azure.net/"``.  Trailing
            slash is optional -- the SDK accepts both forms.
    """

    vault_url: str

    _credential: DefaultAzureCredential = PrivateAttr()
    _client: SecretClient = PrivateAttr()

    def setup_for_execution(self, context: InitResourceContext) -> None:  # noqa: ARG002
        """Instantiate the credential + secret client once per run."""
        self._credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=self.vault_url, credential=self._credential)

    def get_secret(self, name: str) -> str:
        """Return the current value of the named secret.

        No caching across calls -- each invocation hits Key Vault so a
        rotated secret is picked up on the next call rather than
        stale-cached on this resource instance (load-bearing for the
        per-call credential-lifecycle posture documented in #216).

        Raises:
            KeyError: When the secret does not exist in the vault.  The
                message includes the secret name and vault URL so an
                operator can diagnose without leaking any value
                (there isn't one to leak).
            ValueError: When the secret exists but has no value -- rare;
                indicates a manually-purged secret whose audit record
                was retained.
        """
        try:
            secret = self._client.get_secret(name)
        except ResourceNotFoundError as e:
            raise KeyError(f"Secret {name!r} not found in vault {self.vault_url!r}") from e
        if secret.value is None:
            raise ValueError(f"Secret {name!r} in vault {self.vault_url!r} has no value")
        return secret.value
