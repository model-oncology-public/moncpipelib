"""Tests for the ReleaseResolver registry.

Covers:

- The built-in ``uts_release`` resolver is registered at import time.
- ``get_resolver`` returns the registered instance.
- Unknown name raises ``KeyError`` listing known resolvers.
- ADR-2 anti-pattern sweep: no registered resolver opens a socket
  during ``validate_config``.
"""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import patch

import pytest

from moncpipelib.ingest.resolvers import (
    RESOLVERS,
    ReleaseResolver,
    get_resolver,
    register_resolver,
)
from moncpipelib.ingest.resolvers.uts import UtsReleaseResolver


def test_builtin_uts_resolver_is_registered() -> None:
    assert "uts_release" in RESOLVERS
    assert isinstance(RESOLVERS["uts_release"], UtsReleaseResolver)


def test_get_resolver_returns_registered_instance() -> None:
    r = get_resolver("uts_release")
    assert isinstance(r, UtsReleaseResolver)


def test_get_resolver_unknown_raises_with_known_list() -> None:
    with pytest.raises(KeyError, match="Unknown release resolver 'mystery'"):
        get_resolver("mystery")
    # The known list is sorted in the error for diagnosability
    try:
        get_resolver("mystery")
    except KeyError as e:
        assert "uts_release" in str(e)


def test_uts_resolver_satisfies_release_resolver_protocol() -> None:
    """Runtime-checkable Protocol: ``isinstance`` confirms the contract."""
    r = get_resolver("uts_release")
    assert isinstance(r, ReleaseResolver)


def test_register_resolver_overwrites_same_name() -> None:
    """Test-only ergonomic: registering with the same name overwrites
    the previous entry so tests can swap a stub in.  We restore the
    real resolver in the cleanup so other tests aren't affected."""
    original = RESOLVERS["uts_release"]
    try:

        class _Stub:
            name = "uts_release"

            def validate_config(self, config: dict[str, Any]) -> list[str]:
                del config
                return []

            def current_release(
                self, api_key: str, config: dict[str, Any], ctx: Any
            ) -> dict[str, Any]:
                del api_key, config, ctx
                return {"partition_key": "stub"}

            def resolve_url(
                self,
                api_key: str,
                partition_key: str,
                config: dict[str, Any],
                ctx: Any,
            ) -> str:
                del api_key, partition_key, config, ctx
                return "stub://"

        register_resolver(_Stub())  # type: ignore[arg-type]
        assert RESOLVERS["uts_release"].name == "uts_release"
        assert not isinstance(RESOLVERS["uts_release"], UtsReleaseResolver)
    finally:
        register_resolver(original)


def test_no_registered_resolver_opens_a_connection_during_validate_config() -> None:
    """ADR-2 anti-pattern sweep.

    ``validate_config`` runs at contract-load time including in CI; it
    MUST be deterministic and offline.  We patch ``socket.socket.connect``
    (catches every HTTP library plus subprocess and raw-socket
    access -- not just httpx) and assert no resolver triggers it
    during a sweep over the registry.
    """

    def boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise AssertionError(
            "A registered resolver opened a socket during validate_config -- "
            "this is forbidden by ADR-2."
        )

    with patch.object(socket.socket, "connect", side_effect=boom):
        for name, resolver in RESOLVERS.items():
            del name
            # Empty config triggers required-field errors but should not
            # touch the network.
            resolver.validate_config({})
