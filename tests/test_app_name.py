"""Tests for Postgres ``application_name`` run-to-backend correlation (#365)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from moncpipelib.resources import _app_name
from moncpipelib.resources._app_name import (
    _FALLBACK_APP_NAME,
    _MAX_APP_NAME_LEN,
    bind_run_id,
    resolve_application_name,
)
from moncpipelib.resources.postgres import PostgresResource

_RUN_UUID = "357c5cf8-0fcd-4488-8aa3-b8299938fb22"


@pytest.fixture(autouse=True)
def _isolate_run_id_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the run_id contextvar and DAGSTER_RUN_ID env between tests.

    The contextvar persists across calls within a thread, so every test starts
    from an unbound state to keep resolution order assertions deterministic.
    """
    token = _app_name._RUN_ID.set(None)
    monkeypatch.delenv("DAGSTER_RUN_ID", raising=False)
    try:
        yield
    finally:
        _app_name._RUN_ID.reset(token)


class TestResolveApplicationName:
    """Resolution-order coverage for :func:`resolve_application_name`."""

    def test_bound_run_id_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bound run_id takes precedence over env and hostname."""
        monkeypatch.setenv("DAGSTER_RUN_ID", "env-run")
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: f"dagster-run-{_RUN_UUID}-abc")
        bind_run_id(_RUN_UUID)
        assert resolve_application_name() == _RUN_UUID

    def test_env_var_used_when_unbound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``DAGSTER_RUN_ID`` is used when no run_id is bound."""
        monkeypatch.setenv("DAGSTER_RUN_ID", "env-run-id")
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: "some-other-pod")
        assert resolve_application_name() == "env-run-id"

    def test_hostname_parsed_for_run_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run-worker pod hostname yields the run_id without any binding."""
        monkeypatch.setattr(
            _app_name.socket, "gethostname", lambda: f"dagster-run-{_RUN_UUID}-8mjh9"
        )
        assert resolve_application_name() == _RUN_UUID

    def test_step_pod_hostname_does_not_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step-executor pod hostnames do not encode a run_id -> fallback."""
        monkeypatch.setattr(
            _app_name.socket,
            "gethostname",
            lambda: "dagster-step-89866e26730cc5844da7a4819721d5ae-sthrb",
        )
        assert resolve_application_name() == _FALLBACK_APP_NAME

    def test_fallback_when_nothing_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No binding, no env, non-Dagster hostname -> stable fallback."""
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: "laptop")
        assert resolve_application_name() == _FALLBACK_APP_NAME

    def test_clamped_to_max_len(self) -> None:
        """A long bound identifier is clamped to the Postgres limit."""
        bind_run_id("x" * 200)
        assert len(resolve_application_name()) == _MAX_APP_NAME_LEN


class TestBindRunId:
    """Binding semantics for :func:`bind_run_id`."""

    def test_none_does_not_clobber(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Binding ``None`` leaves a previously-bound run_id intact."""
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: "laptop")
        bind_run_id(_RUN_UUID)
        bind_run_id(None)
        assert resolve_application_name() == _RUN_UUID

    def test_empty_string_does_not_clobber(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Binding an empty string is a no-op."""
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: "laptop")
        bind_run_id(_RUN_UUID)
        bind_run_id("")
        assert resolve_application_name() == _RUN_UUID


class TestConnectionSitesTagApplicationName:
    """The resource connect sites pass ``application_name`` to psycopg."""

    @pytest.fixture
    def resource(self) -> PostgresResource:
        return PostgresResource(
            host="localhost",
            port=5432,
            user="u",
            password="p",
            database="db",
        )

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_get_connection_passes_app_name(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        bind_run_id(_RUN_UUID)
        with resource.get_connection():
            pass
        assert mock_connect.call_args.kwargs["application_name"] == _RUN_UUID

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_get_connection_raw_passes_app_name(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        bind_run_id(_RUN_UUID)
        resource.get_connection_raw()
        assert mock_connect.call_args.kwargs["application_name"] == _RUN_UUID

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_fallback_app_name_when_unbound(
        self, mock_connect: MagicMock, resource: PostgresResource, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_app_name.socket, "gethostname", lambda: "laptop")
        resource.get_connection_raw()
        assert mock_connect.call_args.kwargs["application_name"] == _FALLBACK_APP_NAME

    def test_engine_resolves_identity_per_connect(self, resource: PostgresResource) -> None:
        """The engine resolves application_name/options at connect time, not at
        engine-creation time -- a run_id bound *after* get_engine() still lands.

        Fires the ``do_connect`` event the resource registers (no real DB
        connection is opened) and asserts the connect params it mutates.
        """
        engine = resource.get_engine()  # created before any bind_run_id
        bind_run_id(_RUN_UUID)  # bind happens afterwards -- must still take effect
        cparams: dict[str, object] = {}
        engine.dialect.dispatch.do_connect(engine.dialect, MagicMock(), [], cparams)
        assert cparams["application_name"] == _RUN_UUID
        assert cparams["options"] == "-c client_connection_check_interval=10s"


class TestClientConnectionCheckInterval:
    """``client_connection_check_interval`` per-session containment (#365)."""

    @pytest.fixture
    def resource(self) -> PostgresResource:
        return PostgresResource(host="localhost", port=5432, user="u", password="p", database="db")

    def test_default_options_string(self, resource: PostgresResource) -> None:
        """Default field renders the libpq ``-c`` option for 10s."""
        assert resource._connection_options() == "-c client_connection_check_interval=10s"

    def test_custom_interval(self) -> None:
        res = PostgresResource(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="db",
            client_connection_check_interval="5s",
        )
        assert res._connection_options() == "-c client_connection_check_interval=5s"

    @pytest.mark.parametrize("disabled", [None, "off", "none", "disabled", ""])
    def test_disabled_returns_empty(self, disabled: str | None) -> None:
        """Disable sentinels render an empty (no-op) options string."""
        res = PostgresResource(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="db",
            client_connection_check_interval=disabled,
        )
        assert res._connection_options() == ""

    def test_invalid_interval_raises(self, resource: PostgresResource) -> None:
        """The error names client_connection_check_interval, not statement_timeout."""
        with pytest.raises(ValueError, match="invalid client_connection_check_interval"):
            resource._connection_options("not-a-duration")

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_get_connection_passes_options(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        resource.get_connection_raw()
        assert mock_connect.call_args.kwargs["options"] == "-c client_connection_check_interval=10s"


class TestCheckConnectionFactory:
    """The asset-check connection factory carries the #365 identity.

    This is the factory shared by both ``PostgresResource.make_contract_checks``
    and ``PostgresIOManager.make_contract_checks`` -- the IO-manager site that
    was previously missing both kwargs.
    """

    @pytest.fixture
    def resource(self) -> PostgresResource:
        return PostgresResource(host="localhost", port=5432, user="u", password="p", database="db")

    @patch("moncpipelib.resources.postgres.psycopg.connect")
    def test_factory_passes_app_name_and_options(
        self, mock_connect: MagicMock, resource: PostgresResource
    ) -> None:
        bind_run_id(_RUN_UUID)
        factory = resource._make_check_connection_factory()
        factory()
        assert mock_connect.call_args.kwargs["application_name"] == _RUN_UUID
        assert mock_connect.call_args.kwargs["options"] == "-c client_connection_check_interval=10s"
        # Check connections get the same connect_timeout guard as the other
        # sites so an unreachable host fails fast rather than hanging.
        assert mock_connect.call_args.kwargs["connect_timeout"] == resource.connect_timeout

    def test_io_manager_uses_the_same_factory(self, resource: PostgresResource) -> None:
        """The IO manager delegates to the resource factory (no drift)."""
        from moncpipelib.io_managers.postgres import PostgresIOManager

        with patch("moncpipelib.resources.postgres.psycopg.connect") as mock_connect:
            io_manager = PostgresIOManager(postgres_resource=resource)
            io_manager.postgres_resource._make_check_connection_factory()()
            assert "application_name" in mock_connect.call_args.kwargs
            assert "options" in mock_connect.call_args.kwargs
