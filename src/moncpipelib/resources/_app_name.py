"""Resolve the Postgres ``application_name`` for run-to-backend correlation.

Dagster DB connections opened by :class:`~moncpipelib.resources.postgres.PostgresResource`
are tagged with the owning Dagster ``run_id`` so a backend in
``pg_stat_activity`` can be tied to the run that owns it. This is the
correlation key the data-platform zombie-backend reaper uses to terminate
backends whose owning run has reached a terminal state: when a run-worker pod
is torn down, the server-side query keeps executing until Postgres next
notices the dead client, pinning the xmin horizon and stalling reconciles.
Without ``application_name`` the reaper cannot positively tie a live backend
to a terminal run and must leave it alone. See issue #365.

Connections opened outside a Dagster run fall back to a stable identifier so
the column is never empty.

Security context: ``application_name`` is operational metadata only -- the
Dagster ``run_id`` is an opaque UUID, carries no PHI, and is already present
in ``pg_stat_activity`` indirectly (via the connecting service account). This
adds no new data exposure.
"""

from __future__ import annotations

import os
import re
import socket
from contextvars import ContextVar

_FALLBACK_APP_NAME = "moncpipelib"
"""Stable identifier for connections opened outside a Dagster run."""

# Postgres truncates application_name to NAMEDATALEN-1 (63) bytes. A Dagster
# run_id is a 36-char UUID so this is headroom-only, but clamp defensively so
# a stray long identifier can never error or get silently truncated server-side.
_MAX_APP_NAME_LEN = 63

_RUN_ID: ContextVar[str | None] = ContextVar("moncpipelib_dagster_run_id", default=None)
"""Run_id bound by the context-aware write / reconcile / IO-manager entry points."""

# The Dagster k8s run launcher names the run-worker pod
# ``dagster-run-<run_id>-<suffix>``; its hostname therefore carries the run_id
# even when no run context was bound (e.g. a connection opened by the
# run-worker outside a tracked write). Step-executor pods are named
# ``dagster-step-<hash>-…`` and do NOT encode the run_id -- those rely on
# :func:`bind_run_id` being called from the op/asset that opens the connection.
_RUN_WORKER_HOST_RE = re.compile(
    r"^dagster-run-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})-"
)


def bind_run_id(run_id: str | None) -> None:
    """Bind the Dagster ``run_id`` for connections opened on this execution.

    Called from the entry points that receive a Dagster context (the write,
    reconcile, and IO-manager paths) before any connection is opened. The
    connect sites then read it via :func:`resolve_application_name`.

    Idempotent and safe to call repeatedly; the most recent non-empty value
    wins. ``None`` / empty values are ignored so a context-less caller cannot
    clobber a run_id already bound within the same execution.
    """
    if run_id:
        _RUN_ID.set(run_id)


def resolve_application_name() -> str:
    """Resolve the ``application_name`` to stamp on a new connection.

    Resolution order (first hit wins):

    1. A ``run_id`` bound via :func:`bind_run_id` -- set by the write /
       reconcile / IO-manager entry points that receive a Dagster context.
       This is the reliable source for step-executor pods, whose hostname does
       not encode the run_id.
    2. The ``DAGSTER_RUN_ID`` environment variable, if the deployment exports
       it (future-proofing; not currently set in our run/step pods).
    3. The run_id parsed from a ``dagster-run-<run_id>-…`` pod hostname -- the
       k8s run-worker case, covering connections opened outside a bound
       context.
    4. The stable fallback ``"moncpipelib"``.

    Returns a value clamped to 63 bytes so it can never be rejected or
    silently truncated by Postgres.
    """
    bound = _RUN_ID.get()
    if bound:
        return bound[:_MAX_APP_NAME_LEN]

    env_run_id = os.environ.get("DAGSTER_RUN_ID")
    if env_run_id:
        return env_run_id[:_MAX_APP_NAME_LEN]

    match = _RUN_WORKER_HOST_RE.match(socket.gethostname())
    if match:
        return match.group(1)

    return _FALLBACK_APP_NAME
