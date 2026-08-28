#!/usr/bin/env python3
"""Reproducer / bench harness for issue #294: per-tx work_mem on reconcile_scd2.

Stages a copy of ``reference_silver.npi_address`` into a bench schema, then
runs ``PostgresResource.reconcile_scd2()`` twice back-to-back against the
staged copy:

  1. with ``work_mem=None`` (cluster default, typically 32 MB on npe).
  2. with ``work_mem="256MB"`` (the new resource-field default).

Reconcile is idempotent: once the table is collapsed and timeline-stitched,
the second run finds nothing to write but still pays the full sort cost
(ROW_NUMBER / LEAD over the whole table). That makes the back-to-back
comparison clean -- the only knob is ``work_mem``.

Polls ``pg_stat_activity`` from a sidecar thread every 2s while each
statement runs and records the wait_event distribution, so the
spill-vs-CPU profile is captured alongside wall time.

Usage:
    # Set write credentials for pg-nonprod (or wherever you want
    # the bench to run).
    export PGHOST=pg-nonprod.example.com
    export PGUSER=bench_user
    export PGPASSWORD=...
    export PGDATABASE=analytics
    # Optional: override the default sandbox target.  Default lands the
    # bench copy in reference_sandbox (an existing writable schema on npe
    # for bench_user), which avoids needing CREATE on the database.
    export BENCH_SCHEMA=reference_sandbox       # default
    export BENCH_TABLE=npi_address_bench_294    # default
    export BENCH_LIMIT=20000000                 # optional; cap stage row count

    # Stage the fixture (idempotent, safe to re-run; uses CREATE TABLE
    # IF NOT EXISTS + INSERT ... SELECT skipped if the row count already
    # matches the source). On a 25 GB source this is the slow part of the
    # whole bench; budget ~10-30 minutes the first time.
    uv run python scripts/bench_reconcile_work_mem.py stage

    # Run the comparison.
    uv run python scripts/bench_reconcile_work_mem.py bench

    # Drop the staged fixture when done.
    uv run python scripts/bench_reconcile_work_mem.py teardown

The script never touches the production-shape source table; it only reads
from it during stage. The bench schema is created fresh and dropped on
teardown.

Output: a Markdown table you can paste into the PR / issue, plus a JSON
file at ``$BENCH_OUTPUT`` (default: ``bench_work_mem_results.json``).

Compliance: ``reference_silver.npi_address`` is provider reference data
(public NPI registry shape). The bench schema is private to the running
backend; teardown drops it. No PHI handled by this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any

import psycopg

# Resource is imported lazily in `bench` so `stage` and `teardown` can run
# even if the moncpipelib environment is not fully wired up.

DEFAULT_SOURCE_SCHEMA = "reference_silver"
DEFAULT_SOURCE_TABLE = "npi_address"
DEFAULT_BENCH_SCHEMA = "reference_sandbox"
DEFAULT_BENCH_TABLE = "npi_address_bench_294"
DEFAULT_OUTPUT = "bench_work_mem_results.json"
POLL_INTERVAL_SEC = 2.0


def _lookup_pgpass(host: str, port: str, dbname: str, user: str) -> str | None:
    """Look up a password from ``~/.pgpass`` matching the connection tuple.

    libpq itself consults pgpass when the password isn't supplied, but
    ``PostgresResource(password: str)`` requires an explicit value, so the
    bench harness reads pgpass directly to forward the right password into
    the resource constructor.
    """
    path = os.path.expanduser("~/.pgpass")
    if not os.path.isfile(path):
        return None
    if (os.stat(path).st_mode & 0o077) != 0:
        # libpq itself ignores pgpass with permissive modes; mirror that.
        print(f"[pgpass] {path} permissions too open; ignoring", file=sys.stderr)
        return None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) != 5:
                continue
            f_host, f_port, f_db, f_user, f_pass = parts
            if (
                (f_host in (host, "*"))
                and (f_port in (port, "*"))
                and (f_db in (dbname, "*"))
                and (f_user in (user, "*"))
            ):
                return f_pass
    return None


def _conn_kwargs() -> dict[str, Any]:
    """Build psycopg connect kwargs.

    Resolves the password from ``PGPASSWORD`` when set, otherwise from
    ``~/.pgpass``.  Raises if neither is available -- the bench needs an
    explicit password to construct ``PostgresResource``.
    """
    host = os.environ["PGHOST"]
    port = os.environ.get("PGPORT", "5432")
    dbname = os.environ.get("PGDATABASE", "analytics")
    user = os.environ["PGUSER"]
    password = os.environ.get("PGPASSWORD") or _lookup_pgpass(host, port, dbname, user)
    if not password:
        raise RuntimeError(
            f"no password resolved for user={user!r} host={host!r} db={dbname!r}: "
            "set PGPASSWORD or add a matching line to ~/.pgpass"
        )
    return {
        "host": host,
        "port": int(port),
        "user": user,
        "password": password,
        "dbname": dbname,
        "sslmode": os.environ.get("PGSSLMODE", "require"),
    }


def _bench_target() -> tuple[str, str]:
    return (
        os.environ.get("BENCH_SCHEMA", DEFAULT_BENCH_SCHEMA),
        os.environ.get("BENCH_TABLE", DEFAULT_BENCH_TABLE),
    )


def _source_target() -> tuple[str, str]:
    return (
        os.environ.get("BENCH_SOURCE_SCHEMA", DEFAULT_SOURCE_SCHEMA),
        os.environ.get("BENCH_SOURCE_TABLE", DEFAULT_SOURCE_TABLE),
    )


@contextmanager
def _connect():
    conn = psycopg.connect(**_conn_kwargs())
    try:
        yield conn
    finally:
        conn.close()


def cmd_stage(_args: argparse.Namespace) -> int:
    """Stage a copy of the source table into the bench schema."""
    schema, table = _bench_target()
    src_schema, src_table = _source_target()
    bench_fqn = f"{schema}.{table}"
    src_fqn = f"{src_schema}.{src_table}"
    limit = os.environ.get("BENCH_LIMIT")

    with _connect() as conn, conn.cursor() as cur:
        # Default ``reference_sandbox`` already exists on the npe cluster;
        # only attempt CREATE SCHEMA if the user has overridden BENCH_SCHEMA
        # to something that doesn't exist yet (avoids needing CREATE on the
        # database itself).
        cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))
        if cur.fetchone() is None:
            print(f"[stage] creating schema {schema}")
            cur.execute(f"CREATE SCHEMA {schema}")
            conn.commit()
        cur.execute(
            "SELECT to_regclass(%s)",
            (bench_fqn,),
        )
        row = cur.fetchone()
        already_exists = row is not None and row[0] is not None

        if not already_exists:
            print(f"[stage] creating {bench_fqn} (LIKE {src_fqn} INCLUDING ALL)")
            cur.execute(
                f"CREATE TABLE {bench_fqn} "
                f"(LIKE {src_fqn} INCLUDING DEFAULTS INCLUDING IDENTITY "
                "INCLUDING CONSTRAINTS INCLUDING INDEXES)"
            )
            conn.commit()
        else:
            print(f"[stage] {bench_fqn} already exists, skipping CREATE")

        cur.execute(f"SELECT count(*) FROM {bench_fqn}")
        bench_rows = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {src_fqn}")
        src_rows = cur.fetchone()[0]

        print(f"[stage] source rows: {src_rows:,}; bench rows: {bench_rows:,}")

        if bench_rows == 0:
            if limit:
                print(f"[stage] inserting up to {int(limit):,} rows from {src_fqn}")
                cur.execute(
                    f"INSERT INTO {bench_fqn} SELECT * FROM {src_fqn} LIMIT %s",
                    (int(limit),),
                )
            else:
                print(f"[stage] inserting all rows from {src_fqn}")
                cur.execute(f"INSERT INTO {bench_fqn} SELECT * FROM {src_fqn}")
            conn.commit()
            cur.execute(f"SELECT count(*) FROM {bench_fqn}")
            bench_rows = cur.fetchone()[0]
            print(f"[stage] inserted; bench rows now: {bench_rows:,}")
        else:
            print("[stage] bench non-empty; skipping INSERT")

        print(f"[stage] ANALYZE {bench_fqn}")
        cur.execute(f"ANALYZE {bench_fqn}")
        conn.commit()

        cur.execute(
            "SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
            (bench_fqn,),
        )
        size = cur.fetchone()[0]
        print(f"[stage] {bench_fqn}: {bench_rows:,} rows, {size}")

    return 0


def cmd_teardown(_args: argparse.Namespace) -> int:
    """Drop the bench table (leaving the sandbox schema intact)."""
    schema, table = _bench_target()
    bench_fqn = f"{schema}.{table}"
    with _connect() as conn, conn.cursor() as cur:
        print(f"[teardown] DROP TABLE {bench_fqn}")
        cur.execute(f"DROP TABLE IF EXISTS {bench_fqn}")
        conn.commit()
    return 0


def _build_resource():
    from moncpipelib.resources.postgres import PostgresResource

    kwargs = _conn_kwargs()
    return PostgresResource(
        host=kwargs["host"],
        port=kwargs["port"],
        user=kwargs["user"],
        password=kwargs["password"],
        database=kwargs["dbname"],
        sslmode=kwargs["sslmode"],
        # Bench-specific tuning: disable lineage / openlineage / contracts so
        # the only thing exercised is the reconcile path itself.
        enable_row_lineage=False,
        enforce_contracts="silent",
    )


def _poll_query_backends(target_table: str, stop_event: threading.Event) -> list[dict[str, Any]]:
    """Poll pg_stat_activity for any backend executing reconcile against the target."""
    samples: list[dict[str, Any]] = []
    try:
        conn = psycopg.connect(**_conn_kwargs())
    except Exception as e:
        print(f"[poll] could not connect: {e}", file=sys.stderr)
        return samples
    try:
        while not stop_event.is_set():
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT now(), pid, state, wait_event_type, wait_event, "
                        "  substring(query for 200) "
                        "FROM pg_stat_activity "
                        "WHERE query ILIKE %s "
                        "  AND state = 'active' "
                        "  AND pid <> pg_backend_pid()",
                        (f"%{target_table}%",),
                    )
                    rows = cur.fetchall()
                conn.rollback()
                for now, pid, state, wt, we, q in rows:
                    samples.append(
                        {
                            "ts": now.isoformat(),
                            "pid": pid,
                            "state": state,
                            "wait_event_type": wt,
                            "wait_event": we,
                            "query_prefix": (q or "")[:120],
                        }
                    )
            except Exception as e:
                print(f"[poll] error: {e}", file=sys.stderr)
            stop_event.wait(POLL_INTERVAL_SEC)
    finally:
        conn.close()
    return samples


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    wait_events: Counter[str] = Counter()
    for s in samples:
        wait_events[s["wait_event"] or "<active-no-wait>"] += 1
    return {
        "sample_count": len(samples),
        "wait_event_distribution": dict(wait_events.most_common()),
    }


def _run_one_reconcile(
    resource: Any,
    target_fqn: str,
    work_mem: str | None,
) -> dict[str, Any]:
    print(f"\n[bench] reconcile_scd2({target_fqn}, work_mem={work_mem!r})")
    stop = threading.Event()
    samples_holder: list[list[dict[str, Any]]] = [[]]

    def _poll() -> None:
        samples_holder[0] = _poll_query_backends(target_fqn.split(".", 1)[1], stop)

    poll_thread = threading.Thread(target=_poll, name="bench-poller", daemon=True)
    poll_thread.start()

    started = time.monotonic()
    started_wall = dt.datetime.now(dt.UTC).isoformat()
    err: str | None = None
    result: dict[str, int] | None = None
    try:
        result = resource.reconcile_scd2(
            target=target_fqn,
            business_key=["npi", "address_type", "sequence"],
            work_mem=work_mem,
            run_id=f"bench:{target_fqn}",
        )
    except Exception as exc:
        err = repr(exc)
    elapsed = time.monotonic() - started
    stop.set()
    poll_thread.join(timeout=10)

    summary = _summarize_samples(samples_holder[0])
    print(f"[bench] elapsed: {elapsed:,.1f}s ({elapsed / 60:.2f} min)")
    print(f"[bench] result: {result}")
    print(f"[bench] wait_event distribution: {summary.get('wait_event_distribution')}")
    if err:
        print(f"[bench] error: {err}", file=sys.stderr)
    return {
        "work_mem": work_mem,
        "target": target_fqn,
        "started_utc": started_wall,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "reconcile_result": result,
        "error": err,
        "polling": summary,
        "raw_samples": samples_holder[0],
    }


def cmd_bench(_args: argparse.Namespace) -> int:
    """Run two reconciles back-to-back: cluster default, then 256MB."""
    schema, table = _bench_target()
    target_fqn = f"{schema}.{table}"
    output_path = os.environ.get("BENCH_OUTPUT", DEFAULT_OUTPUT)

    resource = _build_resource()

    # Sanity check: row count, size, current cluster work_mem.
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {target_fqn}")
        row_count = cur.fetchone()[0]
        cur.execute(
            "SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
            (target_fqn,),
        )
        size = cur.fetchone()[0]
        cur.execute("SHOW work_mem")
        cluster_work_mem = cur.fetchone()[0]

    print(f"[bench] target: {target_fqn}")
    print(f"[bench] row count: {row_count:,}")
    print(f"[bench] total size: {size}")
    print(f"[bench] cluster default work_mem: {cluster_work_mem}")

    # BENCH_WORK_MEMS=256MB,1GB or =256MB lets a caller pin the bench to a
    # specific subset (e.g. when re-running just the bumped pass after a
    # canceled baseline).  Empty values map to None (cluster default).
    work_mems_env = os.environ.get("BENCH_WORK_MEMS")
    if work_mems_env:
        work_mems: list[str | None] = [(v.strip() or None) for v in work_mems_env.split(",")]
    else:
        work_mems = [None, "256MB"]
    runs = []
    for work_mem in work_mems:
        runs.append(_run_one_reconcile(resource, target_fqn, work_mem))

    output = {
        "issue": "moncpipelib#294",
        "target": target_fqn,
        "row_count": row_count,
        "total_size": size,
        "cluster_default_work_mem": cluster_work_mem,
        "runs": runs,
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[bench] wrote {output_path}")

    # Markdown table for easy paste into the PR / issue.
    print("\n=== Markdown summary ===")
    print("| work_mem | wall time | wall time (min) | result |")
    print("|---|---|---|---|")
    for r in runs:
        wm = r["work_mem"] if r["work_mem"] else f"cluster default ({cluster_work_mem})"
        print(
            f"| {wm} | {r['elapsed_seconds']:.1f}s | "
            f"{r['elapsed_minutes']:.2f} | {r['reconcile_result']} |"
        )

    if (
        len(runs) == 2
        and all(r["error"] is None for r in runs)
        and runs[0]["work_mem"] is None
        and runs[1]["work_mem"] == "256MB"
    ):
        baseline = runs[0]["elapsed_seconds"]
        bumped = runs[1]["elapsed_seconds"]
        delta = (baseline - bumped) / baseline * 100
        print(f"\n[bench] wall-time delta (256MB vs cluster default): {delta:+.1f}%")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", help="Stage a bench copy of the source table")
    sub.add_parser("bench", help="Run the two-pass reconcile comparison")
    sub.add_parser("teardown", help="Drop the bench schema")
    args = parser.parse_args()

    cmd = {
        "stage": cmd_stage,
        "bench": cmd_bench,
        "teardown": cmd_teardown,
    }[args.cmd]
    return cmd(args)


if __name__ == "__main__":
    sys.exit(main())
