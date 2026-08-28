#!/usr/bin/env python3
"""Bench harness for issue #375: per-row executemany upsert vs staging-COPY + merge.

Quantifies the win from lever 2 in #375 -- replacing the current per-row
``INSERT ... ON CONFLICT`` (``cursor.executemany``, one audited statement per
row) with a COPY into a temp staging table followed by a single
``INSERT ... SELECT DISTINCT ON (pk) ... ON CONFLICT`` merge (two audited
statements regardless of row count, the same shape SCD2 already uses).

It runs BOTH strategies against the same staged sandbox target, from the same
client-side row batch, and records for each:

  * wall time (throughput; fewer round-trips should be faster),
  * pg_stat_activity wait-event distribution (the #260 ClientRead angle --
    the per-row path should show ClientRead chatter the COPY path does not),
  * statement count actually executed = the pgAudit WRITE-line proxy
    (executemany: one per row; staging-merge: COPY + merge = 2),
  * a post-state checksum, asserted equal across strategies (correctness:
    proves the staging dedup reproduces the per-row last-write-wins result and
    does NOT raise "ON CONFLICT DO UPDATE command cannot affect row a second
    time" on in-batch duplicate keys).

The candidate merge SQL drafted here is the prototype for the library change;
the bench is the place to get the ``DISTINCT ON (pk) ... ORDER BY pk, _ord
DESC`` last-input-wins dedup right before it lands in ``writers.py``.

Workload shape
--------------
``stage`` builds a baseline copy of a source table. Each ``bench`` run resets
the target to the first ``BENCH_SEED_FRAC`` of the baseline, then upserts the
*entire* baseline as the incoming batch -- so seeded keys are UPDATEs and the
remainder are INSERTs (a realistic mixed upsert load). A small block of exact
in-batch duplicate keys is appended to exercise the staging dedup / double-
affect path that is the headline correctness risk.

Usage
-----
    export PGHOST=pg-nonprod.example.com
    export PGUSER=bench_user                 # needs CREATE on the sandbox schema
    export PGPASSWORD=...                  # or ~/.pgpass
    export PGDATABASE=analytics
    # Source must be a non-PHI table with an explicit (non-identity) PK.
    export BENCH_SOURCE_SCHEMA=synthetic_gold
    export BENCH_SOURCE_TABLE=acme_fact_drug_revenue_line
    export BENCH_SCHEMA=reference_sandbox          # writable sandbox; default
    export BENCH_TABLE=ups_bench_375               # default
    export BENCH_LIMIT=200000                      # optional cap on baseline rows
    export BENCH_SEED_FRAC=0.5                     # fraction pre-seeded => updates
    export BENCH_DUP_ROWS=500                      # in-batch duplicate keys to inject

    uv run python scripts/bench_upsert_staging_merge.py stage
    uv run python scripts/bench_upsert_staging_merge.py bench
    uv run python scripts/bench_upsert_staging_merge.py teardown

Compliance
----------
Intended for synthetic / reference (non-PHI) source tables only; the harness
refuses to run against schemas outside an allowlist unless BENCH_ALLOW_PHI=1 is
set explicitly. The staging table is a session-private TEMP table dropped at
disconnect; the bench target lives in the sandbox schema and is dropped on
teardown. No logging configuration is touched -- this measures statement
*count*, the audit-trail content is unchanged (HIPAA 164.312(b) coverage
preserved; see #375).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any

import psycopg

DEFAULT_SOURCE_SCHEMA = "synthetic_gold"
DEFAULT_SOURCE_TABLE = "acme_fact_drug_revenue_line"
DEFAULT_BENCH_SCHEMA = "reference_sandbox"
DEFAULT_BENCH_TABLE = "ups_bench_375"
DEFAULT_OUTPUT = "bench_upsert_staging_merge_results.json"
POLL_INTERVAL_SEC = 0.5  # tighter than #294: per-row ClientRead spikes are brief
STAGE_TABLE = "_ups_stage_375"

# Schemas the bench is allowed to read a source table from without an explicit
# PHI override. Keep this conservative; synthetic_* / reference_* are non-PHI.
NON_PHI_SOURCE_PREFIXES = ("synthetic_", "reference_")


def _lookup_pgpass(host: str, port: str, dbname: str, user: str) -> str | None:
    """Resolve a password from ``~/.pgpass`` matching the connection tuple."""
    path = os.path.expanduser("~/.pgpass")
    if not os.path.isfile(path):
        return None
    if (os.stat(path).st_mode & 0o077) != 0:
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
    schema = os.environ.get("BENCH_SOURCE_SCHEMA", DEFAULT_SOURCE_SCHEMA)
    table = os.environ.get("BENCH_SOURCE_TABLE", DEFAULT_SOURCE_TABLE)
    if not schema.startswith(NON_PHI_SOURCE_PREFIXES) and os.environ.get("BENCH_ALLOW_PHI") != "1":
        raise RuntimeError(
            f"source schema {schema!r} is not in the non-PHI allowlist "
            f"{NON_PHI_SOURCE_PREFIXES}; set BENCH_ALLOW_PHI=1 to override "
            "(only if you are certain the bench data is not PHI)"
        )
    return (schema, table)


@contextmanager
def _connect():
    conn = psycopg.connect(**_conn_kwargs())
    try:
        yield conn
    finally:
        conn.close()


def _qident(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def _table_shape(cur: psycopg.Cursor, schema: str, table: str) -> tuple[list[str], list[str]]:
    """Return (insertable_columns, primary_key_columns) for a table.

    Insertable columns exclude GENERATED ALWAYS / identity columns, which the
    upsert path neither writes nor copies.
    """
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
          AND is_generated = 'NEVER'
          AND (is_identity = 'NO' OR identity_generation IS DISTINCT FROM 'ALWAYS')
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    columns = [r[0] for r in cur.fetchall()]

    cur.execute(
        """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """,
        (f"{schema}.{table}",),
    )
    pk = [r[0] for r in cur.fetchall()]
    if not pk:
        raise RuntimeError(f"{schema}.{table} has no primary key; cannot bench upsert")
    if not columns:
        raise RuntimeError(f"{schema}.{table} exposes no insertable columns")
    return columns, pk


def cmd_stage(_args: argparse.Namespace) -> int:
    """Stage a baseline copy of the source table into the bench schema."""
    schema, table = _bench_target()
    src_schema, src_table = _source_target()
    bench_fqn = f"{_qident(schema)}.{_qident(table)}"
    src_fqn = f"{_qident(src_schema)}.{_qident(src_table)}"
    limit = os.environ.get("BENCH_LIMIT")

    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))
        if cur.fetchone() is None:
            print(f"[stage] creating schema {schema}")
            cur.execute(f"CREATE SCHEMA {_qident(schema)}")
            conn.commit()

        cur.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        row = cur.fetchone()
        if row is None or row[0] is None:
            print(f"[stage] creating {bench_fqn} (LIKE {src_fqn} INCLUDING ALL)")
            cur.execute(
                f"CREATE TABLE {bench_fqn} (LIKE {src_fqn} "
                "INCLUDING DEFAULTS INCLUDING IDENTITY INCLUDING CONSTRAINTS "
                "INCLUDING INDEXES)"
            )
            conn.commit()
        else:
            print(f"[stage] {bench_fqn} already exists, skipping CREATE")

        cur.execute(f"SELECT count(*) FROM {bench_fqn}")
        bench_rows = cur.fetchone()[0]
        if bench_rows == 0:
            if limit:
                print(f"[stage] inserting up to {int(limit):,} rows from {src_fqn}")
                cur.execute(f"INSERT INTO {bench_fqn} SELECT * FROM {src_fqn} LIMIT %s", (int(limit),))
            else:
                print(f"[stage] inserting all rows from {src_fqn}")
                cur.execute(f"INSERT INTO {bench_fqn} SELECT * FROM {src_fqn}")
            conn.commit()
            cur.execute(f"SELECT count(*) FROM {bench_fqn}")
            bench_rows = cur.fetchone()[0]
        else:
            print("[stage] bench non-empty; skipping INSERT")

        cur.execute(f"ANALYZE {bench_fqn}")
        conn.commit()
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))", (f"{schema}.{table}",))
        size = cur.fetchone()[0]
        print(f"[stage] {bench_fqn}: {bench_rows:,} rows, {size}")
    return 0


def cmd_teardown(_args: argparse.Namespace) -> int:
    """Drop the bench table (leaving the sandbox schema intact)."""
    schema, table = _bench_target()
    bench_fqn = f"{_qident(schema)}.{_qident(table)}"
    baseline_fqn = f"{_qident(schema)}.{_qident(table + '__baseline')}"
    with _connect() as conn, conn.cursor() as cur:
        print(f"[teardown] DROP TABLE {bench_fqn} (+ __baseline)")
        cur.execute(f"DROP TABLE IF EXISTS {bench_fqn}")
        cur.execute(f"DROP TABLE IF EXISTS {baseline_fqn}")
        conn.commit()
    return 0


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _poll_query_backends(target_table: str, stop_event: threading.Event) -> list[dict[str, Any]]:
    """Poll pg_stat_activity for backends touching the bench target."""
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
                        "  substring(query for 120) "
                        "FROM pg_stat_activity "
                        "WHERE query ILIKE %s AND state = 'active' "
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
        return {"sample_count": 0, "wait_event_distribution": {}}
    wait_events: Counter[str] = Counter()
    for s in samples:
        wait_events[s["wait_event"] or "<active-no-wait>"] += 1
    return {
        "sample_count": len(samples),
        "wait_event_distribution": dict(wait_events.most_common()),
    }


def _reset_target(
    conn: psycopg.Connection,
    bench_fqn: str,
    baseline_fqn: str,
    columns: list[str],
    pk: list[str],
    seed_frac: float,
) -> int:
    """Reset the target to the first ``seed_frac`` of the persistent baseline.

    The baseline is an immutable sandbox copy of the staged table (created once
    per bench run); re-seeding the live target from it makes every strategy
    start from identical state regardless of the prior run's mutations.
    """
    col_list = ", ".join(_qident(c) for c in columns)
    order_list = ", ".join(_qident(c) for c in pk)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {baseline_fqn}")
        total = cur.fetchone()[0]
        seed_n = int(total * seed_frac)
        cur.execute(f"TRUNCATE {bench_fqn}")
        cur.execute(
            f"INSERT INTO {bench_fqn} ({col_list}) "
            f"SELECT {col_list} FROM {baseline_fqn} ORDER BY {order_list} LIMIT %s",  # noqa: S608
            (seed_n,),
        )
    conn.commit()
    return seed_n


def _incoming_rows(
    conn: psycopg.Connection, baseline_fqn: str, columns: list[str], pk: list[str], dup_rows: int
) -> list[tuple[Any, ...]]:
    """Materialize the incoming batch (all baseline rows + injected dupes), ordered.

    Returns rows as ``(*column_values, _ord)`` with a stable input ordinal so
    both strategies resolve duplicates last-input-wins identically.
    """
    col_list = ", ".join(_qident(c) for c in columns)
    order_list = ", ".join(_qident(c) for c in pk)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {col_list} FROM {baseline_fqn} ORDER BY {order_list}")  # noqa: S608
        base = cur.fetchall()
    rows: list[tuple[Any, ...]] = [(*r, i) for i, r in enumerate(base)]
    # Append exact in-batch duplicates of the first ``dup_rows`` keys with a
    # higher ordinal so last-input-wins is exercised (and the staging merge's
    # DISTINCT ON must collapse them rather than raising double-affect).
    for j in range(min(dup_rows, len(base))):
        rows.append((*base[j], len(base) + j))
    return rows


def _checksum(conn: psycopg.Connection, bench_fqn: str, columns: list[str], pk: list[str]) -> dict[str, Any]:
    """Order-independent content checksum of the target for cross-strategy equality."""
    col_list = ", ".join(_qident(c) for c in columns)
    order_list = ", ".join(_qident(c) for c in pk)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {bench_fqn}")
        n = cur.fetchone()[0]
        cur.execute(
            f"SELECT md5(string_agg(rowhash, '' ORDER BY rowhash)) FROM "
            f"(SELECT md5(ROW({col_list})::text) AS rowhash FROM {bench_fqn}) s"  # noqa: S608
        )
        digest = cur.fetchone()[0]
    return {"row_count": n, "content_md5": digest, "ordered_by": order_list}


def _run_executemany(
    conn: psycopg.Connection,
    bench_fqn: str,
    columns: list[str],
    pk: list[str],
    rows: list[tuple[Any, ...]],
) -> int:
    """Current path: one INSERT ... ON CONFLICT per row via executemany.

    Returns the executed-statement count (== rows, the pgAudit WRITE-line proxy).
    """
    col_list = ", ".join(_qident(c) for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_cols = [c for c in columns if c not in pk]
    set_clause = ", ".join(f"{_qident(c)} = EXCLUDED.{_qident(c)}" for c in update_cols)
    conflict = ", ".join(_qident(c) for c in pk)
    sql = (
        f"INSERT INTO {bench_fqn} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"
    )
    # Strip the trailing _ord ordinal; apply in input order so last-wins holds.
    payload = [r[:-1] for r in rows]
    with conn.cursor() as cur:
        cur.executemany(sql, payload)
    conn.commit()
    return len(payload)


def _run_staging_merge(
    conn: psycopg.Connection,
    bench_fqn: str,
    columns: list[str],
    pk: list[str],
    rows: list[tuple[Any, ...]],
) -> int:
    """Candidate path: COPY into temp staging, then one DISTINCT ON merge.

    Returns the executed-statement count (== 2: the COPY and the merge).
    """
    col_list = ", ".join(_qident(c) for c in columns)
    update_cols = [c for c in columns if c not in pk]
    set_clause = ", ".join(f"{_qident(c)} = EXCLUDED.{_qident(c)}" for c in update_cols)
    conflict = ", ".join(_qident(c) for c in pk)
    pk_list = ", ".join(_qident(c) for c in pk)
    stage = _qident(STAGE_TABLE)

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {stage}")
        # Staging carries the input columns plus an input-order ordinal; no PK
        # constraint so duplicates are allowed until the merge dedupes them.
        cur.execute(
            f"CREATE TEMP TABLE {stage} (LIKE {bench_fqn}) ON COMMIT DROP"
        )
        cur.execute(f"ALTER TABLE {stage} ADD COLUMN _ord bigint")

        # 1) COPY the whole batch into staging (one audited COPY statement).
        copy_cols = col_list + ", _ord"
        buf = io.StringIO()
        writer = csv.writer(buf)
        for r in rows:
            writer.writerow(["\\N" if v is None else v for v in r])
        buf.seek(0)
        with cur.copy(
            f"COPY {stage} ({copy_cols}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        ) as cp:
            cp.write(buf.read())

        # 2) One merge: dedupe staging last-input-wins, then upsert (one
        #    audited INSERT). DISTINCT ON keeps the highest _ord per key.
        cur.execute(
            f"INSERT INTO {bench_fqn} ({col_list}) "
            f"SELECT {col_list} FROM ("
            f"  SELECT DISTINCT ON ({pk_list}) {col_list} "
            f"  FROM {stage} ORDER BY {pk_list}, _ord DESC"
            f") d "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}"  # noqa: S608
        )
    conn.commit()
    return 2


STRATEGIES = {
    "executemany": _run_executemany,
    "staging_merge": _run_staging_merge,
}


def _run_strategy(
    name: str,
    bench_fqn: str,
    baseline_fqn: str,
    target_table: str,
    columns: list[str],
    pk: list[str],
    rows: list[tuple[Any, ...]],
    seed_frac: float,
) -> dict[str, Any]:
    print(f"\n[bench] strategy={name}")
    with _connect() as conn:
        seed_n = _reset_target(conn, bench_fqn, baseline_fqn, columns, pk, seed_frac)

        stop = threading.Event()
        holder: list[list[dict[str, Any]]] = [[]]

        def _poll() -> None:
            holder[0] = _poll_query_backends(target_table, stop)

        poll_thread = threading.Thread(target=_poll, name="bench-poller", daemon=True)
        poll_thread.start()

        started = time.monotonic()
        started_wall = dt.datetime.now(dt.UTC).isoformat()
        err: str | None = None
        stmt_count = 0
        try:
            stmt_count = STRATEGIES[name](conn, bench_fqn, columns, pk, rows)
        except Exception as exc:
            err = repr(exc)
        elapsed = time.monotonic() - started
        stop.set()
        poll_thread.join(timeout=10)

        checksum = None if err else _checksum(conn, bench_fqn, columns, pk)

    summary = _summarize_samples(holder[0])
    print(f"[bench] elapsed={elapsed:,.2f}s statements={stmt_count:,} seed_rows={seed_n:,}")
    print(f"[bench] wait_event distribution: {summary.get('wait_event_distribution')}")
    if err:
        print(f"[bench] error: {err}", file=sys.stderr)
    return {
        "strategy": name,
        "started_utc": started_wall,
        "elapsed_seconds": elapsed,
        "executed_statements": stmt_count,
        "seed_rows": seed_n,
        "error": err,
        "polling": summary,
        "checksum": checksum,
        "raw_samples": holder[0],
    }


def cmd_bench(_args: argparse.Namespace) -> int:
    schema, table = _bench_target()
    bench_fqn = f"{_qident(schema)}.{_qident(table)}"
    output_path = os.environ.get("BENCH_OUTPUT", DEFAULT_OUTPUT)
    seed_frac = float(os.environ.get("BENCH_SEED_FRAC", "0.5"))
    dup_rows = int(os.environ.get("BENCH_DUP_ROWS", "500"))

    with _connect() as conn, conn.cursor() as cur:
        columns, pk = _table_shape(cur, schema, table)
        cur.execute(f"SELECT count(*) FROM {bench_fqn}")
        baseline_rows = cur.fetchone()[0]
        cur.execute("SHOW work_mem")
        cluster_work_mem = cur.fetchone()[0]
    if baseline_rows == 0:
        print("[bench] baseline is empty; run `stage` first", file=sys.stderr)
        return 1

    print(f"[bench] target={bench_fqn} pk={pk} baseline_rows={baseline_rows:,}")
    print(f"[bench] insertable columns: {len(columns)}; seed_frac={seed_frac}; dup_rows={dup_rows}")

    # Persistent immutable baseline so every strategy (each on its own
    # connection) re-seeds from identical state. A TEMP table would not be
    # visible across the per-strategy connections.
    baseline_fqn = f"{_qident(schema)}.{_qident(table + '__baseline')}"
    col_list = ", ".join(_qident(c) for c in columns)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {baseline_fqn}")
            cur.execute(f"CREATE TABLE {baseline_fqn} AS SELECT {col_list} FROM {bench_fqn}")  # noqa: S608
        conn.commit()
        rows = _incoming_rows(conn, baseline_fqn, columns, pk, dup_rows)
    print(f"[bench] incoming batch rows (incl. {dup_rows} dupes): {len(rows):,}")

    try:
        runs = []
        for name in ("executemany", "staging_merge"):
            runs.append(
                _run_strategy(name, bench_fqn, baseline_fqn, table, columns, pk, rows, seed_frac)
            )
    finally:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {baseline_fqn}")
            conn.commit()

    output = {
        "issue": "moncpipelib#375",
        "target": f"{schema}.{table}",
        "baseline_rows": baseline_rows,
        "incoming_rows": len(rows),
        "seed_frac": seed_frac,
        "dup_rows": dup_rows,
        "cluster_work_mem": cluster_work_mem,
        "runs": runs,
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[bench] wrote {output_path}")

    _print_summary(runs)
    return 0


def _print_summary(runs: list[dict[str, Any]]) -> None:
    print("\n=== Markdown summary ===")
    print("| strategy | wall time | executed statements | ClientRead samples | error |")
    print("|---|---|---|---|---|")
    for r in runs:
        cr = r["polling"]["wait_event_distribution"].get("ClientRead", 0)
        print(
            f"| {r['strategy']} | {r['elapsed_seconds']:.2f}s | "
            f"{r['executed_statements']:,} | {cr} | {r['error'] or 'none'} |"
        )

    ok = [r for r in runs if r["error"] is None and r["checksum"]]
    if len(ok) == 2:
        same = ok[0]["checksum"]["content_md5"] == ok[1]["checksum"]["content_md5"]
        print(f"\n[bench] post-state checksums equal across strategies: {same}")
        if not same:
            print("[bench] WARNING: strategies produced different target state!", file=sys.stderr)
        em = next((r for r in runs if r["strategy"] == "executemany"), None)
        sm = next((r for r in runs if r["strategy"] == "staging_merge"), None)
        if em and sm and em["executed_statements"] and sm["executed_statements"]:
            ratio = em["executed_statements"] / sm["executed_statements"]
            print(
                f"[bench] statement-count reduction: {em['executed_statements']:,} -> "
                f"{sm['executed_statements']:,} ({ratio:,.0f}x fewer audited statements)"
            )
        if em and sm and sm["elapsed_seconds"]:
            speedup = em["elapsed_seconds"] / sm["elapsed_seconds"]
            print(f"[bench] wall-time speedup (executemany / staging_merge): {speedup:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", help="Stage a baseline bench copy of the source table")
    sub.add_parser("bench", help="Run the two-strategy upsert comparison")
    sub.add_parser("teardown", help="Drop the bench table")
    args = parser.parse_args()
    return {"stage": cmd_stage, "bench": cmd_bench, "teardown": cmd_teardown}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
