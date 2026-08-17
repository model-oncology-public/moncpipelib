"""Polars ``schema_overrides`` builder + psycopg type-loader registration.

Co-locates :class:`PostgresPolarsSchema` (OID-to-dtype mapping; cursor / SA
probes) with the small psycopg ``Loader`` subclasses that back it.  Lives in
its own module so the postgres resource module is not weighed down by the
~225-line schema class plus its private loader classes -- the resource just
re-exports the public symbols for callers that import by the historical
``moncpipelib.resources.postgres`` path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import polars as pl
import psycopg
import sqlalchemy as sa
from psycopg.adapt import Loader as _PgLoader


class _StringJsonLoader(_PgLoader):
    """Pass JSON / JSONB wire bytes through as a Python ``str``.

    psycopg3's default ``JsonLoader`` / ``JsonbLoader`` call ``json.loads``
    to return parsed Python objects (dicts / lists).  The Polars
    contract-check path requires raw strings -- parsed JSONB arrays of
    varying lengths produce ``pl.Object`` and break ``pl.read_database``
    schema inference (#118).
    """

    def load(self, data: bytes | bytearray | memoryview | None) -> str | None:
        return bytes(data).decode("utf-8") if data is not None else None


class _StringUUIDLoader(_PgLoader):
    """Pass UUID wire bytes through as a Python ``str``.

    psycopg3's default ``UUIDLoader`` returns ``uuid.UUID`` objects.
    Polars stores those as ``pl.Object`` with no path to ``pl.String``.
    This loader yields the canonical hyphenated lowercase string form.
    """

    def load(self, data: bytes | bytearray | memoryview | None) -> str | None:
        return bytes(data).decode("utf-8") if data is not None else None


def restore_default_handlers(connection: psycopg.Connection) -> None:
    """Restore psycopg3's default JSON / JSONB / UUID handlers on a connection.

    Reverses :meth:`PostgresPolarsSchema.register_uuid_adapter` /
    :meth:`PostgresPolarsSchema.register_json_adapters`: callers that
    need parsed Python objects (instead of raw strings) re-register the
    stock loaders.  Used by the opt-out path in
    ``tests/integration/test_json_polars.py``.

    Mock-tolerant: short-circuits when the connection lacks an
    ``adapters`` attribute (e.g. ``MagicMock`` instances).
    """
    adapters = getattr(connection, "adapters", None)
    if adapters is None:
        return
    # Lazy imports: psycopg's default loader classes live in submodules
    # that are otherwise irrelevant to the read path.
    from psycopg.types.json import JsonbLoader, JsonLoader
    from psycopg.types.uuid import UUIDLoader

    adapters.register_loader("json", JsonLoader)
    adapters.register_loader("jsonb", JsonbLoader)
    adapters.register_loader("uuid", UUIDLoader)


class PostgresPolarsSchema:
    """Maps PostgreSQL column types to Polars dtypes for deterministic schema inference.

    Provides a central OID-to-dtype mapping and classmethods to probe a live
    connection and return a ``schema_overrides`` dict ready for
    ``pl.read_database``.  Using explicit overrides prevents Polars from
    inferring column types from sampled data, which eliminates cross-batch type
    mismatches and ensures UUID columns are always read as ``pl.String`` rather
    than ``pl.Object``.

    Importantly, this class also provides :meth:`register_uuid_adapter` and
    :meth:`register_json_adapters` which **must** be called on every psycopg
    connection before reading data into Polars.  Without them, psycopg
    returns ``uuid.UUID`` objects and deserialized JSON dicts/lists that Polars
    cannot handle consistently -- ``schema_overrides`` alone cannot fix this.

    Attributes:
        OID_MAP: Mapping of common PostgreSQL type OIDs to Polars data types.

    Example:
        ```python
        from moncpipelib.resources.postgres import PostgresPolarsSchema

        with database.get_connection() as conn:
            PostgresPolarsSchema.register_uuid_adapter(conn)
            overrides = PostgresPolarsSchema.from_psycopg2_connection(conn, query)
            df = pl.read_database(query, conn, schema_overrides=overrides)
        ```
    """

    OID_MAP: dict[int, type[pl.DataType]] = {
        16: pl.Boolean,  # bool
        17: pl.Binary,  # bytea
        20: pl.Int64,  # int8 (bigint)
        21: pl.Int16,  # int2 (smallint)
        23: pl.Int32,  # int4 (integer)
        25: pl.String,  # text
        114: pl.String,  # json
        700: pl.Float32,  # float4
        701: pl.Float64,  # float8
        1042: pl.String,  # bpchar (char)
        1043: pl.String,  # varchar
        1082: pl.Date,  # date
        1114: pl.Datetime,  # timestamp
        1184: pl.Datetime,  # timestamptz
        1700: pl.Float64,  # numeric/decimal
        2950: pl.String,  # uuid
        3802: pl.String,  # jsonb
    }

    @classmethod
    def register_uuid_adapter(cls, connection: psycopg.Connection) -> None:
        """Register UUID / JSON / JSONB string-passthrough loaders on a connection.

        Registers :class:`_StringJsonLoader` for OIDs ``json`` and
        ``jsonb``, and :class:`_StringUUIDLoader` for OID ``uuid`` on
        the connection's adapter registry.  ``register_loader`` is
        idempotent (re-registering replaces the prior loader).

        The classmethod name is preserved for back-compat -- callers
        that previously used this for "UUID only" semantics now also
        get JSON / JSONB string passthrough, which is what every real
        call site already paired this with.

        Mock-tolerant: short-circuits on objects without an
        ``adapters`` attribute (e.g. ``MagicMock`` instances in unit
        tests).

        Args:
            connection: A psycopg connection object.  The loader is
                registered per-connection to avoid global side
                effects.
        """
        adapters = getattr(connection, "adapters", None)
        if adapters is None:
            return
        adapters.register_loader("json", _StringJsonLoader)
        adapters.register_loader("jsonb", _StringJsonLoader)
        adapters.register_loader("uuid", _StringUUIDLoader)

    @classmethod
    def register_uuid_adapter_sa(cls, sa_conn: sa.engine.Connection) -> None:
        """Register the type loaders on a SQLAlchemy connection.

        Extracts the underlying psycopg DBAPI connection from a
        SQLAlchemy ``Connection`` and registers the loaders on it.
        Load-bearing for the ``read_batched`` / ``iter_batches=True``
        path -- without this registration on the SA-managed
        connection, JSONB columns come through as parsed Python
        objects and break Polars schema inference.

        Args:
            sa_conn: A SQLAlchemy connection (wrapping psycopg).
        """
        try:
            dbapi_conn = sa_conn.connection.dbapi_connection
        except Exception:
            return
        if dbapi_conn is None:
            return
        # SQLAlchemy types ``dbapi_connection`` as the abstract
        # ``DBAPIConnection`` protocol; under our ``postgresql+psycopg``
        # dialect it is a real ``psycopg.Connection`` and the
        # mock-tolerance short-circuit on ``adapters`` covers any
        # exotic case.
        cls.register_uuid_adapter(cast("psycopg.Connection", dbapi_conn))

    @classmethod
    def register_json_adapters(cls, connection: psycopg.Connection) -> None:
        """Register UUID / JSON / JSONB string-passthrough loaders on a connection.

        Equivalent to :meth:`register_uuid_adapter` -- both classmethods
        register the same set of loaders.  The two methods are kept
        separate for back-compat with callers that group registration
        semantically (UUID registration vs JSON registration).
        Re-registration is idempotent.

        If a caller needs parsed Python objects on a specific
        connection, restore the driver's default loaders via
        :func:`restore_default_handlers`.  Polars'
        ``str.json_decode()`` is preferred for typed struct/list
        parsing on the result DataFrame.

        Args:
            connection: A psycopg connection object.
        """
        cls.register_uuid_adapter(connection)

    @classmethod
    def register_json_adapters_sa(cls, sa_conn: sa.engine.Connection) -> None:
        """Register the type loaders on a SQLAlchemy connection.

        Equivalent to :meth:`register_uuid_adapter_sa`; the alias is
        preserved for back-compat (callers can pair registration
        semantically by intent).

        Args:
            sa_conn: A SQLAlchemy connection (wrapping psycopg).
        """
        cls.register_uuid_adapter_sa(sa_conn)

    @classmethod
    def from_cursor_description(
        cls,
        description: Sequence[Any],
    ) -> dict[str, type[pl.DataType]]:
        """Build a schema_overrides dict from a DBAPI cursor.description.

        Each entry in *description* is a 7-tuple where index 0 is the column
        name and index 1 is the PostgreSQL type OID.  Columns whose OID is not
        in ``OID_MAP`` are omitted so Polars will infer those types.

        Args:
            description: DBAPI cursor.description sequence.

        Returns:
            dict mapping column names to Polars data types.
        """
        schema: dict[str, type[pl.DataType]] = {}
        for col_desc in description:
            name: str = col_desc[0]
            oid: int = col_desc[1]
            pl_type = cls.OID_MAP.get(oid)
            if pl_type is not None:
                schema[name] = pl_type
        return schema

    @classmethod
    def from_sa_connection(
        cls,
        conn: sa.engine.Connection,
        query: str,
    ) -> dict[str, type[pl.DataType]] | None:
        """Probe a SQLAlchemy connection and return schema_overrides.

        Executes a ``LIMIT 0`` probe query to read column OIDs without
        fetching data, then maps them to Polars types.

        Args:
            conn: SQLAlchemy connection.
            query: SQL SELECT query whose result schema should be probed.

        Returns:
            dict mapping column names to Polars data types, or None if the
            probe fails or yields no mappable columns.
        """
        try:
            result = conn.execute(sa.text(f"SELECT * FROM ({query}) AS _schema_probe LIMIT 0"))  # noqa: S608
            cursor_desc = result.cursor.description
            result.close()
            if cursor_desc:
                return cls.from_cursor_description(cursor_desc) or None
        except Exception:
            pass
        return None

    @classmethod
    def from_psycopg2_connection(
        cls,
        connection: psycopg.Connection,
        query: str,
    ) -> dict[str, type[pl.DataType]] | None:
        """Probe a psycopg2 connection and return schema_overrides.

        Executes a ``LIMIT 0`` probe query to read column OIDs without
        fetching data, then maps them to Polars types.

        Args:
            connection: psycopg2 connection.
            query: SQL SELECT query whose result schema should be probed.

        Returns:
            dict mapping column names to Polars data types, or None if the
            probe fails or yields no mappable columns.
        """
        try:
            with connection.cursor() as cur:
                cur.execute(f"SELECT * FROM ({query}) AS _schema_probe LIMIT 0")  # noqa: S608
                if cur.description:
                    return cls.from_cursor_description(cur.description) or None
        except Exception:
            pass
        return None
