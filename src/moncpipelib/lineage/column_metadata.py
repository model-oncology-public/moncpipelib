"""SQLAlchemy model for column-level metadata tracking (SCD2).

This table stores column classification metadata (PII, PHI, sensitivity, etc.)
using an SCD2 pattern for full audit history. Tags are stored as JSONB for
extensibility -- new classification types require no DDL changes.

Current state is queried with ``WHERE valid_to IS NULL``.
Point-in-time queries use ``WHERE valid_from <= ts AND (valid_to IS NULL OR valid_to > ts)``.

This table lives in the ``lineage`` schema alongside ``data_lineage``.
Consumer repos create the table via Alembic migration using this model as
the source of truth.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from moncpipelib.lineage.models import LineageBase


class ColumnMetadata(LineageBase):
    """SCD2 column classification metadata.

    Each row represents a period during which a column had a specific set
    of classification tags. When tags change, the current record is closed
    (``valid_to`` set) and a new record is inserted.

    Attributes:
        schema_name: Database schema containing the table.
        table_name: Table name (unqualified).
        column_name: Column name.
        tags: JSONB classification tags, e.g. ``{"pii": true, "phi": false}``.
        valid_from: When this classification became effective.
        valid_to: When this classification was superseded (NULL = current).
        updated_by: Run ID or user that wrote this record.
        contract_name: Source data contract identifier.
    """

    __tablename__ = "column_metadata"

    schema_name: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    table_name: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    column_name: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    tags: Mapped[dict[str, object]] = mapped_column(
        JSONB(astext_type=Text()),
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    valid_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        primary_key=True,
        server_default=text("NOW()"),
    )

    valid_to: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    updated_by: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    contract_name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<ColumnMetadata({self.schema_name}.{self.table_name}.{self.column_name}, "
            f"tags={self.tags!r}, current={self.valid_to is None})>"
        )


# GIN index for JSONB containment queries (e.g. WHERE tags @> '{"pii": true}')
Index(
    "ix_column_metadata_tags",
    ColumnMetadata.tags,
    postgresql_using="gin",
)

# Partial index for efficient current-state lookups
Index(
    "ix_column_metadata_current",
    ColumnMetadata.schema_name,
    ColumnMetadata.table_name,
    ColumnMetadata.column_name,
    postgresql_where=ColumnMetadata.valid_to.is_(None),
)
