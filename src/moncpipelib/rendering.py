"""DataFrame rendering utilities with PII-aware masking.

Provides polars_to_md() for converting Polars DataFrames to markdown tables
with automatic PII masking based on data contract metadata. This is designed
for safe logging in HIPAA/SOC2 environments -- columns marked as PII (or
unannotated, since pii defaults to True) are replaced with a mask value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from moncpipelib.contracts.models import DataContract


def polars_to_md(
    df: pl.DataFrame,
    *,
    contract: DataContract | None = None,
    pii_columns: list[str] | None = None,
    mask_value: str = "***",
    max_rows: int | None = 10,
) -> str:
    """Render a Polars DataFrame as a markdown table with PII masking.

    When a contract is provided, columns with pii=True (the default for
    unannotated columns) are replaced with ``mask_value``. Additional
    columns can be masked via the ``pii_columns`` parameter. The two
    sources are unioned.

    When neither ``contract`` nor ``pii_columns`` is provided, no masking
    is applied -- the caller is responsible for knowing their data.

    Args:
        df: The DataFrame to render.
        contract: Optional data contract for PII column lookup.
        pii_columns: Optional explicit list of column names to mask.
        mask_value: The string to substitute for PII values.
        max_rows: Maximum rows to include. None for unlimited.

    Returns:
        A markdown-formatted table string.
    """
    if df.is_empty():
        return _render_header(df.columns) + "\n" + _render_separator(df.columns)

    # Build set of columns to mask
    pii_set: set[str] = set()
    if contract is not None:
        pii_set.update(contract.get_pii_column_names())
    if pii_columns is not None:
        pii_set.update(pii_columns)

    # Only mask columns that actually exist in the DataFrame
    pii_set &= set(df.columns)

    # Truncate
    truncated = False
    if max_rows is not None and len(df) > max_rows:
        display_df = df.head(max_rows)
        truncated = True
    else:
        display_df = df

    # Apply masking
    if pii_set:
        display_df = display_df.with_columns(
            [pl.lit(mask_value).alias(col) for col in df.columns if col in pii_set]
        )

    # Render
    lines: list[str] = []
    lines.append(_render_header(display_df.columns))
    lines.append(_render_separator(display_df.columns))

    for row in display_df.iter_rows():
        cells = [_format_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    if truncated:
        lines.append("")
        lines.append(f"*Showing {max_rows} of {len(df)} rows.*")

    return "\n".join(lines)


def _render_header(columns: list[str]) -> str:
    """Render the markdown table header row."""
    return "| " + " | ".join(columns) + " |"


def _render_separator(columns: list[str]) -> str:
    """Render the markdown table separator row."""
    return "| " + " | ".join("---" for _ in columns) + " |"


def _format_cell(value: object) -> str:
    """Format a single cell value for markdown output."""
    if value is None:
        return ""
    return str(value)
