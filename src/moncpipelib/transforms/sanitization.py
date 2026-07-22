"""Data sanitization and type conversion utilities for Polars."""

import re

import polars as pl


def normalize_ndc(
    ndc: object,
    *,
    force_package: bool = False,
    package_suffix: str = "00",
    with_hyphens: bool = True,
) -> str | None:
    """Normalize an NDC code, preserving segment structure from dashes.

    When dashes are present, uses them to identify segment boundaries
    and pads each segment independently:

    - Labeler (first segment): zero-pad to 5 digits
    - Product (second segment): zero-pad to 4 digits
    - Package (third segment, if present): zero-pad to 2 digits

    Two-segment (product-only) NDCs are returned as ``5-4`` by default.
    Set ``force_package=True`` to append a synthetic package segment,
    but note that this fabricates data -- ``-00`` is a valid package
    code and may collide with real NDCs.

    When no dashes are present (pure digits), applies the legacy
    11-digit padding heuristic for backwards compatibility with
    integer or pre-stripped inputs.

    This is a scalar function designed for use with Polars ``map_elements``.

    Args:
        ndc: NDC code as any type (string, int, float, None).
        force_package: When True, append ``-{package_suffix}`` to
            two-segment (product-only) NDCs.  Defaults to False.
        package_suffix: Package segment to append when ``force_package``
            is True.  Defaults to ``"00"``.
        with_hyphens: When True (default), return the hyphenated form
            (e.g., ``"00536-1327-01"``).  When False, return the
            non-hyphenated form (e.g., ``"00536132701"``).  The result
            is still a string to preserve leading zeros.

    Returns:
        Formatted NDC string or None if the input is None or contains
        no digits.

    Example:
        ```python
        df = df.with_columns([
            pl.col("ndc")
            .map_elements(normalize_ndc, return_dtype=pl.String)
            .alias("ndc_normalized"),
        ])
        ```
    """
    if ndc is None:
        return None
    # Convert float to int first to avoid trailing .0 adding spurious digits
    if isinstance(ndc, float) and ndc == int(ndc):
        ndc = int(ndc)
    raw = str(ndc).strip()
    if not raw:
        return None

    result: str | None = None

    # Segment-aware path: dashes indicate segment boundaries
    if "-" in raw:
        parts = [p for p in raw.split("-") if p]
        if len(parts) == 2:
            # Product NDC (labeler-product), no package segment
            base = f"{parts[0].zfill(5)}-{parts[1].zfill(4)}"
            result = f"{base}-{package_suffix.zfill(2)}" if force_package else base
        elif len(parts) >= 3:
            # Full NDC (labeler-product-package)
            result = f"{parts[0].zfill(5)}-{parts[1].zfill(4)}-{parts[2].zfill(2)}"

    # No dashes: strip non-digits, pad, segment as 5-4-2
    if result is None:
        digits = re.sub(r"[^0-9]", "", raw)
        if not digits:
            return None
        if len(digits) == 10:
            digits = "0" + digits
        elif len(digits) < 10:
            digits = digits.zfill(11)
        elif len(digits) > 11:
            digits = digits[:11]
        result = f"{digits[:5]}-{digits[5:9]}-{digits[9:11]}"

    if not with_hyphens:
        return result.replace("-", "")
    return result


def safe_decimal(col_name: str) -> pl.Expr:
    """Create a Polars expression that safely parses a column to Float64.

    Handles null values, empty strings, and whitespace by converting them to null.
    Strips whitespace before parsing. Accepts any input type (including Null dtype
    from empty result sets) by casting to String internally before processing.

    Args:
        col_name: Name of the column to parse.

    Returns:
        Polars expression that evaluates to Float64 or null.

    Example:
        ```python
        df = df.select([
            safe_decimal("price"),
            safe_decimal("quantity"),
        ])
        ```
    """
    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    # Clean the string, replacing empty/whitespace with null
    cleaned = as_str.str.strip_chars().replace("", None)
    # Then cast to float
    return cleaned.cast(pl.Float64).alias(col_name)


def safe_bool(col_name: str) -> pl.Expr:
    """Create a Polars expression that safely parses a column to Boolean.

    Handles various boolean representations:
    - True: 't', 'true', '1', 'yes', 'y' (case-insensitive)
    - False: 'f', 'false', '0', 'no', 'n' (case-insensitive)
    - Null: null, empty string, whitespace-only, unrecognized values

    Accepts any input type (including Null dtype from empty result sets) by
    casting to String internally before processing.

    Args:
        col_name: Name of the column to parse.

    Returns:
        Polars expression that evaluates to Boolean or null.

    Example:
        ```python
        df = df.select([
            safe_bool("is_active"),
            safe_bool("has_discount"),
        ])
        ```
    """
    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    # Clean and lowercase the string
    cleaned = as_str.str.strip_chars().str.to_lowercase().replace("", None)
    # Map to boolean values
    return (
        pl.when(cleaned.is_null())
        .then(None)
        .when(cleaned.is_in(["t", "true", "1", "yes", "y"]))
        .then(True)
        .when(cleaned.is_in(["f", "false", "0", "no", "n"]))
        .then(False)
        .otherwise(None)
        .alias(col_name)
    )


def clean_text(col_name: str) -> pl.Expr:
    """Create a Polars expression that cleans and normalizes text.

    Strips leading/trailing whitespace and converts empty strings to null.
    Accepts any input type (including Null dtype from empty result sets) by
    casting to String internally before processing.

    Args:
        col_name: Name of the column to clean.

    Returns:
        Polars expression that evaluates to cleaned String or null.

    Example:
        ```python
        df = df.select([
            clean_text("name"),
            clean_text("description"),
        ])
        ```
    """
    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    return as_str.str.strip_chars().replace("", None).alias(col_name)


def safe_int(col_name: str) -> pl.Expr:
    """Create a Polars expression that safely parses a column to Int64.

    Handles null values, empty strings, and whitespace by converting them to null.
    Strips whitespace before parsing. Accepts any input type (including Null dtype
    from empty result sets) by casting to String internally before processing.

    Args:
        col_name: Name of the column to parse.

    Returns:
        Polars expression that evaluates to Int64 or null.

    Example:
        ```python
        df = df.select([
            safe_int("count"),
            safe_int("quantity"),
        ])
        ```
    """
    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    cleaned = as_str.str.strip_chars().replace("", None)
    return cleaned.cast(pl.Int64).alias(col_name)


_UNAMBIGUOUS_DATE_FORMATS: list[str] = [
    "%Y-%m-%d",  # 2024-01-15 (ISO 8601)
    "%Y%m%d",  # 20240115
    "%d-%b-%y",  # 15-Jan-24  (abbreviated month, 2-digit year)
    "%d-%b-%Y",  # 15-Jan-2024 (abbreviated month, 4-digit year)
    "%b %d, %Y",  # Jan 15, 2024
    "%Y/%m/%d",  # 2024/01/15
]
"""Date formats that are unambiguous (no MM/DD vs DD/MM confusion).

Used as fallback formats when ``safe_date`` is called without an explicit
format.  Ordered by likelihood -- common formats first for early exit.

Note: Full month-name formats (``%B``, e.g. "January") are excluded because
Polars <= 1.38 panics instead of returning null when ``strict=False`` and the
input contains an abbreviated month name.  Pass ``formats=["%d-%B-%Y"]``
explicitly if you need full month-name parsing.
"""


def safe_date(
    col_name: str,
    *,
    format: str | None = None,
    formats: list[str] | None = None,
) -> pl.Expr:
    """Create a Polars expression that safely parses a column to Date.

    Handles null values, empty strings, and whitespace by converting them to null.
    Strips whitespace before parsing. Accepts any input type (including Null dtype
    from empty result sets) by casting to String internally before processing.

    When neither ``format`` nor ``formats`` is provided, tries all built-in
    unambiguous formats (ISO 8601, YYYYMMDD, DD-Mon-YY, etc.) using
    ``pl.coalesce`` with ``strict=False``.  Only unambiguous formats are
    included -- patterns like MM/DD/YYYY vs DD/MM/YYYY require an explicit
    ``format`` argument.

    Args:
        col_name: Name of the column to parse.
        format: Single strptime format string.  When provided, only this
            format is attempted (fastest path).
        formats: List of strptime format strings to try in order.  The first
            successful parse wins.  Cannot be combined with ``format``.

    Returns:
        Polars expression that evaluates to Date or null.

    Example:
        ```python
        df = df.select([
            # Auto-detect from unambiguous formats (handles mixed YYYYMMDD and DD-Mon-YY)
            safe_date("marketing_start_date"),
            # Explicit single format (fastest)
            safe_date("birth_date", format="%Y-%m-%d"),
            # Explicit list of formats to try
            safe_date("event_date", formats=["%Y%m%d", "%d-%b-%y"]),
        ])
        ```
    """
    if format is not None and formats is not None:
        raise ValueError("Cannot specify both 'format' and 'formats'")

    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    cleaned = as_str.str.strip_chars().replace("", None)

    # Single explicit format -- fast path
    if format is not None:
        return cleaned.str.to_date(format).alias(col_name)

    # Multiple formats -- coalesce with strict=False
    fmt_list = formats if formats is not None else _UNAMBIGUOUS_DATE_FORMATS
    return pl.coalesce([cleaned.str.to_date(fmt, strict=False) for fmt in fmt_list]).alias(col_name)


def safe_datetime(col_name: str, format: str = "%Y-%m-%dT%H:%M:%S") -> pl.Expr:
    """Create a Polars expression that safely parses a column to Datetime.

    Handles null values, empty strings, and whitespace by converting them to null.
    Strips whitespace before parsing. Accepts any input type (including Null dtype
    from empty result sets) by casting to String internally before processing.

    Args:
        col_name: Name of the column to parse.
        format: strptime format string. Defaults to ISO format ('%Y-%m-%dT%H:%M:%S').

    Returns:
        Polars expression that evaluates to Datetime or null.

    Example:
        ```python
        df = df.select([
            safe_datetime("created_at"),
            safe_datetime("updated_at", format="%Y-%m-%d %H:%M:%S"),
        ])
        ```
    """
    # Cast to string first (handles any input type including Null dtype)
    as_str = pl.col(col_name).cast(pl.String)
    cleaned = as_str.str.strip_chars().replace("", None)
    return cleaned.str.to_datetime(format).alias(col_name)
