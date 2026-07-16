"""Cookbook examples for data transformation utilities."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.cookbook(
    title="NDC Normalization: All Input Formats",
    description=(
        "``normalize_ndc()`` normalizes National Drug Codes by preserving "
        "segment boundaries from dashes and padding each segment independently. "
        "Two-segment (product-only) NDCs are returned as 5-4 by default. "
        "Three-segment (package) NDCs are returned as 5-4-2. The pure-digit "
        "fallback path pads to 11 digits and segments as 5-4-2."
    ),
    category="transforms",
)
def test_ndc_normalization_formats() -> None:
    """Demonstrate normalize_ndc() with every common NDC format."""
    # --- cookbook:start ---
    from moncpipelib import normalize_ndc

    # 3-segment inputs (labeler-product-package) normalize to 5-4-2
    print("=== Full NDC (labeler-product-package) -> 5-4-2 ===")
    print(f'  "50242-0918-01"  (5-4-2) -> {normalize_ndc("50242-0918-01")}')
    print(f'  "0536-1327-01"   (4-4-2) -> {normalize_ndc("0536-1327-01")}')
    print(f'  "50242-918-01"   (5-3-2) -> {normalize_ndc("50242-918-01")}')
    print(f'  "0536-327-01"    (4-3-2) -> {normalize_ndc("0536-327-01")}')

    # 2-segment inputs (labeler-product) normalize to 5-4 -- NO package appended
    print()
    print("=== Product NDC (labeler-product) -> 5-4 only ===")
    print(f'  "50242-0918"     (5-4)   -> {normalize_ndc("50242-0918")}')
    print(f'  "50242-918"      (5-3)   -> {normalize_ndc("50242-918")}')
    print(f'  "0536-1327"      (4-4)   -> {normalize_ndc("0536-1327")}')

    # Pure digits (no dashes): legacy heuristic, pads to 11 digits -> 5-4-2
    print()
    print("=== Pure digits (no dashes -- legacy heuristic) -> 5-4-2 ===")
    print(f'  "00536132701"    (11)    -> {normalize_ndc("00536132701")}')
    print(f'  "0536132701"     (10)    -> {normalize_ndc("0536132701")}')

    # Type handling
    print()
    print("=== Type handling ===")
    print(f"  536132701        (int)   -> {normalize_ndc(536132701)}")
    print(f"  536132701.0      (float) -> {normalize_ndc(536132701.0)}")
    print(f"  None                     -> {normalize_ndc(None)}")
    print(f'  ""                       -> {normalize_ndc("")}')
    # --- cookbook:end ---

    # Verify 3-segment formats -> 5-4-2
    assert normalize_ndc("50242-0918-01") == "50242-0918-01"
    assert normalize_ndc("0536-1327-01") == "00536-1327-01"
    assert normalize_ndc("50242-918-01") == "50242-0918-01"
    # Verify 2-segment formats -> 5-4 (no package appended)
    assert normalize_ndc("50242-0918") == "50242-0918"
    assert normalize_ndc("50242-918") == "50242-0918"
    assert normalize_ndc("0536-1327") == "00536-1327"
    # Verify pure digit formats
    assert normalize_ndc("00536132701") == "00536-1327-01"
    assert normalize_ndc("0536132701") == "00536-1327-01"
    # Verify type handling
    assert normalize_ndc(536132701) == "00536-1327-01"
    assert normalize_ndc(536132701.0) == "00536-1327-01"
    assert normalize_ndc(None) is None
    assert normalize_ndc("") is None


@pytest.mark.cookbook(
    title="NDC Normalization: Forced Package Suffix",
    description=(
        "Some downstream systems require all NDCs to include a package segment "
        "(5-4-2 format). ``force_package=True`` appends a synthetic package "
        "code to 2-segment product NDCs. Use this with caution: the default "
        "suffix ``-00`` is itself a valid package code and will collide with "
        "real package NDCs that happen to use ``00``. This should only be used "
        "when you are certain the target system requires a package segment and "
        "the synthetic value will not be confused with real data."
    ),
    category="transforms",
)
def test_ndc_force_package_suffix() -> None:
    """Demonstrate force_package mode and its risks."""
    # --- cookbook:start ---
    from moncpipelib import normalize_ndc

    # Default: 2-segment NDCs stay as 5-4, no data is fabricated
    print("=== Default (no forced package) ===")
    print(f'  "50242-0918" -> {normalize_ndc("50242-0918")}')

    # force_package=True: appends a synthetic package segment
    # WARNING: "-00" is a real package code -- this fabricates data
    print()
    print("=== force_package=True (synthetic suffix) ===")
    print(f'  "50242-0918" -> {normalize_ndc("50242-0918", force_package=True)}')

    # Custom suffix: use a non-standard value if -00 conflicts
    print()
    print("=== Custom package_suffix ===")
    print(
        f'  "50242-0918" (suffix="99") -> '
        f"{normalize_ndc('50242-0918', force_package=True, package_suffix='99')}"
    )

    # 3-segment NDCs are unaffected by force_package
    print()
    print("=== 3-segment NDCs are unaffected ===")
    print(f'  "50242-0918-01" -> {normalize_ndc("50242-0918-01", force_package=True)}')
    # --- cookbook:end ---

    # Default: 5-4 only
    assert normalize_ndc("50242-0918") == "50242-0918"
    # force_package appends suffix
    assert normalize_ndc("50242-0918", force_package=True) == "50242-0918-00"
    assert normalize_ndc("50242-0918", force_package=True, package_suffix="99") == "50242-0918-99"
    # 3-segment unaffected
    assert normalize_ndc("50242-0918-01", force_package=True) == "50242-0918-01"


@pytest.mark.cookbook(
    title="NDC Normalization: Non-Hyphenated Output",
    description=(
        "Some systems store NDCs as plain digit strings without hyphens. "
        "``with_hyphens=False`` strips dashes from the normalized result "
        "while still returning a string to preserve leading zeros. "
        "This can be combined with ``force_package`` when the downstream "
        "system expects a fixed-width 11-digit code."
    ),
    category="transforms",
)
def test_ndc_without_hyphens() -> None:
    """Demonstrate with_hyphens=False for non-hyphenated NDC output."""
    # --- cookbook:start ---
    from moncpipelib import normalize_ndc

    # Default: hyphenated output
    print("=== Default (with_hyphens=True) ===")
    print(f'  "0536-1327-01"   -> {normalize_ndc("0536-1327-01")}')
    print(f'  "50242-918"      -> {normalize_ndc("50242-918")}')

    # Non-hyphenated: still a string to preserve leading zeros
    print()
    print("=== with_hyphens=False ===")
    print(f'  "0536-1327-01"   -> {normalize_ndc("0536-1327-01", with_hyphens=False)}')
    print(f'  "50242-918"      -> {normalize_ndc("50242-918", with_hyphens=False)}')

    # Combined with force_package for fixed-width 11-digit output
    print()
    print("=== with_hyphens=False + force_package=True (11-digit string) ===")
    print(
        f'  "50242-0918"     -> '
        f"{normalize_ndc('50242-0918', with_hyphens=False, force_package=True)}"
    )
    # --- cookbook:end ---

    # Hyphenated (default)
    assert normalize_ndc("0536-1327-01") == "00536-1327-01"
    assert normalize_ndc("50242-918") == "50242-0918"
    # Non-hyphenated
    assert normalize_ndc("0536-1327-01", with_hyphens=False) == "00536132701"
    assert normalize_ndc("50242-918", with_hyphens=False) == "502420918"
    # Combined
    assert normalize_ndc("50242-0918", with_hyphens=False, force_package=True) == "50242091800"


@pytest.mark.cookbook(
    title="Safe Type Casting: Text, Decimal, Int, and Bool",
    description=(
        "``clean_text``, ``safe_decimal``, ``safe_int``, and ``safe_bool`` build "
        "Polars expressions that defensively parse messy source columns. Each "
        "casts the input to String first (so they tolerate any dtype, including "
        "the ``Null`` dtype of an empty result set), strips surrounding "
        "whitespace, and converts empty/whitespace-only strings to null instead "
        "of raising. ``safe_bool`` recognizes common truthy/falsy spellings "
        "(``t``/``true``/``1``/``yes``/``y`` and their negatives, "
        "case-insensitively) and maps anything unrecognized to null."
    ),
    category="transforms",
)
def test_safe_casts() -> None:
    """Demonstrate the whitespace- and null-tolerant casting expressions."""
    # --- cookbook:start ---
    from moncpipelib import clean_text, safe_bool, safe_decimal, safe_int

    # Raw source data: padded whitespace, empty strings, and nulls mixed in
    raw = pl.DataFrame(
        {
            "name": ["  Acme  ", "", "   ", "Globex"],
            "price": ["  12.50 ", "", "0", None],
            "qty": [" 10 ", "", "3", None],
            "active": ["YES", "f", "", "1"],
        }
    )

    cleaned = raw.select(
        clean_text("name"),  # trim + empty-string -> null
        safe_decimal("price"),  # -> Float64 or null
        safe_int("qty"),  # -> Int64 or null
        safe_bool("active"),  # -> Boolean or null
    )

    print("=== Cleaned & cast ===")
    print(cleaned)
    print()
    print("Note: empty/whitespace cells became null, not errors.")
    # --- cookbook:end ---

    assert cleaned["name"].to_list() == ["Acme", None, None, "Globex"]
    assert cleaned["price"].to_list() == [12.5, None, 0.0, None]
    assert cleaned["qty"].to_list() == [10, None, 3, None]
    assert cleaned["active"].to_list() == [True, False, None, True]


@pytest.mark.cookbook(
    title="Safe Date and Datetime Parsing",
    description=(
        "``safe_date`` and ``safe_datetime`` parse string columns into temporal "
        "types, converting blanks to null. Called with no ``format``, "
        "``safe_date`` coalesces over a set of *unambiguous* built-in formats "
        "(ISO 8601, ``YYYYMMDD``, ``DD-Mon-YY``, etc.), so a column mixing those "
        "shapes parses in one pass. Ambiguous patterns like ``MM/DD/YYYY`` vs "
        "``DD/MM/YYYY`` are intentionally excluded -- pass an explicit ``format`` "
        "(fastest) or ``formats`` list for those. ``safe_datetime`` defaults to "
        "ISO ``%Y-%m-%dT%H:%M:%S`` and takes a single ``format`` override."
    ),
    category="transforms",
)
def test_safe_date_parsing() -> None:
    """Demonstrate auto-detection and explicit-format date parsing."""
    # --- cookbook:start ---
    from moncpipelib import safe_date, safe_datetime

    # A column mixing several unambiguous date shapes plus blanks
    df = pl.DataFrame(
        {
            "received": ["2024-01-15", "20240115", "15-Jan-24", "  ", None],
            "event": ["01/15/2024", "02/20/2024", None, None, None],
            "created_at": ["2024-01-15T08:30:00", None, None, None, None],
        }
    )

    parsed = df.select(
        # Auto-detect: handles ISO, YYYYMMDD, and DD-Mon-YY together
        safe_date("received"),
        # Ambiguous format -> must be explicit
        safe_date("event", format="%m/%d/%Y"),
        # Datetime with the default ISO format
        safe_datetime("created_at"),
    )

    print("=== Parsed temporals (blanks -> null) ===")
    print(parsed)
    # --- cookbook:end ---

    import datetime as _dt

    assert parsed["received"].to_list() == [
        _dt.date(2024, 1, 15),
        _dt.date(2024, 1, 15),
        _dt.date(2024, 1, 15),
        None,
        None,
    ]
    assert parsed["event"][0] == _dt.date(2024, 1, 15)
    assert parsed["created_at"][0] == _dt.datetime(2024, 1, 15, 8, 30, 0)


@pytest.mark.cookbook(
    title="Deterministic Row Hashing for Change Detection",
    description=(
        "``compute_row_hash`` builds a Polars expression that produces a "
        "deterministic 64-character SHA-256 digest over a chosen set of columns. "
        "Each column is cast to String, nulls are replaced with a sentinel, and "
        "the values are joined with a separator before hashing -- so the digest "
        "is stable across sessions and platforms, making it the canonical input "
        "to SCD2 change detection. Column order is part of the hash: the same "
        "columns in a different order produce a different digest."
    ),
    category="transforms",
)
def test_compute_row_hash() -> None:
    """Demonstrate deterministic, order-sensitive row hashing."""
    # --- cookbook:start ---
    from moncpipelib import compute_row_hash

    df = pl.DataFrame(
        {
            "ndc": ["50242-0918-01", "0536-1327-01"],
            "name": ["Drug A", None],
            "strength": ["10mg", "20mg"],
        }
    )

    hashed = df.with_columns(compute_row_hash(["ndc", "name", "strength"], alias="row_hash"))
    print("=== Row hashes ===")
    print(hashed.select("ndc", "row_hash"))

    # Determinism: recomputing the same columns yields identical digests
    again = df.with_columns(compute_row_hash(["ndc", "name", "strength"]))
    print()
    print("Stable across calls:", hashed["row_hash"].to_list() == again["row_hash"].to_list())
    # --- cookbook:end ---

    assert hashed["row_hash"].dtype == pl.String
    assert all(len(h) == 64 for h in hashed["row_hash"].to_list())
    assert hashed["row_hash"].to_list() == again["row_hash"].to_list()
    # Column order changes the digest
    reordered = df.with_columns(compute_row_hash(["name", "ndc", "strength"]))
    assert reordered["row_hash"].to_list() != hashed["row_hash"].to_list()
