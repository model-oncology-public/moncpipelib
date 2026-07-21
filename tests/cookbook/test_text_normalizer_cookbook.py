"""Cookbook tests for TextNormalizer.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations

import pytest


@pytest.mark.cookbook(
    title="Basic Text Normalization",
    description=(
        "Use ``TextNormalizer`` to collapse messy text variants into a "
        "consistent normalized form. By default it folds case, strips "
        "punctuation, normalizes whitespace, and converts unicode to ASCII."
    ),
    category="transforms",
)
def test_basic_text_normalization() -> None:
    """Demonstrate basic scalar normalization."""
    # --- cookbook:start ---
    from moncpipelib.transforms import TextNormalizer

    normalizer = TextNormalizer()

    raw_values = [
        "  ACME  Pharmaceuticals ",
        "acme pharmaceuticals",
        "Acme, Pharmaceuticals!",
    ]

    for raw in raw_values:
        print(f"{raw!r:>30s}  ->  {normalizer.normalize(raw)!r}")
    # --- cookbook:end ---

    results = {normalizer.normalize(v) for v in raw_values}
    assert len(results) == 1


@pytest.mark.cookbook(
    title="Normalizing a DataFrame Column",
    description=(
        "Use ``as_expr()`` to normalize an entire Polars column in a single "
        "expression. This wraps the scalar ``normalize()`` method via "
        "``map_elements`` so all configuration options are available."
    ),
    category="transforms",
)
def test_normalize_dataframe_column() -> None:
    """Demonstrate column-level normalization with Polars."""
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.transforms import TextNormalizer

    normalizer = TextNormalizer(strip_suffixes=True)

    df = pl.DataFrame(
        {
            "manufacturer": [
                "Beyond Health P.A",
                "Beyond Health Pa",
                "Beyond Health, P.A.",
                "Trifluent Pharma, LLC",
                "Trifluent Pharma, Inc.",
            ]
        }
    )

    result = df.with_columns(
        normalizer.as_expr("manufacturer").alias("normalized"),
    )

    print(result)
    # --- cookbook:end ---

    normalized = result["normalized"].to_list()
    # All Beyond Health variants should be the same.
    assert normalized[0] == normalized[1] == normalized[2] == "beyond health"
    # Both Trifluent variants should be the same.
    assert normalized[3] == normalized[4] == "trifluent pharma"


@pytest.mark.cookbook(
    title="Custom Configuration for Entity Resolution",
    description=(
        "Enable suffix stripping, prefix stripping, and symbol mappings "
        "to maximize entity unification. Built-in presets cover common "
        "company suffixes (LLC, Inc, Corp, etc.), noise prefixes (The), "
        "and symbol mappings (& -> and)."
    ),
    category="transforms",
)
def test_custom_entity_resolution_config() -> None:
    """Demonstrate full MDM-oriented configuration."""
    # --- cookbook:start ---
    from moncpipelib.transforms import TextNormalizer

    normalizer = TextNormalizer(
        strip_suffixes=True,  # built-in: LLC, Inc, Corp, PA, GmbH, ...
        strip_prefixes=True,  # built-in: The
        symbol_mappings=True,  # built-in: " & " -> " and "
    )

    examples = [
        ("Beyond Health P.A", "Beyond Health, P.A."),
        ("Trifluent Pharma, LLC", "Trifluent Pharma, Inc."),
        ("The Procter & Gamble Co", "Procter and Gamble Company"),
    ]

    for a, b in examples:
        na, nb = normalizer.normalize(a), normalizer.normalize(b)
        match = "MATCH" if na == nb else "DIFFER"
        print(f"{a!r:>35s}  ->  {na!r}")
        print(f"{b!r:>35s}  ->  {nb!r}")
        print(f"{'':>35s}      [{match}]\n")
    # --- cookbook:end ---

    for a, b in examples:
        assert normalizer.normalize(a) == normalizer.normalize(b)
