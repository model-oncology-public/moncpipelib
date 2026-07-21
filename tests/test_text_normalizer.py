"""Tests for TextNormalizer."""

from __future__ import annotations

import polars as pl
import pytest

from moncpipelib.transforms.normalization import (
    COMPANY_SUFFIXES,
    DEFAULT_PREFIXES,
    DEFAULT_SYMBOL_MAPPINGS,
    TextNormalizer,
)

# ---------------------------------------------------------------------------
# Defaults / basics
# ---------------------------------------------------------------------------


class TestTextNormalizerDefaults:
    """Default normalizer (case + punctuation + whitespace + unicode)."""

    def test_basic_normalization(self) -> None:
        n = TextNormalizer()
        assert n.normalize("  Hello, World!  ") == "hello world"

    def test_none_returns_none(self) -> None:
        assert TextNormalizer().normalize(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert TextNormalizer().normalize("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert TextNormalizer().normalize("   ") is None


# ---------------------------------------------------------------------------
# Individual options
# ---------------------------------------------------------------------------


class TestIgnoreCase:
    def test_case_folding_enabled(self) -> None:
        n = TextNormalizer(strip_punctuation=False)
        assert n.normalize("HELLO World") == "hello world"

    def test_case_folding_disabled(self) -> None:
        n = TextNormalizer(ignore_case=False, strip_punctuation=False)
        assert n.normalize("HELLO World") == "HELLO World"


class TestStripPunctuation:
    def test_punctuation_removed(self) -> None:
        n = TextNormalizer()
        assert n.normalize("Hello, World!") == "hello world"

    def test_punctuation_preserved(self) -> None:
        n = TextNormalizer(strip_punctuation=False)
        assert n.normalize("Hello, World!") == "hello, world!"


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self) -> None:
        n = TextNormalizer(strip_punctuation=False)
        assert n.normalize("hello   world") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        n = TextNormalizer(strip_punctuation=False)
        assert n.normalize("  hello  ") == "hello"

    def test_whitespace_normalization_disabled(self) -> None:
        n = TextNormalizer(normalize_whitespace=False, strip_punctuation=False, ignore_case=False)
        assert n.normalize("  hello   world  ") == "  hello   world  "


class TestUnicodeNormalize:
    def test_accented_characters_to_ascii(self) -> None:
        n = TextNormalizer()
        assert n.normalize("cafe\u0301") == "cafe"  # e with combining acute -> e
        assert n.normalize("\u00e9lan") == "elan"  # e-acute precomposed

    def test_unicode_normalization_disabled(self) -> None:
        n = TextNormalizer(unicode_normalize=False, strip_punctuation=False)
        result = n.normalize("\u00e9lan")
        assert result is not None
        assert "\u00e9" in result


# ---------------------------------------------------------------------------
# Suffix stripping
# ---------------------------------------------------------------------------


class TestStripSuffixes:
    def test_default_company_suffixes(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Trifluent Pharma, LLC") == "trifluent pharma"

    def test_various_suffixes(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Acme Corp") == "acme"
        assert n.normalize("Acme Corp.") == "acme"
        assert n.normalize("Acme Corporation") == "acme"
        assert n.normalize("Acme Inc") == "acme"
        assert n.normalize("Acme Inc.") == "acme"
        assert n.normalize("Acme Ltd") == "acme"
        assert n.normalize("Acme Limited") == "acme"
        assert n.normalize("Acme GmbH") == "acme"

    def test_custom_suffix_list(self) -> None:
        n = TextNormalizer(strip_suffixes=["Pharma", "Labs"])
        assert n.normalize("Trifluent Pharma") == "trifluent"
        assert n.normalize("Acme Labs") == "acme"

    def test_suffix_word_boundary(self) -> None:
        """'Shellc' must NOT have 'llc' stripped -- no word boundary."""
        n = TextNormalizer(strip_suffixes=True)
        result = n.normalize("Shellc")
        assert result == "shellc"

    def test_suffix_with_periods(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Beyond Health P.A.") == "beyond health"
        assert n.normalize("Beyond Health P.A") == "beyond health"

    def test_suffix_with_comma(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Trifluent Pharma, Inc.") == "trifluent pharma"

    def test_suffix_disabled_by_default(self) -> None:
        n = TextNormalizer()
        # "LLC" should remain (as "llc" after casefolding + punctuation strip)
        result = n.normalize("Acme LLC")
        assert result is not None
        assert "llc" in result

    def test_multi_word_suffix(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Foo Pty Ltd") == "foo"

    def test_pa_suffix_variants(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        assert n.normalize("Beyond Health Pa") == "beyond health"
        assert n.normalize("Beyond Health PA") == "beyond health"
        assert n.normalize("Beyond Health P.A.") == "beyond health"
        assert n.normalize("Beyond Health, P.A.") == "beyond health"


# ---------------------------------------------------------------------------
# Prefix stripping
# ---------------------------------------------------------------------------


class TestStripPrefixes:
    def test_default_prefix_the(self) -> None:
        n = TextNormalizer(strip_prefixes=True)
        assert n.normalize("The Company") == "company"

    def test_prefix_word_boundary(self) -> None:
        """'Therapy' must NOT have 'The' stripped."""
        n = TextNormalizer(strip_prefixes=True)
        assert n.normalize("Therapy Corp") == "therapy corp"

    def test_custom_prefix_list(self) -> None:
        n = TextNormalizer(strip_prefixes=["Dr", "Mr"])
        assert n.normalize("Dr Smith") == "smith"
        assert n.normalize("Mr Jones") == "jones"

    def test_prefix_disabled_by_default(self) -> None:
        n = TextNormalizer()
        result = n.normalize("The Company")
        assert result is not None
        assert "the" in result


# ---------------------------------------------------------------------------
# Symbol mappings
# ---------------------------------------------------------------------------


class TestSymbolMappings:
    def test_default_ampersand_mapping(self) -> None:
        n = TextNormalizer(symbol_mappings=True, strip_punctuation=False)
        assert n.normalize("A & B") == "a and b"

    def test_custom_symbol_mappings(self) -> None:
        n = TextNormalizer(symbol_mappings={" + ": " and "}, strip_punctuation=False)
        assert n.normalize("A + B") == "a and b"
        # Default & mapping should NOT be active.
        assert n.normalize("A & B") == "a & b"

    def test_symbol_mappings_disabled_by_default(self) -> None:
        n = TextNormalizer(strip_punctuation=False)
        assert n.normalize("A & B") == "a & b"

    def test_symbol_mapping_space_aware(self) -> None:
        """'AT&T' should NOT have & replaced (no surrounding spaces)."""
        n = TextNormalizer(symbol_mappings=True, strip_punctuation=False)
        assert n.normalize("AT&T") == "at&t"


# ---------------------------------------------------------------------------
# Polars expression
# ---------------------------------------------------------------------------


class TestAsExpr:
    def test_polars_expression_basic(self) -> None:
        n = TextNormalizer()
        df = pl.DataFrame({"name": ["  Hello, World!  ", "FOO BAR"]})
        result = df.with_columns(n.as_expr("name"))
        assert result["name"].to_list() == ["hello world", "foo bar"]

    def test_polars_expression_with_nulls(self) -> None:
        n = TextNormalizer()
        df = pl.DataFrame({"name": ["Hello", None, ""]})
        result = df.with_columns(n.as_expr("name"))
        assert result["name"].to_list() == ["hello", None, None]

    def test_polars_expression_preserves_column_name(self) -> None:
        n = TextNormalizer()
        df = pl.DataFrame({"manufacturer": ["Test"]})
        result = df.with_columns(n.as_expr("manufacturer"))
        assert "manufacturer" in result.columns


# ---------------------------------------------------------------------------
# Acceptance scenarios from issue
# ---------------------------------------------------------------------------


class TestBeyondHealthScenario:
    """All five 'Beyond Health' variants from the issue must unify."""

    VARIANTS = [
        "Beyond Health P.A",
        "Beyond Health Pa",
        "Beyond Health, P.A.",
        "Beyond Health, Pa",
        "Beyond Health. Pa",
    ]

    def test_all_variants_normalize_same(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        results = {n.normalize(v) for v in self.VARIANTS}
        assert len(results) == 1, f"Expected 1 unique result, got {results}"
        assert results.pop() == "beyond health"


class TestTrifluentPharmaScenario:
    """LLC vs Inc must unify after suffix stripping."""

    def test_suffix_stripping_unifies(self) -> None:
        n = TextNormalizer(strip_suffixes=True)
        a = n.normalize("Trifluent Pharma, LLC")
        b = n.normalize("Trifluent Pharma, Inc.")
        assert a == b == "trifluent pharma"


# ---------------------------------------------------------------------------
# Combined / misc
# ---------------------------------------------------------------------------


class TestCombinedOptions:
    def test_all_options_enabled(self) -> None:
        n = TextNormalizer(
            strip_suffixes=True,
            strip_prefixes=True,
            symbol_mappings=True,
        )
        result = n.normalize("The Procter & Gamble Co.")
        assert result == "procter and gamble"

    def test_minimal_options(self) -> None:
        n = TextNormalizer(
            ignore_case=False,
            strip_punctuation=False,
            normalize_whitespace=False,
            unicode_normalize=False,
        )
        assert n.normalize("Hello, World!") == "Hello, World!"

    def test_frozen_dataclass(self) -> None:
        n = TextNormalizer()
        with pytest.raises(AttributeError):
            n.ignore_case = False  # type: ignore[misc]


class TestConstants:
    """Sanity checks on built-in constant lists."""

    def test_company_suffixes_not_empty(self) -> None:
        assert len(COMPANY_SUFFIXES) > 0

    def test_default_prefixes_not_empty(self) -> None:
        assert len(DEFAULT_PREFIXES) > 0

    def test_default_symbol_mappings_has_ampersand(self) -> None:
        assert " & " in DEFAULT_SYMBOL_MAPPINGS
