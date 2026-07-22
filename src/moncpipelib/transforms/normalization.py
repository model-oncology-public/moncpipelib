"""Configurable text normalization for entity standardization.

Provides :class:`TextNormalizer`, a frozen dataclass that collapses syntactic
variants of the same entity (case, punctuation, legal suffixes, unicode) into a
single normalized form.  Designed for MDM seed workflows where dirty source
values must be grouped before canonical mapping.

Security note
-------------
If used on columns containing PHI (e.g., patient names), the caller is
responsible for ensuring the normalized output receives the same PHI
protections as the original input.  The normalizer itself does not persist,
transmit, or log any data.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import polars as pl

# ---------------------------------------------------------------------------
# Built-in constants
# ---------------------------------------------------------------------------

COMPANY_SUFFIXES: list[str] = [
    # US corporate
    "LLC",
    "PLLC",
    "Inc",
    "Inc.",
    "Incorporated",
    "Corp",
    "Corp.",
    "Corporation",
    "Co",
    "Co.",
    "Company",
    "Ltd",
    "Ltd.",
    "Limited",
    "LP",
    "LLP",
    "LLLP",
    # Professional associations
    "PA",
    "P.A.",
    "P.A",
    "Pa",
    "Pa.",
    "PC",
    "P.C.",
    "P.C",
    # DBA
    "DBA",
    "D.B.A.",
    "D.B.A",
    # International
    "PLC",
    "GmbH",
    "AG",
    "S.A.",
    "SA",
    "S.A",
    "NV",
    "N.V.",
    "N.V",
    "Pty",
    "Pty Ltd",
    "Pty. Ltd.",
    "BV",
    "B.V.",
    "B.V",
]
"""Built-in list of common company / legal suffixes for :class:`TextNormalizer`.

Multi-word entries (e.g. ``"Pty Ltd"``) are matched greedily by sorting
longest-first at compile time.
"""

DEFAULT_PREFIXES: list[str] = ["The"]
"""Built-in list of noise prefixes stripped by :class:`TextNormalizer`."""

DEFAULT_SYMBOL_MAPPINGS: dict[str, str] = {" & ": " and "}
"""Default space-delimited symbol replacements for :class:`TextNormalizer`."""


# ---------------------------------------------------------------------------
# TextNormalizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextNormalizer:
    """Configurable text normalizer for collapsing entity name variants.

    Each normalization step can be independently enabled or disabled.  The
    pipeline executes in a fixed order designed so that earlier steps do not
    destroy information needed by later steps (see *Normalization Pipeline*
    below).

    Args:
        ignore_case: Fold to lowercase via :meth:`str.casefold`.
        strip_punctuation: Remove all non-alphanumeric, non-whitespace chars.
        normalize_whitespace: Collapse whitespace runs and strip edges.
        unicode_normalize: NFKD-normalize then drop non-ASCII characters.
        strip_suffixes: ``True`` uses :data:`COMPANY_SUFFIXES`, a list
            provides custom suffixes, ``False`` disables.
        strip_prefixes: ``True`` uses :data:`DEFAULT_PREFIXES`, a list
            provides custom prefixes, ``False`` disables.
        symbol_mappings: ``True`` uses :data:`DEFAULT_SYMBOL_MAPPINGS`, a dict
            provides custom mappings, ``False`` disables.  Replacements are
            applied literally (space-delimited by convention so ``"AT&T"``
            is not affected by the default ``" & "`` -> ``" and "`` rule).

    Normalization Pipeline (in order):
        1. Unicode normalization
        2. Symbol mappings
        3. Case folding
        4. Prefix stripping (word-boundary aware)
        5. Suffix stripping (word-boundary aware)
        6. Punctuation removal
        7. Whitespace normalization

    Example:
        ```python
        from moncpipelib.transforms import TextNormalizer

        n = TextNormalizer(strip_suffixes=True, symbol_mappings=True)
        n.normalize("Beyond Health, P.A.")   # -> "beyond health"
        n.normalize("Beyond Health Pa")      # -> "beyond health"
        n.normalize("Trifluent Pharma, LLC") # -> "trifluent pharma"
        n.normalize("Trifluent Pharma, Inc.") # -> "trifluent pharma"
        ```
    """

    ignore_case: bool = True
    strip_punctuation: bool = True
    normalize_whitespace: bool = True
    unicode_normalize: bool = True
    strip_suffixes: bool | list[str] = False
    strip_prefixes: bool | list[str] = False
    symbol_mappings: dict[str, str] | bool = False

    # -- Public API ---------------------------------------------------------

    def normalize(self, value: str | None) -> str | None:
        """Normalize a single string value.

        Args:
            value: The raw text to normalize.  ``None`` and empty strings
                return ``None``.

        Returns:
            The normalized string, or ``None`` if the input is ``None`` or
            the result is empty after all transformations.
        """
        if value is None:
            return None

        result = str(value)
        if not result.strip():
            return None

        # 1. Unicode normalization (before everything so accented variants
        #    collapse before matching / folding).
        if self.unicode_normalize:
            result = self._apply_unicode_normalize(result)

        # 2. Symbol mappings (before punctuation strip would eat symbols like &).
        mappings = self._get_symbol_mappings()
        if mappings:
            for old, new in mappings.items():
                result = result.replace(old, new)

        # 3. Case folding (before suffix / prefix matching).
        if self.ignore_case:
            result = result.casefold()

        # 4. Strip prefixes (word-boundary aware, before punctuation strip).
        prefixes = self._get_prefix_list()
        if prefixes:
            result = self._strip_prefix(result, prefixes)

        # 5. Strip suffixes (word-boundary aware, before punctuation strip so
        #    "P.A." can still be matched).
        suffixes = self._get_suffix_list()
        if suffixes:
            result = self._strip_suffix(result, suffixes)

        # 6. Strip punctuation.
        if self.strip_punctuation:
            result = re.sub(r"[^\w\s]", "", result)

        # 7. Normalize whitespace (always last to clean up gaps).
        if self.normalize_whitespace:
            result = re.sub(r"\s+", " ", result).strip()

        # Final empty check.
        if not result.strip():
            return None

        return result

    def as_expr(self, col_name: str) -> pl.Expr:
        """Return a Polars expression that normalizes a string column.

        Wraps :meth:`normalize` via ``map_elements``.  This is row-by-row and
        slower than native Polars expressions, but the regex-based
        word-boundary suffix / prefix stripping cannot be expressed in native
        Polars string operations alone.

        Args:
            col_name: Name of the column to normalize.

        Returns:
            Polars expression that evaluates to the normalized String column.

        Example:
            ```python
            df = df.with_columns(
                normalizer.as_expr("manufacturer_name").alias("normalized_name"),
            )
            ```
        """
        return pl.col(col_name).map_elements(self.normalize, return_dtype=pl.String).alias(col_name)

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _apply_unicode_normalize(value: str) -> str:
        """NFKD-normalize and drop non-ASCII characters."""
        nfkd = unicodedata.normalize("NFKD", value)
        return nfkd.encode("ascii", "ignore").decode("ascii")

    def _get_suffix_list(self) -> list[str]:
        if isinstance(self.strip_suffixes, list):
            return self.strip_suffixes
        if self.strip_suffixes is True:
            return COMPANY_SUFFIXES
        return []

    def _get_prefix_list(self) -> list[str]:
        if isinstance(self.strip_prefixes, list):
            return self.strip_prefixes
        if self.strip_prefixes is True:
            return DEFAULT_PREFIXES
        return []

    def _get_symbol_mappings(self) -> dict[str, str]:
        if isinstance(self.symbol_mappings, dict):
            return self.symbol_mappings
        if self.symbol_mappings is True:
            return DEFAULT_SYMBOL_MAPPINGS
        return {}

    @staticmethod
    def _strip_suffix(value: str, suffixes: list[str]) -> str:
        """Remove a trailing suffix if preceded by whitespace or comma+whitespace.

        Space-aware: ``"Shellc"`` will *not* match ``"llc"`` because no word
        boundary exists.  Multi-word suffixes (e.g. ``"Pty Ltd"``) are tried
        longest-first.
        """
        # Build alternatives sorted longest-first for greedy matching.
        escaped = sorted(
            (re.escape(s.casefold()) for s in suffixes),
            key=len,
            reverse=True,
        )
        pattern = r"(?:\s*,)?\s+(?:" + "|".join(escaped) + r")\.?\s*$"
        return re.sub(pattern, "", value, flags=re.IGNORECASE)

    @staticmethod
    def _strip_prefix(value: str, prefixes: list[str]) -> str:
        """Remove a leading prefix if followed by whitespace.

        Space-aware: ``"Therapy"`` will *not* have ``"The"`` stripped because
        the match requires a following space.
        """
        escaped = sorted(
            (re.escape(s.casefold()) for s in prefixes),
            key=len,
            reverse=True,
        )
        pattern = r"^(?:" + "|".join(escaped) + r")\s+"
        return re.sub(pattern, "", value, flags=re.IGNORECASE)
