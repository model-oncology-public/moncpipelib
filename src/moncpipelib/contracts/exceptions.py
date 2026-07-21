"""Contract-related exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moncpipelib.contracts.models import ValidationResult


class ContractValidationError(Exception):
    """Raised when contract YAML structure is invalid.

    This indicates the contract file itself has syntax errors or
    missing required fields, not that data failed validation.
    """

    pass


class ContractNotFoundError(Exception):
    """Raised when a contract file doesn't exist."""

    pass


class ContractViolationError(Exception):
    """Raised when data fails contract validation at write time.

    This indicates the actual data being written doesn't match
    the contract's expectations.

    Attributes:
        asset_name: Name of the asset being validated
        violations: List of validation failures
    """

    def __init__(
        self,
        message: str,
        asset_name: str | None = None,
        violations: list[ValidationResult] | None = None,
    ) -> None:
        super().__init__(message)
        self.asset_name = asset_name
        self.violations = violations or []
