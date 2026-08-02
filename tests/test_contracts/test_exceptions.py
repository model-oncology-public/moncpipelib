"""Tests for contract exceptions."""

import pytest

from moncpipelib.contracts import ValidationResult
from moncpipelib.contracts.exceptions import (
    ContractNotFoundError,
    ContractValidationError,
    ContractViolationError,
)


class TestContractValidationError:
    """Tests for ContractValidationError exception."""

    def test_basic_error(self):
        """Test basic ContractValidationError."""
        error = ContractValidationError("Invalid contract structure")

        assert str(error) == "Invalid contract structure"

    def test_can_be_raised(self):
        """Test that ContractValidationError can be raised and caught."""
        with pytest.raises(ContractValidationError, match="Missing required field"):
            raise ContractValidationError("Missing required field: version")

    def test_inherits_from_exception(self):
        """Test that ContractValidationError inherits from Exception."""
        error = ContractValidationError("Test")

        assert isinstance(error, Exception)


class TestContractNotFoundError:
    """Tests for ContractNotFoundError exception."""

    def test_basic_error(self):
        """Test basic ContractNotFoundError."""
        error = ContractNotFoundError("Contract file not found: missing.yaml")

        assert str(error) == "Contract file not found: missing.yaml"

    def test_can_be_raised(self):
        """Test that ContractNotFoundError can be raised and caught."""
        with pytest.raises(ContractNotFoundError, match="not found"):
            raise ContractNotFoundError("Contract not found: test.yaml")

    def test_inherits_from_exception(self):
        """Test that ContractNotFoundError inherits from Exception."""
        error = ContractNotFoundError("Test")

        assert isinstance(error, Exception)


class TestContractViolationError:
    """Tests for ContractViolationError exception."""

    def test_basic_error(self):
        """Test basic ContractViolationError without additional attributes."""
        error = ContractViolationError("Data validation failed")

        assert str(error) == "Data validation failed"
        assert error.asset_name is None
        assert error.violations == []

    def test_error_with_asset_name(self):
        """Test ContractViolationError with asset_name."""
        error = ContractViolationError(
            "Validation failed for orders",
            asset_name="orders_bronze",
        )

        assert error.asset_name == "orders_bronze"

    def test_error_with_violations(self):
        """Test ContractViolationError with violation results."""
        violations = [
            ValidationResult(
                passed=False,
                message="Column 'id' has null values",
                failed_count=5,
                total_count=100,
            ),
            ValidationResult(
                passed=False,
                message="Column 'status' has invalid values",
                failed_count=3,
                total_count=100,
            ),
        ]

        error = ContractViolationError(
            "Multiple validation failures",
            asset_name="orders",
            violations=violations,
        )

        assert len(error.violations) == 2
        assert error.violations[0].failed_count == 5
        assert error.violations[1].message == "Column 'status' has invalid values"

    def test_error_can_be_raised(self):
        """Test that ContractViolationError can be raised and caught."""
        violations = [
            ValidationResult(passed=False, message="Test failure"),
        ]

        with pytest.raises(ContractViolationError) as exc_info:
            raise ContractViolationError(
                "Contract validation failed",
                asset_name="test_asset",
                violations=violations,
            )

        error = exc_info.value
        assert error.asset_name == "test_asset"
        assert len(error.violations) == 1

    def test_inherits_from_exception(self):
        """Test that ContractViolationError inherits from Exception."""
        error = ContractViolationError("Test")

        assert isinstance(error, Exception)

    def test_error_with_sample_failures(self):
        """Test ContractViolationError with sample failures in ValidationResult."""
        violations = [
            ValidationResult(
                passed=False,
                message="Unique constraint violated",
                failed_count=2,
                total_count=100,
                sample_failures=[
                    {"id": 1, "name": "duplicate1"},
                    {"id": 1, "name": "duplicate2"},
                ],
            ),
        ]

        error = ContractViolationError(
            "Validation failed",
            violations=violations,
        )

        assert error.violations[0].sample_failures is not None
        assert len(error.violations[0].sample_failures) == 2
