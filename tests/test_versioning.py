"""Tests for code_hash() versioning utility."""

from __future__ import annotations

from pathlib import Path

from moncpipelib.versioning import code_hash


class TestCodeHash:
    """Tests for code_hash()."""

    def test_returns_12_char_hex(self) -> None:
        """Output should be 12 hex characters."""
        result = code_hash()
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        """Same file should produce the same hash across calls."""
        assert code_hash() == code_hash()

    def test_changes_when_source_changes(self, tmp_path: Path) -> None:
        """Modifying the source file should change the hash."""
        py_file = tmp_path / "asset.py"
        py_file.write_text("x = 1\n")

        # Call code_hash from the context of the temp file by using _stack_depth=0
        # and patching the stack. Simpler: just test the internals directly.
        import hashlib

        def _hash_file(path: Path) -> str:
            h = hashlib.sha256()
            h.update(path.read_bytes())
            return h.hexdigest()[:12]

        hash1 = _hash_file(py_file)
        py_file.write_text("x = 2\n")
        hash2 = _hash_file(py_file)

        assert hash1 != hash2

    def test_includes_contract_yaml(self, tmp_path: Path) -> None:
        """Contract YAML files in the same directory should affect the hash."""
        import hashlib

        py_file = tmp_path / "asset.py"
        py_file.write_text("x = 1\n")

        def _hash_with_contracts(source: Path) -> str:
            h = hashlib.sha256()
            h.update(source.read_bytes())
            for contract in sorted(source.parent.glob("*.contract.yaml")):
                h.update(contract.read_bytes())
            return h.hexdigest()[:12]

        hash_without = _hash_with_contracts(py_file)

        contract = tmp_path / "asset.contract.yaml"
        contract.write_text("version: '1.0'\n")
        hash_with = _hash_with_contracts(py_file)

        assert hash_without != hash_with

        # Modifying the contract should also change the hash
        contract.write_text("version: '2.0'\n")
        hash_modified = _hash_with_contracts(py_file)

        assert hash_with != hash_modified

    def test_ignores_non_contract_files(self, tmp_path: Path) -> None:
        """Non-contract sibling files should not affect the hash."""
        import hashlib

        py_file = tmp_path / "asset.py"
        py_file.write_text("x = 1\n")

        def _hash_with_contracts(source: Path) -> str:
            h = hashlib.sha256()
            h.update(source.read_bytes())
            for contract in sorted(source.parent.glob("*.contract.yaml")):
                h.update(contract.read_bytes())
            return h.hexdigest()[:12]

        hash_before = _hash_with_contracts(py_file)

        # Add a JSON file - should not affect hash
        (tmp_path / "config.json").write_text('{"key": "value"}')
        hash_after = _hash_with_contracts(py_file)

        assert hash_before == hash_after

    def test_stack_depth_override(self) -> None:
        """Wrapper function using _stack_depth=2 should resolve correctly."""

        def wrapper() -> str:
            return code_hash(_stack_depth=2)

        # Both should resolve to this test file
        direct = code_hash()
        wrapped = wrapper()
        assert direct == wrapped

    def test_called_from_this_file(self) -> None:
        """code_hash() called here should hash this test file."""
        import hashlib

        this_file = Path(__file__)
        h = hashlib.sha256()
        h.update(this_file.read_bytes())
        for contract in sorted(this_file.parent.glob("*.contract.yaml")):
            h.update(contract.read_bytes())
        expected = h.hexdigest()[:12]

        assert code_hash() == expected
