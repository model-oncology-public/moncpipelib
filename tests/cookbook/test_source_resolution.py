"""Regression tests for cookbook source resolution (#392).

Cookbook tests wrapped by ``@freeze_time`` or ``@mock.patch`` used to resolve
their source through the decorator's wrapper closure, inlining freezegun /
stdlib-mock source into ``docs/cookbook.md``.  ``_resolve_test_source`` must
unwrap the decorator chain and reject any resolution outside ``tests/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from freezegun import freeze_time

from tests.cookbook.conftest import _resolve_test_source

TESTS_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()


def _plain_example() -> None:
    pass


@freeze_time("2026-01-01")
def _freezegun_example() -> None:
    pass


@patch("json.loads")
def _mock_patch_example(mock_loads: MagicMock) -> None:
    pass


class TestResolveTestSource:
    def test_plain_function_resolves_to_test_module(self) -> None:
        source_file, lineno = _resolve_test_source(_plain_example, TESTS_ROOT)
        assert Path(source_file).resolve() == THIS_FILE
        assert lineno > 0

    def test_freezegun_wrapped_function_resolves_to_test_module(self) -> None:
        source_file, _ = _resolve_test_source(_freezegun_example, TESTS_ROOT)
        assert Path(source_file).resolve() == THIS_FILE

    def test_mock_patch_wrapped_function_resolves_to_test_module(self) -> None:
        source_file, _ = _resolve_test_source(_mock_patch_example, TESTS_ROOT)
        assert Path(source_file).resolve() == THIS_FILE

    def test_source_outside_tests_raises(self) -> None:
        with pytest.raises(RuntimeError, match="outside tests/"):
            _resolve_test_source(json.loads, TESTS_ROOT)
