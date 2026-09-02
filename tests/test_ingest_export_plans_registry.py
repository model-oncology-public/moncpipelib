"""Tests for the ExportPlan protocol + registry (#463 Step 2)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, ClassVar

import pytest

import moncpipelib
from moncpipelib.ingest._browser import BrowserSession
from moncpipelib.ingest.export_plans import (
    EXPORT_PLANS,
    ExportedFile,
    ExportPlan,
    get_export_plan,
    register_export_plan,
)
from moncpipelib.ingest.types import IngestContext


class _StubPlan:
    name: ClassVar[str] = "stub_plan"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def export(
        self,
        session: BrowserSession,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=Path("/tmp/x.download"), suggested_filename="x.csv")


class _DuckTypedPlanWithoutMultiFile:
    """No ``multi_file`` ClassVar -- pins the ``getattr``-with-default read (Gap 4)."""

    name: ClassVar[str] = "duck_plan"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def export(
        self,
        session: BrowserSession,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=Path("/tmp/y.download"), suggested_filename="y.csv")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Snapshot + restore the registry so tests don't leak stubs."""
    before = dict(EXPORT_PLANS)
    try:
        yield
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before)


def test_register_and_get_export_plan_roundtrip() -> None:
    plan = _StubPlan()
    register_export_plan(plan)
    assert get_export_plan("stub_plan") is plan


def test_get_export_plan_unknown_name_lists_known_plans() -> None:
    register_export_plan(_StubPlan())
    with pytest.raises(KeyError, match=r"Unknown export plan 'nope'.*stub_plan"):
        get_export_plan("nope")


def test_register_export_plan_overwrites_same_name() -> None:
    first = _StubPlan()
    second = _StubPlan()
    register_export_plan(first)
    register_export_plan(second)
    assert get_export_plan("stub_plan") is second


def test_moncpipelib_ships_no_builtin_export_plans() -> None:
    """Per-source plans live with consumers (data-platform); the library
    registry starts empty. Run in a subprocess so a sibling test's
    registration cannot leak into this assertion (per D3)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moncpipelib.ingest as m; assert m.EXPORT_PLANS == {}, m.EXPORT_PLANS",
        ],
        env={"PYTHONPATH": str(Path(moncpipelib.__file__).parent.parent)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_export_plan_protocol_isinstance_accepts_duck_typed_plan() -> None:
    stub = _DuckTypedPlanWithoutMultiFile()
    assert isinstance(stub, ExportPlan)
    assert getattr(stub, "multi_file", False) is False


def test_exported_file_is_frozen() -> None:
    exported = ExportedFile(path=Path("/tmp/x.download"), suggested_filename="x.csv")
    with pytest.raises(FrozenInstanceError):
        exported.path = Path("/tmp/y.download")  # type: ignore[misc]
