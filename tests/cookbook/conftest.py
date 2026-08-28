"""Pytest plugin for cookbook test artifact collection.

Cookbook tests are normal pytest tests annotated with ``@pytest.mark.cookbook(...)``
and delimited by ``# --- cookbook:start ---`` / ``# --- cookbook:end ---`` comment
markers.  When the ``--cookbook-collect`` flag is passed, the plugin captures each
test's example source code, stdout, and log output, then writes a JSON artifact
file at session end for the generation script to consume.

Without ``--cookbook-collect``, the plugin is inert -- cookbook tests run as
ordinary unit tests with zero overhead.

Architecture note: pytest hook ``makereport(call)`` fires *before* fixture
teardown, so the hook only stashes pass/fail status on the test item.  The
autouse fixture reads caplog/capsys during its own teardown (after yield),
builds the full ``CookbookArtifact``, and appends it to the session-level list.
"""

from __future__ import annotations

import inspect
import json
import logging
import platform
import re
import textwrap
from collections.abc import Generator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stash key for pass/fail (written by hook, read by fixture teardown)
# ---------------------------------------------------------------------------

_cookbook_passed_key = pytest.StashKey[bool]()

# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------


@dataclass
class CookbookArtifact:
    """Data collected from a single cookbook test."""

    test_id: str
    title: str
    description: str
    category: str
    source_file: str
    source_line: int
    example_code: str
    log_output: str
    passed: bool
    order: int


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

_START_MARKER = re.compile(r"^\s*#\s*---\s*cookbook:start\s*---\s*$")
_END_MARKER = re.compile(r"^\s*#\s*---\s*cookbook:end\s*---\s*$")


def _resolve_test_source(func: Any, tests_root: Path) -> tuple[str, int]:
    """Resolve the source file and definition line of a cookbook test function.

    Decorators such as ``freezegun.freeze_time`` and ``unittest.mock.patch``
    replace the test with a wrapper whose code object lives in the wrapping
    library, so resolving ``item.function`` directly inlines that library's
    source into the cookbook (#392).  Unwrap the ``__wrapped__`` chain first,
    and refuse any resolution that lands outside ``tests/`` so a future
    non-unwrappable decorator fails generation loudly instead of corrupting
    the docs.
    """
    unwrapped = inspect.unwrap(func)
    source_file = inspect.getsourcefile(unwrapped)
    if source_file is None:
        raise RuntimeError(f"Could not resolve a source file for cookbook test {func!r}.")
    if not Path(source_file).resolve().is_relative_to(tests_root.resolve()):
        raise RuntimeError(
            f"Cookbook source for {func!r} resolved outside tests/: {source_file}. "
            "A decorator wrapper is likely masking the test function; it must set "
            "__wrapped__ (functools.wraps) so it can be unwrapped."
        )
    _, start_lineno = inspect.getsourcelines(unwrapped)
    return source_file, start_lineno


def _extract_example_code(filepath: str, func_lineno: int) -> tuple[str, int]:
    """Extract the code block between cookbook markers from a test function.

    Returns ``(example_code, absolute_line_number_of_start_marker)``.
    Falls back to the entire function body if no markers are found.
    """
    with open(filepath) as f:
        all_lines = f.readlines()

    # Slice from the function definition onward
    func_lines = all_lines[func_lineno - 1 :]

    start_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(func_lines):
        if _START_MARKER.match(line):
            start_idx = i + 1  # skip the marker line
        elif _END_MARKER.match(line) and start_idx is not None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        block = func_lines[start_idx:end_idx]
        abs_line = func_lineno + start_idx
    else:
        # Fallback: skip def line, use whole body
        block = func_lines[1:]
        abs_line = func_lineno + 1

    return textwrap.dedent("".join(block)).strip() + "\n", abs_line


# ---------------------------------------------------------------------------
# Session-level artifact collector
# ---------------------------------------------------------------------------

_collected_artifacts: list[CookbookArtifact] = []
_artifact_counter: int = 0


# ---------------------------------------------------------------------------
# Plugin hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cookbook", "Cookbook documentation generation")
    group.addoption(
        "--cookbook-collect",
        action="store_true",
        default=False,
        help="Collect cookbook test artifacts for documentation generation.",
    )
    group.addoption(
        "--cookbook-artifact",
        default=".cookbook_artifacts.json",
        help="Path to write the cookbook JSON artifact (default: .cookbook_artifacts.json).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cookbook(title, description, category): mark test as a cookbook example",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Any:  # noqa: ARG001
    """Stash test pass/fail status for the fixture to pick up during teardown."""
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    collecting = item.config.getoption("--cookbook-collect", default=False)
    if not collecting:
        return

    marker = item.get_closest_marker("cookbook")
    if marker is None:
        return

    item.stash[_cookbook_passed_key] = report.passed


@pytest.fixture(autouse=True)
def _cookbook_log_capture(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> Generator[None, None, None]:
    """Autouse fixture that captures logs/stdout and builds the artifact on teardown.

    Lifecycle:
      1. Hook ``makereport(call)`` stashes pass/fail on the item.
      2. This fixture's teardown (after yield) runs during fixture cleanup,
         *after* makereport(call) has fired.
      3. We read caplog, capsys, marker metadata, and source -- then build and
         append the ``CookbookArtifact``.
    """
    marker = request.node.get_closest_marker("cookbook")
    collecting = request.config.getoption("--cookbook-collect", default=False)

    if marker is None or not collecting:
        yield
        return

    with caplog.at_level(logging.DEBUG):
        yield

    # --- Teardown: build the artifact ---
    global _artifact_counter  # noqa: PLW0603

    item = request.node

    title = marker.kwargs.get("title", item.name)
    description = marker.kwargs.get("description", "")
    category = marker.kwargs.get("category", "general")

    tests_root = item.config.rootpath / "tests"
    source_file, start_lineno = _resolve_test_source(item.function, tests_root)  # type: ignore[union-attr]

    example_code, source_line = _extract_example_code(source_file, start_lineno)

    # Combine stdout + log output
    captured = capsys.readouterr()
    stdout_text = captured.out.rstrip()
    log_text = caplog.text.rstrip()
    parts = [p for p in (stdout_text, log_text) if p]
    log_output = "\n\n".join(parts)

    # Make source_file relative to the repo root
    repo_root = str(item.config.rootpath)
    if source_file.startswith(repo_root):
        source_file = source_file[len(repo_root) :].lstrip("/")

    passed = item.stash.get(_cookbook_passed_key, False)

    artifact = CookbookArtifact(
        test_id=item.nodeid,
        title=title,
        description=description,
        category=category,
        source_file=source_file,
        source_line=source_line,
        example_code=example_code,
        log_output=log_output,
        passed=passed,
        order=_artifact_counter,
    )
    _collected_artifacts.append(artifact)
    _artifact_counter += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Write collected cookbook artifacts to JSON."""
    collecting = session.config.getoption("--cookbook-collect", default=False)
    if not collecting:
        return

    if not _collected_artifacts:
        return

    artifact_path = session.config.getoption("--cookbook-artifact")
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pytest_version": pytest.__version__,
        "python_version": platform.python_version(),
        "artifacts": [asdict(a) for a in _collected_artifacts],
    }

    with open(artifact_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
