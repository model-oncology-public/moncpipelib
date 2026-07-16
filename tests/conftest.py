"""Root conftest.py for moncpipelib tests.

The integration marker is registered in pyproject.toml [tool.pytest.ini_options].
Integration tests live in tests/integration/ and are skipped by default
(via addopts = "-m 'not integration'"). Run them with:

    uv run pytest -m integration -v
"""
