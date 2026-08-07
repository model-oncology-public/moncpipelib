"""Asset versioning utilities for Dagster code_version."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from moncpipelib.config import CONTRACT_FILE_PATTERN


def code_hash(*, _stack_depth: int = 1) -> str:
    """Generate a deterministic hash of the calling module and its sibling contracts.

    Hashes the caller's ``.py`` file and any ``*.contract.yaml`` files in the
    same directory. Returns a short hex digest suitable for Dagster's
    ``code_version`` parameter.

    Runs once at import time per ``@asset`` decorator -- negligible cost.

    Usage::

        from moncpipelib import code_hash

        @asset(code_version=code_hash())
        def my_asset(...): ...

    The hash changes when:

    - The asset's Python source file changes
    - Any contract YAML in the same directory changes

    Args:
        _stack_depth: Internal parameter for call-site resolution. Do not
            override unless wrapping this function in another helper.

    Returns:
        12-character hex digest string.
    """
    caller_file = Path(inspect.stack()[_stack_depth].filename)
    h = hashlib.sha256()
    h.update(caller_file.read_bytes())
    for contract in sorted(caller_file.parent.glob(CONTRACT_FILE_PATTERN)):
        h.update(contract.read_bytes())
    return h.hexdigest()[:12]
