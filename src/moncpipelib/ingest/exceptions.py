"""Runtime exceptions for the ingest boundary."""

from __future__ import annotations


class IngestResolutionError(Exception):
    """Raised when the consumer-side resolver cannot locate a blob.

    Distinct from :class:`moncpipelib.contracts.exceptions.ContractValidationError`
    because it fires at materialize / read time, not at contract load.
    The exactly-one-match rule in ``resolve_source_for_partition`` is
    the most common trigger: a glob that matches zero files usually
    means the ingest hasn't materialized yet; a glob matching more than
    one usually means upstream drift introduced an unexpected sibling.
    """
