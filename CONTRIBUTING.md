# Contributing to moncpipelib

Thanks for your interest in contributing.

`moncpipelib` is maintained by Model Oncology and developed primarily against
our internal data-pipeline needs. We welcome issues and pull requests from the
community.

## How this repository is maintained

This public repository is a **one-way mirror** generated from an internal
repository. The `main` branch is republished (force-pushed) on each release, so
**commits pushed directly to `main` here do not persist**. That does not mean
your contribution is unwelcome -- it changes the mechanics:

- **Issues and discussions** are the best way to report bugs and propose
  changes; we triage them directly.
- **Pull requests** are reviewed here. When we accept one, we reincorporate the
  change into the internal source and it lands in the public repo on the next
  sync (with attribution preserved). Your PR branch is the unit of review; the
  public `main` is not a durable merge target.

If this workflow ever becomes a friction point, open an issue -- we would rather
adjust the process than lose a good contribution.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync --all-extras --dev
```

## Before you open a pull request

Run the full local check suite -- CI runs the same steps:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src
uv run pytest
```

If you change anything under `tests/cookbook/`, regenerate the cookbook docs
(CI fails if they are stale):

```bash
uv run pytest tests/cookbook/ --cookbook-collect
uv run python scripts/generate_cookbook.py
```

## Guidelines

- Target Python 3.11+. The codebase is fully typed and checked under mypy
  strict mode -- new code must type-check cleanly.
- I/O at boundaries (blob storage, database, archive, network, filesystem)
  must stream by default. Methods that return whole payloads as `bytes` are
  reserved for content that is contractually bounded to a few MB, and the
  docstring must say so.
- Keep changes focused and include tests.
- **Test fixtures must be synthetic.** Never derive a fixture from a real data
  extract. All sample data (identifiers, names, dates, claims, etc.) must be
  fabricated. PRs that add fixtures resembling real records will be rejected.
  Do not add binary fixtures (e.g. `.parquet`) without prior discussion in an
  issue.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## Security

Please do not file public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the disclosure process.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0, consistent with the rest of the project.
