# Security Policy

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report privately via GitHub's **[private vulnerability reporting](https://github.com/model-oncology-public/moncpipelib/security/advisories/new)**
(Security tab -> "Report a vulnerability"). If you cannot use that channel,
email **security@modeloncology.com** with details and, if possible, a
proof-of-concept.

We aim to acknowledge reports within 5 business days and will coordinate a fix
and disclosure timeline with you. Please give us a reasonable window to release
a fix before any public disclosure.

## Supported Versions

Security fixes are applied to the latest released version on PyPI. Pin a recent
release and upgrade promptly when security releases are published.

## Security Posture

`moncpipelib` is the data-pipeline boundary library for a healthcare-data
platform, so its controls are designed for sensitive-data handling:

- **No secrets in source.** Database and cloud credentials are supplied at
  runtime via environment variables / secret managers, never committed.
- **Credential-safe connections.** Engines are built with
  `sqlalchemy.engine.URL.create()` and `hide_parameters=True`; connection
  errors are re-raised with host/port/database but never credentials.
- **TLS by default.** Database connections default to `sslmode=require`.
- **Parameterized queries.** User-provided values are always bound as query
  parameters; dynamic identifiers come from developer-controlled definitions,
  not external input.
- **No data values in logs.** Logging captures operational metadata (row
  counts, table/column names, run IDs), not row data.
- **Contract enforcement.** Schema conformance and column/table expectations
  are validated before writes; PII classification metadata is written in the
  same transaction as the data it describes.

This repository is a public, generated mirror of an internal codebase. It
contains source, tests with **synthetic fixtures only**, and documentation --
no production data, credentials, or internal infrastructure identifiers.
