# Migration: Automated GitHub Releases + PyPI publishing on mirror sync

- Created: 2026-07-22
- State: started
- Owner: engineering@modeloncology.com
- Scope: CI/CD only (GitHub Actions). No package or runtime code changes.

Completion criteria: this change is only durable once the two workflow files are
reproduced by the private export tooling (see "Porting into the private tooling"),
and it is only "complete" after the first post-port mirror sync produces a matching
GitHub Release and PyPI release (see the verification appendix). Until then the files
added here are reference-only and are overwritten on the next release force-push.

## Summary

Before: `publish-pypi.yml` publishes to PyPI only when a GitHub Release is published,
and nothing creates that Release automatically -- it was a manual step. So PyPI and
GitHub releases did not mirror, and a mirror sync did not result in a PyPI publish.

After: a new `release.yml` watches for a successful CI run on a push to `main`
(what the upstream mirror force-push triggers). If `pyproject.toml` carries a version
with no existing `vX.Y.Z` tag, it cuts the GitHub Release (auto-generated notes) and
then invokes `publish-pypi.yml` to build and publish that exact tag to PyPI. The two
release channels now mirror each other and both fall out of the upstream mirror run.

## Motivation

- Make GitHub Releases and PyPI releases mirror each other (single source: the
  version bumped in the internal repo and carried over by the mirror).
- Make a PyPI release trigger automatically on an upstream mirror run, removing the
  manual "create a GitHub Release" step.

## Changes

- New: `.github/workflows/release.yml`
  - Triggers: `workflow_run` (CI completed) and manual `workflow_dispatch`.
  - `tag-and-release` job (`contents: write`): reads the version from `pyproject.toml`
    with `tomllib`; creates Release `vX.Y.Z` only if the tag does not already exist
    (idempotent -- tags survive the `main` force-push, so non-version-bump syncs are
    no-ops).
  - `publish` job: calls `publish-pypi.yml` via `workflow_call` for the new tag only
    when a release was created.
- Modified: `.github/workflows/publish-pypi.yml`
  - Added a `workflow_call` trigger with an optional `ref` input; the build checks out
    `inputs.ref || github.ref` so it builds the exact tag. The `release: published`
    and `workflow_dispatch` triggers, the two jobs, the `pypi` environment, and the
    SHA-pinned publish action are unchanged, so the manual path and the existing
    Trusted Publisher config keep working.

## Design constraints (why it looks this way)

- The default `GITHUB_TOKEN` does not chain workflows, so a Release created here would
  not fire `publish-pypi.yml`'s `release: published` trigger. `release.yml` therefore
  calls the publish workflow directly via `workflow_call` -- no PAT/App token, no new
  secrets.
- The squashed force-push leaves no diffable history, so "did the version change?" is
  answered by "does tag `vX.Y.Z` already exist?".
- Publishing to PyPI is irreversible (no re-upload of a version), so release creation
  is gated on CI success for the exact commit.

## Security controls (HIPAA / SOC2 / ISO 27001 / HITRUST context)

- Publish authentication remains OIDC Trusted Publishing; no long-lived PyPI token is
  introduced. No new repository secrets.
- Least privilege: `contents: write` is scoped to the release-creation job only;
  `id-token: write` is scoped to the publish path only. `release.yml` is otherwise
  `contents: read`.
- Supply-chain gate: release/publish only proceeds after CI (ruff, mypy, pytest,
  gitleaks) passes on the exact mirrored commit, so a malformed export cannot reach
  PyPI.
- The `pypi` GitHub Environment is retained and can carry a required-reviewer rule as
  an optional human approval gate before publish.
- Actions are SHA/major-tag pinned as in the existing workflows; no new third-party
  actions beyond `astral-sh/setup-uv`, `actions/*`, and the SHA-pinned
  `pypa/gh-action-pypi-publish` already in use.
- Injection-safe by construction: the `tag-and-release` job runs in the privileged
  `workflow_run` context but is gated to `event == 'push' && head_branch == 'main' &&
  conclusion == 'success'` (fork PRs run as `pull_request` and are excluded, defeating
  the fork-branch-naming spoof), checks out only maintainer-controlled `main` commits,
  and passes interpolated values (`version`, target SHA) through `env:` rather than into
  the shell body. OIDC is isolated to the non-checkout, `pypi`-gated publish job.

## Porting into the private tooling (durability follow-up -- REQUIRED)

The public tree is force-pushed on every release, so changes must be reproduced by the
private export tooling to persist:

1. Add `release.yml` to the workflows `scripts/oss-export/export.py` writes into
   `.github/workflows/`, carrying the "generated -- do not hand-edit" banner.
2. Update the private source/template for `publish-pypi.yml` with the `workflow_call`
   trigger and `ref` input.
3. Re-run the exporter so both land on the next sync.

## Prerequisites (one-time, no new secrets)

- PyPI Trusted Publisher already configured (Owner `model-oncology-public`, Repo
  `moncpipelib`, Workflow `publish-pypi.yml`, Environment `pypi`). See risk R1.
- The `pypi` GitHub Environment must exist.
- Confirm no org policy caps the `GITHUB_TOKEN` below the job-level `contents: write`
  requested by `release.yml`.

## Risks

- R1 -- Trusted Publishing under `workflow_call`. PyPI matches the reusable workflow via
  the OIDC `job_workflow_ref` claim (`publish-pypi.yml`), so the existing publisher
  config should hold. If the first reused publish is rejected as an invalid publisher,
  add a second Trusted Publisher on PyPI for the caller filename (`release.yml`).
  Exercised by verification step 3 before it is relied on.
- R2 -- Publish fails after the Release is created. The tag then exists, so later syncs
  skip re-release. Recovery: re-run the failed publish job, or run `publish-pypi.yml`
  via `workflow_dispatch` for that release.

## Rollback

Remove `release.yml` and revert the `workflow_call`/`ref` additions to
`publish-pypi.yml` in the private export tooling; re-run the exporter. Manual releases
(create a GitHub Release -> `publish-pypi.yml` on `release: published`) continue to work
unchanged.

## Verification appendix

Record command outputs here as each step is executed. Repo:
`model-oncology-public/moncpipelib`.

### BEFORE (capture before the first post-port sync)

```
# GitHub Releases (expect none, or only manually-created ones):
gh release list --repo model-oncology-public/moncpipelib

# Current version live on PyPI:
curl -s https://pypi.org/pypi/moncpipelib/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
```

Output (record):
```
<paste output>
```

### Static validation (this PR)

```
podman run --rm -v "$PWD":/repo:ro -w /repo docker.io/rhysd/actionlint:latest -no-color
```

Output (2026-07-22): no findings; exit 0. Both `ci.yml`, `publish-pypi.yml`, and
`release.yml` parse and lint clean (expressions, contexts, and shell steps).

### OIDC/publisher plumbing check (safe, pre-sync)

Manually run `publish-pypi.yml` via `workflow_dispatch` against the current version.
Expected: OIDC auth succeeds, then PyPI rejects the upload as a duplicate
(`400 File already exists`) -- which confirms the Trusted Publisher path works.

Output (record):
```
<paste run URL + result>
```

### Reusable-call + backfill check

Manually run `release.yml` via `workflow_dispatch`. Expected: creates Release `v0.45.0`
if missing (backfilling the currently-unreleased version), then calls `publish-pypi.yml`
via `workflow_call`; publish safely duplicate-rejects on PyPI. This exercises the
`job_workflow_ref` claim (R1).

Output (record):
```
<paste run URL + result>
```

### AFTER (first post-port sync with a new version)

```
gh release list --repo model-oncology-public/moncpipelib
curl -s https://pypi.org/pypi/moncpipelib/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
```

Expected: a new `vX.Y.Z` GitHub Release exists with generated notes, and the same new
version is live on PyPI, with no manual action.

Output (record):
```
<paste output>
```
