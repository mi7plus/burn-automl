# Security policy

## Supported versions

`burn-automl` is under strict semver from 1.0 (see [docs/STABILITY.md](docs/STABILITY.md)).
Security fixes target the latest `1.x` release.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**, not via a public issue:

- Use GitHub's [private vulnerability reporting](https://github.com/mi7plus/burn-automl/security/advisories/new)
  ("Report a vulnerability" under the repository's *Security* tab), or
- open a regular issue **without exploit details** asking a maintainer to open a
  private channel.

Please include the affected version, a description, and a minimal reproduction if
you have one. We aim to acknowledge a report within a few days and to ship a fix
or mitigation as a patch release.

## Scope and assurances

- The core, task, and Burn adapter crates are `#![forbid(unsafe_code)]`; there is
  no `unsafe` in the workspace's own code.
- Dependencies are scanned against the [RustSec advisory database](https://rustsec.org)
  by a `cargo audit` CI job, and dependency updates are tracked by Dependabot.
- Deliberately held dependencies (`rand`, `burn` — see the root `Cargo.toml`) are
  monitored; a security advisory affecting a held version would override the hold.
