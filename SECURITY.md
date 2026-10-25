# Security Policy

## Supported versions

Security fixes are provided for the latest released version on PyPI only.
The project is in the `0.x` series, where breaking changes between minor
versions are allowed, so fixes are not backported.

## Reporting a vulnerability

Please report vulnerabilities **privately**. Do not open a public issue or pull
request for a suspected vulnerability.

Use GitHub's private vulnerability reporting:
<https://github.com/masahiroyecon1997dev/econometricsmodels/security/advisories/new>

Please include:

- the affected version (`pip show econometricsmodels`) and your Python version
  and platform
- a description of the problem and its impact
- a minimal reproduction, if possible

This project is maintained by one person on a best-effort basis. There is no
guaranteed response or fix time, but reports are read and handled as quickly as
possible. Once a fix is released, the advisory is published and the reporter is
credited unless they prefer otherwise.

## Scope

In scope: vulnerabilities in the code of this repository and in the published
wheels and source distribution (for example, memory-safety problems, crashes
reachable from crafted input, or problems in the release pipeline).

Out of scope: vulnerabilities in dependencies such as `polars` or the Rust
crates in `Cargo.lock`. Please report those to the upstream project. If a
dependency issue affects this package, a report here is still welcome so the
dependency can be updated.
