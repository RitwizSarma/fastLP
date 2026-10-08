# PyPI Release Checklist

Use this checklist for the first PyPI release and for subsequent releases.

Last audited: 2026-08-16. Local tests, documentation, Linux wheel, source
distribution, Twine validation, and clean-install smoke tests pass. Items still
unchecked require repository configuration, cross-platform CI, TestPyPI/PyPI
authorization, documentation/README decisions, or a final tagged release.

## Release blockers

- [x] Keep Polars genuinely optional: declare it only under the `polars` extra,
  and ensure the README describes the same installation behavior.
- [x] Start from a clean, reviewed Git worktree. Commit intended source,
  documentation, tests, metadata, and lockfile changes; exclude generated
  benchmark output and operating-system metadata unless deliberately retained.
- [x] Add a GitHub Actions job that publishes the assembled wheel and source
  distribution artifacts to PyPI. Prefer PyPI Trusted Publishing with
  `pypa/gh-action-pypi-publish` and a protected GitHub environment.
- [ ] Configure the GitHub repository as a trusted publisher on PyPI. For the
  first release, create or reserve the PyPI project through the trusted
  publisher workflow as supported by PyPI.
- [x] Configure the TestPyPI account and pending trusted publisher.
- [x] Add a manually triggered TestPyPI publishing job that downloads all
  matrix artifacts and publishes through OIDC Trusted Publishing.
- [x] Exercise the complete release workflow on TestPyPI before publishing the
  first production release.

## Package metadata

- [x] Replace every `ritwiz_pls_fill` placeholder in `pyproject.toml` with the
  final author identity and documentation URL before building the release.
- [x] Fix API reference page in documentation.
- [ ] Confirm that the distribution name (`fastlp-py`) is accepted and
  available on both TestPyPI and PyPI.
  Search results are suggestive, but only a successful project registration or
  first upload conclusively establishes availability.
- [x] Confirm that the version in `pyproject.toml` and `rust/Cargo.toml` agrees
  with the intended release and has not already been uploaded. PyPI releases
  are immutable and an existing filename/version cannot be replaced.
- [x] Add the documentation URL after the site is published. Repository and
  issue-tracker URLs are already present.
- [x] Add author or maintainer metadata.
- [x] Add appropriate PyPI classifiers, including supported Python versions,
  operating systems, development status, audience, topic, and license.
- [x] Confirm the GPL license expression and bundled MIT attribution are
  correct and included in both wheels and source distributions.
- [x] Use Python 3.12 as the minimum. Keep
  `requires-python` and the PyO3 `abi3` minimum synchronized.
- [x] Expose the installed distribution version through `fastlp.__version__`.

## Documentation and user experience

- [x] Create and publish a package documentation site. Choose Read the Docs or
  a repository-specific GitHub Pages site such as
  `https://ritwizsarma.github.io/fastLP/`, then add its final URL to
  `[project.urls]` in `pyproject.toml`.
- [x] Change the primary installation instructions to `pip install fastlp-py` at
  release time. Keep source-development installation instructions separate.
- [x] Document the optional Polars installation command:
  `pip install "fastlp-py[polars]"`.
- [x] Ensure the README quick start is complete and runnable, including creation
  or loading of the example `data` object.
- [ ] Review public API documentation, parameters, accepted input types,
  covariance choices, warnings, fitted attributes, and plotting behavior.
- [ ] Add release notes when preparing the GitHub release; no standalone
  changelog is required for v0.1.0.
- [ ] Check all README links and verify that the rendered long description looks
  correct on TestPyPI.

## Tests and quality gates

- [x] Add a CI job that installs development dependencies with `uv` and runs
  `uv run pytest` on every pull request and release candidate.
- [x] Run the complete test suite locally (61 tests pass; seven expected
  few-cluster warnings).
- [ ] Build all release artifacts from a clean, tagged commit rather than from a
  dirty worktree.
- [x] Run `twine check` on the locally built Linux wheel and source
  distribution.
- [x] Install the locally built Linux wheel in a clean Python 3.12 environment
  with its normal
  dependencies and run a small end-to-end fit, result conversion, and plot.
- [x] Install from the source distribution in a clean Python 3.12 environment
  and run a
  smoke test, confirming that the documented Rust build prerequisites suffice.
- [ ] Test the base installation without Polars and `fastlp-py[polars]` with pandas,
  Polars `DataFrame`, and Polars `LazyFrame` inputs.
- [ ] Test supported Python versions and architectures in CI. At minimum, test
  Python 3.12 and a later stable Python against the `abi3` wheels.
- [ ] Run formatting, linting, type checking, and `git diff --check` once those
  project standards and tools are selected.
- [x] Review warnings emitted by the tests and confirm that each is expected.

## Artifact and platform checks

- [x] Build Linux x86-64 and aarch64, macOS x86-64 and arm64, and Windows
  x86-64 wheels, plus one source distribution.
- [x] Confirm Linux wheels use the intended manylinux compatibility baseline.
- [x] Inspect wheel metadata to verify dependencies, optional extras, license,
  Python requirement, and long-description content.
- [x] Inspect source-distribution contents for all Python and Rust sources,
  `Cargo.lock`, README, licenses, and attribution notices; exclude tests or
  include them deliberately according to the release policy.
- [x] Confirm the wheel imports without Cargo or a Rust toolchain.
- [x] Confirm no credentials, large benchmark artifacts, notebooks with private
  outputs, caches, profiles, or local machine metadata enter the artifacts.

## Release workflow

- [x] Use a published GitHub Release as the sole production publishing trigger,
  backed by an annotated `vX.Y.Z` tag on the reviewed release commit.
- [x] Ensure the publishing job runs only for the intended release event, waits
  for every build/test job, uses pinned or trusted actions, and has minimal
  permissions.
- [x] Prevent duplicate publication when both a tag push and GitHub release
  event occur.
- [x] Download all matrix artifacts into one directory and check that filenames
  are unique before publishing.
- [x] Publish first to TestPyPI and verify installation from TestPyPI on at least
  one clean supported platform.
- [ ] Tag the exact reviewed commit, publish the GitHub release, and allow CI to
  publish the immutable artifacts to PyPI.
- [ ] Verify the PyPI project page, metadata, release files, and install command
  immediately after publication.
- [ ] Install the released version from PyPI in a new environment and run the
  end-to-end smoke test again.
- [ ] Record checksums or retain the CI artifacts for provenance.

## Post-release

- [ ] Announce the release and link to its documentation and changelog.
- [ ] Monitor installation failures and platform-specific import errors.
- [ ] Create follow-up issues for deferred improvements rather than changing or
  attempting to overwrite the published release.
- [ ] Bump to the next development version, if the project adopts explicit
  development versions.
