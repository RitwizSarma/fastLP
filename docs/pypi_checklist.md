# PyPI Release Checklist

Use this checklist for the first PyPI release and for subsequent releases.

## Release blockers

- [x] Keep Polars genuinely optional: declare it only under the `polars` extra,
  and ensure the README describes the same installation behavior.
- [ ] Start from a clean, reviewed Git worktree. Commit intended source,
  documentation, tests, metadata, and lockfile changes; exclude generated
  benchmark output and operating-system metadata unless deliberately retained.
- [ ] Add a GitHub Actions job that publishes the assembled wheel and source
  distribution artifacts to PyPI. Prefer PyPI Trusted Publishing with
  `pypa/gh-action-pypi-publish` and a protected GitHub environment.
- [ ] Configure the GitHub repository as a trusted publisher on PyPI. For the
  first release, create or reserve the PyPI project through the trusted
  publisher workflow as supported by PyPI.
- [ ] Exercise the complete release workflow on TestPyPI before publishing the
  first production release.

## Package metadata

- [ ] Confirm that the distribution name (`fastlp`) is available on PyPI.
  Search results are suggestive, but only a successful project registration or
  first upload conclusively establishes availability.
- [ ] Confirm that the version in `pyproject.toml` and `rust/Cargo.toml` agrees
  with the intended release and has not already been uploaded. PyPI releases
  are immutable and an existing filename/version cannot be replaced.
- [ ] Add project URLs for source repository, issue tracker, and documentation.
- [ ] Add author or maintainer metadata.
- [ ] Add appropriate PyPI classifiers, including supported Python versions,
  operating systems, development status, audience, topic, and license.
- [ ] Confirm the GPL license expression and bundled MIT attribution are
  correct and included in both wheels and source distributions.
- [ ] Decide whether Python 3.12 is the intended minimum. Keep
  `requires-python` and the PyO3 `abi3` minimum synchronized.
- [ ] Consider exposing the installed version through `__version__` or document
  `importlib.metadata.version("fastlp")` as the supported mechanism.

## Documentation and user experience

- [ ] Change the primary installation instructions to `pip install fastlp` at
  release time. Keep source-development installation instructions separate.
- [ ] Document the optional Polars installation command:
  `pip install "fastlp[polars]"`.
- [ ] Ensure the README quick start is complete and runnable, including creation
  or loading of the example `data` object.
- [ ] Review public API documentation, parameters, accepted input types,
  covariance choices, warnings, fitted attributes, and plotting behavior.
- [ ] Add a changelog or release notes describing v0.1.0, known limitations,
  compatibility, and migration expectations.
- [ ] Check all README links and verify that the rendered long description looks
  correct on TestPyPI.

## Tests and quality gates

- [ ] Add a CI job that installs development dependencies with `uv` and runs
  `uv run pytest` on every pull request and release candidate.
- [ ] Run the complete test suite locally from a clean checkout.
- [ ] Build all release artifacts from a clean, tagged commit rather than from a
  dirty worktree.
- [ ] Run `twine check` on every wheel and source distribution.
- [ ] Install each representative wheel in a clean environment with its normal
  dependencies and run a small end-to-end fit, result conversion, and plot.
- [ ] Install from the source distribution in a clean environment and run a
  smoke test, confirming that the documented Rust build prerequisites suffice.
- [ ] Test the base installation without Polars and `fastlp[polars]` with pandas,
  Polars `DataFrame`, and Polars `LazyFrame` inputs.
- [ ] Test supported Python versions and architectures in CI. At minimum, test
  Python 3.12 and a later stable Python against the `abi3` wheels.
- [ ] Run formatting, linting, type checking, and `git diff --check` once those
  project standards and tools are selected.
- [ ] Review warnings emitted by the tests and confirm that each is expected.

## Artifact and platform checks

- [ ] Build Linux x86-64 and aarch64, macOS x86-64 and arm64, and Windows
  x86-64 wheels, plus one source distribution.
- [ ] Confirm Linux wheels use the intended manylinux compatibility baseline.
- [ ] Inspect wheel metadata to verify dependencies, optional extras, license,
  Python requirement, and long-description content.
- [ ] Inspect source-distribution contents for all Python and Rust sources,
  `Cargo.lock`, README, licenses, and attribution notices; exclude tests or
  include them deliberately according to the release policy.
- [ ] Confirm wheel imports do not require Cargo or a Rust toolchain.
- [ ] Confirm no credentials, large benchmark artifacts, notebooks with private
  outputs, caches, profiles, or local machine metadata enter the artifacts.

## Release workflow

- [ ] Choose and document the release trigger: preferably an annotated `vX.Y.Z`
  tag or a published GitHub release from a protected branch.
- [ ] Ensure the publishing job runs only for the intended release event, waits
  for every build/test job, uses pinned or trusted actions, and has minimal
  permissions.
- [ ] Prevent duplicate publication when both a tag push and GitHub release
  event occur.
- [ ] Download all matrix artifacts into one directory and check that filenames
  are unique before publishing.
- [ ] Publish first to TestPyPI and verify installation from TestPyPI on at least
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
