# Contributing

Include tests for changed behavior and update the documentation affected by
your change. The setup and checks below match the repository's CI workflow.

## Setup

[uv](https://docs.astral.sh/uv/) manages the environment:

```bash
uv sync --group dev               # create .venv with dev dependencies
uv sync --group dev --group docs  # include docs dependencies (optional)
```

## Everyday commands

| Command | Purpose |
| --- | --- |
| `make lint` | Check formatting, lint, type-check, and enforce import boundaries |
| `make test` | Check that `uv.lock` uses the standard index, then run tests with coverage |
| `make uv.lock` | Regenerate `uv.lock` from pypi.org, ignoring local uv configuration such as mirrors |
| `uv run pytest` | Run the test suite |
| `uv run pytest --cov` | Run tests with the coverage gate configured in `pyproject.toml` |
| `uv run ruff format .` | Format code |
| `uv run ruff check .` | Lint code |
| `uv run mypy` | Type-check in strict mode |
| `uv run lint-imports` | Enforce import boundaries |
| `uv run --group docs mkdocs serve` | Preview the documentation site |

Before opening a pull request, run:

```bash
make lint
make test
```

For documentation changes, also run `uv run --group docs mkdocs build --strict`.
Fix any reported failures before submitting the change. CI runs these same
checks; the preview server and lockfile regeneration are local tools.

## Architecture rules

The package has two tiers (see
[Architecture and code map](DESIGN.en.md#5-architecture-and-code-map)).
Imports point downward:

```
zoo | metrics | verify   built-in models, evaluation, verification
         core            engine, model types, DataFrame I/O, split
```

Import-linter (`lint-imports`) checks the package layers and keeps
`core.constants` and `core.errors` free of pandas dependencies. Other core
modules may use pandas.

### Checkpoint identity

Checkpoint identity contains only the function's `__module__:__qualname__`
and execution mode. It separates same-named functions in different modules but
accepts changes to code and bound parameters. The host must store and verify
construction and version metadata alongside the checkpoint.

Automatic fingerprints would need stable semantics for arbitrary Python
bindings, including mutable values, unpicklable objects, and representations
that contain memory addresses. A broader identity check therefore requires a
design and migration policy. Read the
[checkpoint rationale](docs/explanation/checkpoint-semantics.md#why-the-identity-is-deliberately-narrow)
before changing this boundary.

### Adding models, metrics, and verification

Add built-in models to `zoo`, evaluation functions to `metrics`, and execution
checks to `verify`. These packages can grow without expanding the core API.
Follow these rules when adding an entry:

1. Import only the core's public surface (`from qfeval_model.core import
   ...`). The three packages must not import one another. Existing verification
   code also uses the internal normalization interface documented in
   [the code map](DESIGN.en.md#51-dependencies-and-public-api); keep new
   dependencies on the public surface.
2. Group entries by category, with one module per category. Keep `__init__.py`
   limited to re-exports and `__all__`.
3. Use `@qm.model` for `zoo` entries by default. Each entry must be
   deterministic, checkpoint-ready at every step boundary, and pass
   `verify.check_all`. Add its factory to `tests/test_zoo_contract.py`. Zoo
   transforms may read only current and past data; forward returns belong to
   `metrics`. Batch mode requires an explicit semantic reason and documented,
   verified behavior across chunk boundaries.
4. Write each `metrics` entry as a pure function of `Features` or raw inputs,
   as appropriate for the metric. Metrics evaluate outputs and cannot be
   composed as model inputs.

## Tests and documentation

Tests live directly under `tests/`. The README's opening example is manually
transcribed in `tests/test_docs.py`. CI executes that copy, so update it
alongside the README whenever the example's behavior changes. Reviewers must
check that the two still match.

## Bilingual documents

English is the source of truth for the bilingual documents (`DESIGN.en.md` /
`DESIGN.ja.md` and `examples/en/` / `examples/ja/`): edit the English version
first and update the Japanese one in the same pull request. For example
scripts, CI checks matching file sets and cell counts, then compares the
parsed executable code after removing translated prose. Reviewers check
translation quality and consistency between the design documents.

## Releasing

The checkpoint envelope (`FORMAT_VERSION`), blake2b seed derivation, Philox
generator, and interpretation of timezone-naive timestamps as UTC are frozen
compatibility contracts. A change to any of them requires an explicit
migration or versioning decision. Bump the format version where applicable
and describe the compatibility impact in `CHANGELOG.md`.

1. Update `__version__` in `src/qfeval_model/_version.py`. Hatch, the public
   API, and checkpoint envelopes all read this value.
2. In `CHANGELOG.md`, add the new version's section and replace its
   `Unreleased` marker with the release date.
3. Confirm that the bilingual documents agree, as described above.
4. Open a pull request from `release/<version>` in this repository to `main`.
   Merge it after CI passes. The release workflow validates the merged
   commit, publishes to PyPI via trusted publishing, and creates a GitHub
   release tagged `v<version>` with the same distributions and checksums.

For the first release, configure the pending PyPI Trusted Publisher for
`pfnet-research/qfeval-model`, workflow `release.yml`, and environment `pypi`.
Publishing a GitHub release manually also runs the same build and PyPI
checks. Use one of these entry points per version.
