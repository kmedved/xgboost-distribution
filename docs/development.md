# Development guide

Local setup, testing, and contribution workflow for `xgboost-distribution`.

## Setting up

The project uses standard `setuptools` + `setuptools_scm`. Python `>=3.10` is required.

```bash
git clone https://github.com/CDonnerer/xgboost-distribution.git
cd xgboost-distribution

python -m venv .venv
source .venv/bin/activate

pip install -e ".[testing]"
```

The `[testing]` extra pulls in `pytest`, `pytest-cov`, and `pandas`. The `[gpu]` extra installs full `xgboost` (with GPU support) instead of the default `xgboost-cpu`.

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com) for linting and formatting (ruff + a few file-hygiene hooks). Install once:

```bash
pip install pre-commit
pre-commit install
```

Hooks run on every commit. Configuration is in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

## Running tests

The test suite lives under [`tests/`](../tests) and uses pytest:

```bash
pytest                    # run everything
pytest tests/test_model.py
pytest -m "not slow"      # skip slow tests
```

Pytest config lives under `[tool:pytest]` in [`setup.cfg`](../setup.cfg). Coverage is automatically reported (configured by `--cov xgboost_distribution --cov-report term-missing`).

### With tox

```bash
pip install tox
tox                       # default env: pytest on the current Python
tox -e docs               # build the Sphinx docs
tox -e build              # build sdist + wheel
```

## Building the docs

Read the Docs is built from [`docs/`](.) via Sphinx. To build locally:

```bash
tox -e docs
# or
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

The Markdown docs in this directory render through [MyST-Parser](https://myst-parser.readthedocs.io/) (added to Sphinx via [`docs/conf.py`](conf.py)).

## Project layout

See [architecture.md](architecture.md) for the full source-tree map and how the pieces fit together.

```
xgboost-distribution/
├── src/xgboost_distribution/   # library code
├── tests/                      # pytest suite (mirrors src/ layout)
├── examples/                   # runnable example scripts (see examples.md)
├── docs/                       # Sphinx + Markdown documentation
├── imgs/                       # README images
├── setup.cfg, pyproject.toml   # packaging + lint config
├── tox.ini                     # tox environments
├── .github/workflows/          # CI: tests, build & publish
├── .pre-commit-config.yaml     # pre-commit hooks
└── .readthedocs.yml            # RTD build config
```

## Continuous integration

Two GitHub Actions workflows in [`.github/workflows/`](../.github/workflows):

- [`test.yml`](../.github/workflows/test.yml) — runs the pytest suite on Python 3.10, 3.11, and 3.12 against every push, plus a weekly cron. Reports coverage to Coveralls.
- [`build.yml`](../.github/workflows/build.yml) — on tag push (`v*`), builds the wheel and publishes to PyPI via `tox -e clean,build,publish`.

## Releasing

Versioning is handled automatically by `setuptools_scm` from git tags.

```bash
git tag -a v0.4.1 -m "Release v0.4.1"
git push origin v0.4.1
```

The `build.yml` workflow then publishes to PyPI (requires `PYPI_TOKEN` in repo secrets). Update [`CHANGELOG.rst`](../CHANGELOG.rst) before tagging.

## Adding a new distribution

The end-to-end recipe is in [architecture.md → "Adding a new distribution"](architecture.md#adding-a-new-distribution).

## Contributing

1. Open an issue describing the change before writing significant code.
2. Branch from `main` (the project does not use a long-lived `dev` branch).
3. Add or update tests under [`tests/`](../tests) — coverage should not drop.
4. Run `pre-commit run --all-files` and `pytest` locally before pushing.
5. Open a pull request. CI must pass before merge.

For numerical / mathematical changes, include the derivation in the docstring (see [`normal.py`](../src/xgboost_distribution/distributions/normal.py) for the template) and add a regression test against `scipy.stats`.
