# Contributing

Thanks for your interest in `xgboost-distribution`. The full development guide — local setup, running tests, building docs, releasing, and the recipe for adding a new distribution — lives in [docs/development.md](docs/development.md).

Quick checklist before opening a PR:

1. Open an issue first for non-trivial changes.
2. Branch from `main`.
3. Add or update tests under [`tests/`](tests).
4. Run `pre-commit run --all-files` and `pytest` locally.
5. Update [CHANGELOG.rst](CHANGELOG.rst) under the development version.

For maths/numerical changes, include the derivation in the docstring of the distribution (see [`normal.py`](src/xgboost_distribution/distributions/normal.py) for the template) and add a regression test against `scipy.stats`.
