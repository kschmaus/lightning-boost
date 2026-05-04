# Contributing

## Development Setup

```bash
git clone https://github.com/kschmaus/ngboost-lightning.git
cd ngboost-lightning
uv sync --group dev
```

This installs the package in editable mode with all development dependencies.

## Running Tests

```bash
uv run pytest
```

The test suite includes 628 tests covering:

- Distribution correctness (gradient checks via finite differences, scipy cross-validation)
- NGBoost parity tests (same algorithm, same results within tolerance)
- Estimator tests (regressor, classifier, survival)
- Feature tests (sample weights, staged predictions, early stopping, col_sample, loss monitors)

Run a specific test file:

```bash
uv run pytest tests/test_normal.py -v
```

Skip slow tests:

```bash
uv run pytest -m "not slow"
```

## Linting and Type Checking

The project enforces strict linting with ruff and strict type checking with mypy:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy ngboost_lightning
```

Pre-commit hooks run these automatically on every commit:

```bash
uv run pre-commit install  # one-time setup
```

### Lint Configuration

- **ruff**: pycodestyle, pyflakes, isort (force-single-line), pydocstyle (Google convention), pyupgrade, flake8-annotations, flake8-bugbear, flake8-simplify, flake8-type-checking
- **mypy**: strict mode, all warnings enabled
- **Line length**: 88 characters

## Docstring Convention

All public classes and functions use Google-style docstrings with `Args:`,
`Returns:`, and `Raises:` sections. This is enforced by ruff's `D` rules
and rendered automatically by the documentation system.

## Building Documentation

```bash
uv sync --group docs
uv run mkdocs serve     # local preview at http://127.0.0.1:8000
uv run mkdocs build     # build static site to site/
```

## Project Structure

```
ngboost_lightning/
    __init__.py              # Public API exports
    engine.py                # Core NGBEngine boosting loop
    regressor.py             # LightningBoostRegressor (sklearn)
    classifier.py            # LightningBoostClassifier (sklearn)
    survival_estimator.py    # LightningBoostSurvival (sklearn)
    scoring.py               # ScoringRule protocol, LogScore, CRPScore
    survival.py              # CensoredLogScore, Y_from_censored
    evaluation.py            # Calibration, PIT, concordance index
    distributions/
        base.py              # Distribution ABC
        normal.py            # Normal
        lognormal.py         # LogNormal
        exponential.py       # Exponential
        gamma.py             # Gamma
        poisson.py           # Poisson
        laplace.py           # Laplace
        studentt.py          # StudentT
        weibull.py           # Weibull
        halfnormal.py        # HalfNormal
        categorical.py       # Bernoulli, Categorical
tests/
benchmarks/
examples/
docs/
```
