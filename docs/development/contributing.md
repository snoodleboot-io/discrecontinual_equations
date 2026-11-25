# Contributing

## Development Setup

1. Clone the repository
2. Install development dependencies: `uv sync`
3. Install pre-commit hooks: `pre-commit install`

## Code Style

- Use Black for formatting
- Use Ruff for linting
- Follow Google docstring style

## Testing

Run tests with coverage: `uv run pytest --cov=discrecontinual_equations`

## Documentation

Update docs in `docs/` and build with `mkdocs build`

## Writing equations in the docs

The docs render LaTeX with MathJax. Write inline maths as `$x^\top S^{-1} x$` and display maths between `$$` lines. Docstrings rendered by mkdocstrings use the same syntax.
