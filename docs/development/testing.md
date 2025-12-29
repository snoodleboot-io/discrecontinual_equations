# Testing

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=discrecontinual_equations --cov-report=html

# Run specific test
uv run pytest tests/test_parameter.py
```

## Coverage Report

Coverage reports are generated in `htmlcov/index.html` after running tests with the `--cov-report=html` flag.

## Test Structure

- `tests/` - Main test directory
- `tests/solver/` - Solver-specific tests
- Unit tests for core functionality
- Integration tests for complete workflows

## Adding Tests

1. Create test files in appropriate directories
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures and parametrize for comprehensive testing
