# Contributing to Pharia Telemetry

Thank you for your interest in contributing to Pharia Telemetry! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all participants with respect and create a welcoming environment for everyone.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Basic understanding of OpenTelemetry concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/pharia-telemetry.git
   cd pharia-telemetry
   ```

2. **Install Dependencies** (with uv - recommended)
   ```bash
   uv sync --group dev --group docs
   ```

   Or with pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,docs]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   uv run pre-commit install  # or just: pre-commit install
   ```

4. **Verify Setup**
   ```bash
   uv run pytest  # or just: pytest
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed
- Follow the established project structure

### 3. Run Tests and Quality Checks

```bash
# Run all tests
uv run pytest

# Run with coverage (generates HTML report)
uv run pytest --cov=pharia_telemetry --cov-report=html --cov-report=term-missing

# View coverage report in browser
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux

# Code formatting and linting with ruff
uv run ruff format .
uv run ruff check --fix .

# Type checking
uv run mypy --install-types --non-interactive src/

# Linting only
uv run ruff check .
```

### 4. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
git commit -m "feat: add new baggage utility function"
git commit -m "fix: resolve logging context issue"
git commit -m "docs: update API documentation"
```

**Commit Types:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Include screenshots if UI changes are involved

## Code Style Guidelines

### Python Code Style

- **Formatting**: We use Black with 88 character line length
- **Import Sorting**: We use isort with Black profile
- **Type Hints**: All public functions should have type hints
- **Docstrings**: Use Google-style docstrings

```python
def example_function(param: str, optional_param: Optional[int] = None) -> bool:
    """Example function with proper type hints and docstring.

    Args:
        param: Description of the parameter.
        optional_param: Description of the optional parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param is invalid.
    """
    pass
```

### Testing Guidelines

- **Test Structure**: Use the Arrange-Act-Assert pattern
- **Test Names**: Descriptive names that explain what's being tested
- **Coverage**: Aim for high test coverage, especially for critical paths
- **Markers**: Use pytest markers to categorize tests (`@pytest.mark.unit`, `@pytest.mark.integration`)

```python
def test_function_should_return_true_when_valid_input():
    # Arrange
    input_value = "valid_input"

    # Act
    result = function_under_test(input_value)

    # Assert
    assert result is True
```

## Project Structure

```
src/pharia_telemetry/
├── __init__.py           # Main package
├── baggage/              # OpenTelemetry baggage utilities
├── constants/            # Shared constants
├── logging/              # Logging utilities
└── utils/                # General utilities

tests/
├── unit/                 # Unit tests
├── integration/          # Integration tests
└── conftest.py          # Pytest configuration
```

## Documentation

- **API Documentation**: Use docstrings that work with Sphinx
- **Examples**: Include usage examples in docstrings
- **README**: Keep the main README.md up to date
- **Changelog**: Update CHANGELOG.md for all changes

## Release Process

1. Update version in `src/pharia_telemetry/__init__.py`
2. Update `CHANGELOG.md` with new version
3. Create a git tag: `git tag v1.0.0`
4. Push tags: `git push --tags`
5. GitHub Actions will handle the rest

## Issue Reporting

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets or logs

## Feature Requests

Before requesting a new feature:

1. Check if it already exists
2. Search existing issues and discussions
3. Provide a clear use case
4. Consider if it fits the library's scope

## Questions and Support

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: conrad.poepke@aleph-alpha.com for urgent matters

## License

By contributing to Pharia Telemetry, you agree that your contributions will be licensed under the MIT License.
