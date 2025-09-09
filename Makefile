.PHONY: help install install-dev test coverage test-cov lint format type-check clean build docs


help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  coverage     Run tests with coverage (alias for test-cov)"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with ruff"
	@echo "  type-check   Run type checking with mypy"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  docs         Build documentation"

install:
	uv sync

install-dev:
	uv sync --group dev --group docs
	uv run pre-commit install

test:
	uv run pytest

coverage:
	uv run pytest --cov=pharia_telemetry --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated at: htmlcov/index.html"

test-cov:
	uv run pytest --cov=pharia_telemetry --cov-report=html --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy --install-types --non-interactive src/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

docs:
	cd docs && make html
