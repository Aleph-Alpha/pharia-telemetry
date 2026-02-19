# CLAUDE.md

## Commands

```bash
make install-dev    # Install all dev dependencies + pre-commit hooks
make test           # Run tests (uv run pytest)
make coverage       # Run tests with coverage report
make lint           # Lint check (uv run ruff check .)
make format         # Auto-format (ruff format + ruff check --fix)
make type-check     # Type check (uv run mypy src/)
make build          # Build package
```

Run a single test: `uv run pytest tests/unit/test_gen_ai.py -k "test_name"`

## Architecture

```
src/pharia_telemetry/
├── __init__.py          # Public API (re-exports from submodules)
├── baggage/             # OpenTelemetry baggage get/set + propagation
├── logging/             # Log context injection (stdlib + structlog)
├── sem_conv/            # Semantic conventions (GenAI spans, baggage keys)
│   ├── baggage.py       # Baggage key constants
│   └── gen_ai.py        # GenAI span creation + constants
└── setup/               # setup_telemetry() + get_tracer()

tests/
├── unit/                # Unit tests (fast, no external deps)
└── integration/         # Integration tests (file export, etc.)
```

## Code Style

- Ruff for linting + formatting (line-length 88)
- Strict mypy (all `disallow_*` flags enabled)
- Google-style docstrings, type hints on all public functions
- `__init__.py` files use `F401` ignore for re-exports

## Conventions

- Conventional commits required: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Release-please automates versioning from commit history (hatch-vcs)
- Version is dynamic via git tags (`hatch-vcs`), not hardcoded
- Python 3.10 is the minimum target version
- Dependencies: `uv` (not pip) for all operations

## Gotchas

- Pre-commit runs ruff + mypy on `src/` only; always run `make format` before committing
- `requirements-dev.txt` is reference only; use `uv sync --group dev` instead
- `__init__.py` suppresses `F401` (unused imports) since it defines the public API
- mypy overrides in `pyproject.toml` relax type strictness for `tests.*`
