# Changelog

## [Unreleased]

### Added
- GenAI constants now exposed directly from `pharia_telemetry.gen_ai` module for improved API convenience
- Comprehensive pre-commit hooks with mypy type checking
- Support for alpha pre-releases in release workflow

### Fixed
- MyPy type checking errors across all modules
- Pre-commit hook configuration to properly scope mypy checks
- Import paths for OpenTelemetry exporters
- Test assertions for baggage key formats

### Changed
- Improved API ergonomics - single import now provides both functions and constants
- Enhanced module documentation with usage examples

## [0.1.0-alpha.0] - 2024-01-XX

### Added
- Initial OpenTelemetry telemetry utilities for Pharia services
- GenAI span convenience functions following OpenTelemetry semantic conventions
- Baggage propagation utilities for context injection
- Structured logging integration with OpenTelemetry context
- Comprehensive test suite with unit and integration tests
