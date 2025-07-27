# Changelog

## [0.0.1-alpha.0](https://github.com/Aleph-Alpha/pharia-telemetry/compare/v0.0.0-alpha.0...v0.0.1-alpha.0) (2025-07-27)


### Features

* add JFrog access token scripts and update CI workflow ([87c4466](https://github.com/Aleph-Alpha/pharia-telemetry/commit/87c44666333652a50bce039c9a243a3334cc086d))
* initial OpenTelemetry telemetry library for Pharia services ([3adf94d](https://github.com/Aleph-Alpha/pharia-telemetry/commit/3adf94d0a0d51efd5df1d09fb10c74b39e9dd9d8))

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
