"""
OpenTelemetry setup utilities for Pharia services.

This module provides a single, consolidated setup function for OpenTelemetry tracing.
"""

from pharia_telemetry.setup.setup import (
    get_tracer,
    setup_telemetry,
)

__all__: list[str] = [
    "setup_telemetry",
    "get_tracer",
]
