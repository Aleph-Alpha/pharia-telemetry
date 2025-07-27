"""Utility functions for telemetry and observability."""

from pharia_telemetry.utils.setup import (
    add_baggage_span_processor,
    add_console_exporter,
    add_otlp_exporter,
    create_tracer_provider,
    get_tracer,
    setup_basic_tracing,
    setup_otel_propagators,
)

__all__: list[str] = [
    "setup_otel_propagators",
    "create_tracer_provider",
    "add_baggage_span_processor",
    "add_otlp_exporter",
    "add_console_exporter",
    "setup_basic_tracing",
    "get_tracer",
]
