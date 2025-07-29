"""
OpenTelemetry baggage utilities for context propagation.

NOTE: If your application uses Pydantic Logfire, these utilities are generally not needed.
Pydantic Logfire provides built-in baggage convenience functions and automatically adds
baggage to spans. These are primarily for applications that don't use Pydantic Logfire.
"""

from pharia_telemetry.baggage.processors import BaggageSpanProcessor
from pharia_telemetry.baggage.propagation import (
    get_all_baggage,
    get_baggage_item,
    set_baggage_item,
    set_baggage_span_attributes,
    set_gen_ai_span_attributes,
)

__all__: list[str] = [
    # Span processors
    "BaggageSpanProcessor",
    # Propagation utilities
    "set_baggage_item",
    "get_baggage_item",
    "get_all_baggage",
    "set_baggage_span_attributes",
    "set_gen_ai_span_attributes",
]
