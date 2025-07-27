"""OpenTelemetry baggage utilities for context propagation."""

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
