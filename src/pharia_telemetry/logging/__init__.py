"""Logging utilities and structured logging components."""

# Primary API: Framework-agnostic injectors
from pharia_telemetry.logging.injectors import (
    BaggageContextInjector,
    CompositeContextInjector,
    TraceContextInjector,
    create_baggage_injector,
    create_full_context_injector,
    create_trace_injector,
)

# Modern alias for cleaner API
create_context_injector = create_full_context_injector

__all__: list[str] = [
    # Primary API: Framework-agnostic injectors
    "TraceContextInjector",
    "BaggageContextInjector",
    "CompositeContextInjector",
    "create_trace_injector",
    "create_baggage_injector",
    "create_full_context_injector",
    "create_context_injector",  # Modern alias
]
