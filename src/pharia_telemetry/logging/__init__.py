"""
Logging utilities and structured logging components.

NOTE: If your application uses Pydantic Logfire, these logging utilities are generally not needed.
Pydantic Logfire provides comprehensive structured logging with automatic trace and baggage
context injection. These are primarily for applications that don't use Pydantic Logfire.
"""

# Primary API: Framework-agnostic injectors
from pharia_telemetry.logging.injectors import (
    BaggageContextInjector,
    CompositeContextInjector,
    TraceContextInjector,
    create_baggage_injector,
    create_full_context_injector,
    create_trace_injector,
)

# Structlog-specific processors
from pharia_telemetry.logging.processor import (
    StructlogBaggageProcessor,
    StructlogCompositeProcessor,
    StructlogTraceProcessor,
    create_structlog_baggage_processor,
    create_structlog_full_context_processor,
    create_structlog_trace_processor,
)

# Modern aliases for cleaner API
create_context_injector = create_full_context_injector
create_structlog_processor = create_structlog_full_context_processor

__all__: list[str] = [
    # Primary API: Framework-agnostic injectors
    "TraceContextInjector",
    "BaggageContextInjector",
    "CompositeContextInjector",
    "create_trace_injector",
    "create_baggage_injector",
    "create_full_context_injector",
    "create_context_injector",  # Modern alias
    # Structlog-specific processors
    "StructlogTraceProcessor",
    "StructlogBaggageProcessor",
    "StructlogCompositeProcessor",
    "create_structlog_trace_processor",
    "create_structlog_baggage_processor",
    "create_structlog_full_context_processor",
    "create_structlog_processor",  # Modern alias
]
