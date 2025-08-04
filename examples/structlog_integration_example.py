"""
Example demonstrating how to integrate pharia-telemetry context processors with structlog.

This example shows how to use the structlog processors to automatically inject
OpenTelemetry trace and baggage context into your structured logs.
"""

import structlog
from opentelemetry import baggage, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from pharia_telemetry.logging import (
    create_structlog_baggage_processor,
    create_structlog_processor,  # Alias for the above
    create_structlog_trace_processor,
)


def setup_basic_structlog_integration():
    """
    Example: Basic structlog integration with full context (trace + baggage).

    This is the recommended approach for most applications.
    """
    # Configure structlog with context processors
    structlog.configure(
        processors=[
            # Add OpenTelemetry context to all log events
            create_structlog_processor(),  # Includes both trace and baggage context
            # Standard structlog processors
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def setup_custom_structlog_integration():
    """
    Example: Custom structlog integration with specific processors.

    This shows how to configure individual processors for more control.
    """
    # Configure structlog with custom processors
    structlog.configure(
        processors=[
            # Add only trace context (no baggage)
            create_structlog_trace_processor(
                include_trace_id=True,
                include_span_id=True,
                trace_id_key="otel_trace_id",
                span_id_key="otel_span_id",
            ),
            # Add baggage context with filtering
            create_structlog_baggage_processor(
                prefix_filter="app.",  # Only include baggage keys starting with "app."
                exclude_keys={"sensitive_data"},  # Exclude specific keys
            ),
            # Standard structlog processors
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def setup_opentelemetry():
    """Set up basic OpenTelemetry tracing for demonstration."""
    # Set up basic tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Add console exporter for demonstration
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    return tracer


def demonstrate_logging_with_context():
    """Demonstrate logging with automatic context injection."""
    # Get a logger
    logger = structlog.get_logger()

    # Get a tracer
    tracer = setup_opentelemetry()

    # Set some baggage context
    ctx = baggage.set_baggage("app.user.id", "user123")
    ctx = baggage.set_baggage("app.session.id", "sess456", context=ctx)
    ctx = baggage.set_baggage("sensitive_data", "secret", context=ctx)

    with tracer.start_as_current_span("example_operation", context=ctx) as span:
        span.set_attribute("operation.type", "demo")

        # Log messages will automatically include trace and baggage context
        logger.info("Processing user request", user_action="login")
        logger.warning("Rate limit approaching", current_requests=95, limit=100)
        logger.error("Failed to process request", error_code="AUTH_FAILED")


def main():
    """Main demonstration function."""
    print("=== Basic Integration Demo ===")
    setup_basic_structlog_integration()
    demonstrate_logging_with_context()

    print("\n=== Custom Integration Demo ===")
    setup_custom_structlog_integration()
    demonstrate_logging_with_context()


if __name__ == "__main__":
    main()
