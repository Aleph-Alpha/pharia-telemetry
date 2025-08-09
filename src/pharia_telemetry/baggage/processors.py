"""
Custom OpenTelemetry span processors for automatic baggage propagation.

This module provides specialized span processors that enhance spans with
additional context and correlation data from OpenTelemetry baggage.

NOTE: If your application uses Pydantic Logfire, this span processor is generally not needed.
Pydantic Logfire automatically adds baggage to spans and provides built-in baggage management.
This is primarily for applications that don't use Pydantic Logfire.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import baggage
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from opentelemetry.trace import Span

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - span processors will be disabled")


class BaggageSpanProcessor(SpanProcessor):
    """
    Custom span processor that automatically adds OpenTelemetry baggage to spans as attributes.

    This processor automatically extracts all baggage items from the current context
    and sets them as span attributes, enabling searchability and correlation without
    manual intervention. Inherits from OpenTelemetry's SpanProcessor for proper integration.

    The processor runs during span start to ensure baggage attributes are present
    before any export logic executes.

    Example:
        from pharia_telemetry.baggage.processors import BaggageSpanProcessor

        processor = BaggageSpanProcessor()
        tracer_provider.add_span_processor(processor)

        # Now all spans automatically include baggage as attributes:
        # - app.user.id
        # - app.session.id
        # - pharia.chat.qa.conversation.id
        # - pharia.user.intent
        # - pharia.data.namespaces
        # - etc.
    """

    def __init__(self, prefix_filter: str | None = None):
        """
        Initialize the baggage span processor.

        Args:
            prefix_filter: Optional prefix to filter baggage keys (e.g., "app." to only include app.* keys)
        """
        self.prefix_filter = prefix_filter

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """
        Called when a span is started. Automatically adds baggage as span attributes.

        Args:
            span: The span being started
            parent_context: The parent context (unused but required by interface)
        """
        if not OTEL_AVAILABLE:
            logger.debug("OpenTelemetry not available - skipping baggage attributes")
            return

        try:
            # Get all baggage items from current context
            baggage_items = baggage.get_all()

            # Set each baggage item as a span attribute
            for key, value in baggage_items.items():
                if value is not None:
                    # Apply prefix filter if specified
                    if self.prefix_filter and not key.startswith(self.prefix_filter):
                        continue

                    span.set_attribute(key, str(value))
                    logger.debug(
                        "Auto-set span attribute %s from baggage: %s", key, value
                    )

        except Exception as e:
            logger.error("Failed to set baggage attributes on span: %s", e)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended. No action needed."""

    def shutdown(self) -> None:
        """Called when the span processor is shut down. No action needed."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Called to force flush any pending spans. No action needed."""
        return True
