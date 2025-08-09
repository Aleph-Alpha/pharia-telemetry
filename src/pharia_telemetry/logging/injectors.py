"""
Generic context injectors for structured logging.

This module provides modular, framework-agnostic injectors for adding OpenTelemetry
context to log records. These work with any logging framework that uses dictionaries
for structured data.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import baggage, trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - context injection will be disabled")


class ContextInjector(Protocol):
    """Protocol for context injectors."""

    def inject(self, log_dict: dict[str, Any]) -> dict[str, Any]:
        """Inject context into log dictionary."""
        ...


class TraceContextInjector:
    """
    Injects OpenTelemetry trace context (trace_id, span_id) into log records.

    This injector is framework-agnostic and works with any logging system that uses
    dictionaries for structured data (structlog, standard logging, custom frameworks).

    Example usage:
        injector = TraceContextInjector()
        log_dict = {"message": "Processing request"}
        enhanced = injector.inject(log_dict)
        # Result: {"message": "Processing request", "trace_id": "abc123...", "span_id": "def456..."}
    """

    def __init__(
        self,
        include_trace_id: bool = True,
        include_span_id: bool = True,
        trace_id_key: str = "trace_id",
        span_id_key: str = "span_id",
    ):
        """
        Initialize the trace context injector.

        Args:
            include_trace_id: Whether to include trace_id in log records
            include_span_id: Whether to include span_id in log records
            trace_id_key: Key name for trace ID in log records
            span_id_key: Key name for span ID in log records
        """
        self.include_trace_id = include_trace_id
        self.include_span_id = include_span_id
        self.trace_id_key = trace_id_key
        self.span_id_key = span_id_key

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available - TraceContextInjector will be disabled"
            )

    def inject(self, log_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Inject trace context into a log dictionary.

        Args:
            log_dict: Dictionary to enhance with trace context

        Returns:
            Enhanced dictionary with trace context
        """
        if not OTEL_AVAILABLE:
            return log_dict

        try:
            # Get current span context
            current_span = trace.get_current_span()
            if not current_span:
                return log_dict

            span_context = current_span.get_span_context()
            if not span_context.is_valid:
                return log_dict

            # Add trace ID if enabled
            if self.include_trace_id:
                log_dict[self.trace_id_key] = f"{span_context.trace_id:032x}"

            # Add span ID if enabled
            if self.include_span_id:
                log_dict[self.span_id_key] = f"{span_context.span_id:016x}"

        except Exception as e:
            # Silently log the error to avoid breaking the logging pipeline
            logger.debug("Failed to inject trace context: %s", e)

        return log_dict


class BaggageContextInjector:
    """
    Injects OpenTelemetry baggage context into log records.

    This injector is framework-agnostic and works with any logging system that uses
    dictionaries for structured data. It automatically includes all baggage items
    for comprehensive correlation across distributed systems.

    Example usage:
        injector = BaggageContextInjector()
        log_dict = {"message": "Processing request"}
        enhanced = injector.inject(log_dict)
        # Result: {"message": "Processing request", "app.user.id": "user123", "app.session.id": "sess456"}
    """

    def __init__(
        self,
        prefix_filter: str | None = None,
        exclude_keys: set[str] | None = None,
    ):
        """
        Initialize the baggage context injector.

        Args:
            prefix_filter: Optional prefix to filter baggage keys (e.g., "app.")
            exclude_keys: Optional set of baggage keys to exclude
        """
        self.prefix_filter = prefix_filter
        self.exclude_keys = exclude_keys or set()

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available - BaggageContextInjector will be disabled"
            )

    def inject(self, log_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Inject baggage context into a log dictionary.

        Args:
            log_dict: Dictionary to enhance with baggage context

        Returns:
            Enhanced dictionary with baggage context
        """
        if not OTEL_AVAILABLE:
            return log_dict

        try:
            # Get all baggage items for comprehensive correlation
            current_baggage = baggage.get_all()

            if current_baggage:
                for key, value in current_baggage.items():
                    if not value:  # Skip empty values
                        continue

                    # Skip excluded keys
                    if key in self.exclude_keys:
                        continue

                    # Apply prefix filter if specified
                    if self.prefix_filter and not key.startswith(self.prefix_filter):
                        continue

                    log_dict[key] = value

        except Exception as e:
            # Silently log the error to avoid breaking the logging pipeline
            logger.debug("Failed to inject baggage context: %s", e)

        return log_dict


class CompositeContextInjector:
    """
    Combines multiple context injectors into a single injector.

    This allows for flexible composition of different context injection strategies
    while maintaining a simple interface.

    Example usage:
        trace_injector = TraceContextInjector()
        baggage_injector = BaggageContextInjector(prefix_filter="app.")

        composite = CompositeContextInjector([trace_injector, baggage_injector])
        enhanced = composite.inject({"message": "test"})
    """

    def __init__(self, injectors: list[Any]):
        """
        Initialize the composite injector.

        Args:
            injectors: List of injector instances with inject() methods
        """
        self.injectors = injectors

    def inject(self, log_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Apply all injectors to the log dictionary.

        Args:
            log_dict: Dictionary to enhance

        Returns:
            Enhanced dictionary with all contexts
        """
        result = log_dict.copy()
        for injector in self.injectors:
            result = injector.inject(result)
        return result


def create_trace_injector(
    include_trace_id: bool = True,
    include_span_id: bool = True,
    trace_id_key: str = "trace_id",
    span_id_key: str = "span_id",
) -> TraceContextInjector:
    """
    Factory function to create a trace context injector.

    Args:
        include_trace_id: Whether to include trace_id in log records
        include_span_id: Whether to include span_id in log records
        trace_id_key: Key name for trace ID in log records
        span_id_key: Key name for span ID in log records

    Returns:
        Configured TraceContextInjector instance
    """
    return TraceContextInjector(
        include_trace_id=include_trace_id,
        include_span_id=include_span_id,
        trace_id_key=trace_id_key,
        span_id_key=span_id_key,
    )


def create_baggage_injector(
    prefix_filter: str | None = None,
    exclude_keys: set[str] | None = None,
) -> BaggageContextInjector:
    """
    Factory function to create a baggage context injector.

    Args:
        prefix_filter: Optional prefix to filter baggage keys
        exclude_keys: Optional set of baggage keys to exclude

    Returns:
        Configured BaggageContextInjector instance
    """
    return BaggageContextInjector(
        prefix_filter=prefix_filter,
        exclude_keys=exclude_keys,
    )


def create_full_context_injector(
    include_trace_id: bool = True,
    include_span_id: bool = True,
    include_baggage: bool = True,
    baggage_prefix_filter: str | None = None,
    baggage_exclude_keys: set[str] | None = None,
    trace_id_key: str = "trace_id",
    span_id_key: str = "span_id",
) -> CompositeContextInjector:
    """
    Factory function to create a full context injector (trace + baggage).

    Args:
        include_trace_id: Whether to include trace_id in log records
        include_span_id: Whether to include span_id in log records
        include_baggage: Whether to include baggage context in log records
        baggage_prefix_filter: Optional prefix to filter baggage keys
        baggage_exclude_keys: Optional set of baggage keys to exclude
        trace_id_key: Key name for trace ID in log records
        span_id_key: Key name for span ID in log records

    Returns:
        Configured CompositeContextInjector instance
    """
    injectors: list[Any] = []

    # Add trace context injector if needed
    if include_trace_id or include_span_id:
        injectors.append(
            create_trace_injector(
                include_trace_id=include_trace_id,
                include_span_id=include_span_id,
                trace_id_key=trace_id_key,
                span_id_key=span_id_key,
            )
        )

    # Add baggage context injector if needed
    if include_baggage:
        injectors.append(
            create_baggage_injector(
                prefix_filter=baggage_prefix_filter,
                exclude_keys=baggage_exclude_keys,
            )
        )

    return CompositeContextInjector(injectors)
