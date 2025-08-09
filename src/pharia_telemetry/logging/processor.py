"""
Structlog-specific processors for OpenTelemetry context injection.

This module provides structlog processors that wrap the framework-agnostic injectors
to provide seamless integration with structlog's processing pipeline.
"""

import logging
from collections.abc import MutableMapping
from typing import Any

from pharia_telemetry.logging.injectors import (
    BaggageContextInjector,
    CompositeContextInjector,
    TraceContextInjector,
)

logger = logging.getLogger(__name__)

# Try to import structlog
try:
    import structlog  # noqa: F401

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class StructlogTraceProcessor:
    """
    Structlog processor that injects OpenTelemetry trace context.

    This processor wraps TraceContextInjector to provide a structlog-compatible
    interface for adding trace context to log events.

    Example usage:
        processor = StructlogTraceProcessor()

        # Add to structlog configuration
        structlog.configure(
            processors=[
                processor,
                structlog.processors.JSONRenderer()
            ]
        )
    """

    def __init__(
        self,
        include_trace_id: bool = True,
        include_span_id: bool = True,
        trace_id_key: str = "trace_id",
        span_id_key: str = "span_id",
    ):
        """
        Initialize the structlog trace processor.

        Args:
            include_trace_id: Whether to include trace_id in log records
            include_span_id: Whether to include span_id in log records
            trace_id_key: Key name for trace ID in log records
            span_id_key: Key name for span ID in log records
        """
        self.injector = TraceContextInjector(
            include_trace_id=include_trace_id,
            include_span_id=include_span_id,
            trace_id_key=trace_id_key,
            span_id_key=span_id_key,
        )

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Process log event and inject trace context.

        Args:
            logger: The logger instance (unused)
            method_name: The logging method name (unused)
            event_dict: The event dictionary to enhance

        Returns:
            Enhanced event dictionary with trace context
        """
        return self.injector.inject(dict(event_dict))


class StructlogBaggageProcessor:
    """
    Structlog processor that injects OpenTelemetry baggage context.

    This processor wraps BaggageContextInjector to provide a structlog-compatible
    interface for adding baggage context to log events.

    Example usage:
        processor = StructlogBaggageProcessor(prefix_filter="app.")

        # Add to structlog configuration
        structlog.configure(
            processors=[
                processor,
                structlog.processors.JSONRenderer()
            ]
        )
    """

    def __init__(
        self,
        prefix_filter: str | None = None,
        exclude_keys: set[str] | None = None,
    ):
        """
        Initialize the structlog baggage processor.

        Args:
            prefix_filter: Optional prefix to filter baggage keys
            exclude_keys: Optional set of baggage keys to exclude
        """
        self.injector = BaggageContextInjector(
            prefix_filter=prefix_filter,
            exclude_keys=exclude_keys,
        )

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Process log event and inject baggage context.

        Args:
            logger: The logger instance (unused)
            method_name: The logging method name (unused)
            event_dict: The event dictionary to enhance

        Returns:
            Enhanced event dictionary with baggage context
        """
        return self.injector.inject(dict(event_dict))


class StructlogCompositeProcessor:
    """
    Structlog processor that combines multiple context injectors.

    This processor wraps CompositeContextInjector to provide a structlog-compatible
    interface for adding multiple types of context to log events.

    Example usage:
        trace_injector = TraceContextInjector()
        baggage_injector = BaggageContextInjector(prefix_filter="app.")
        processor = StructlogCompositeProcessor([trace_injector, baggage_injector])

        # Add to structlog configuration
        structlog.configure(
            processors=[
                processor,
                structlog.processors.JSONRenderer()
            ]
        )
    """

    def __init__(self, injectors: list[Any]):
        """
        Initialize the structlog composite processor.

        Args:
            injectors: List of injector instances with inject() methods
        """
        self.injector = CompositeContextInjector(injectors)

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """
        Process log event and inject all contexts.

        Args:
            logger: The logger instance (unused)
            method_name: The logging method name (unused)
            event_dict: The event dictionary to enhance

        Returns:
            Enhanced event dictionary with all contexts
        """
        return self.injector.inject(dict(event_dict))


def create_structlog_trace_processor(
    include_trace_id: bool = True,
    include_span_id: bool = True,
    trace_id_key: str = "trace_id",
    span_id_key: str = "span_id",
) -> StructlogTraceProcessor:
    """
    Factory function to create a structlog trace processor.

    Args:
        include_trace_id: Whether to include trace_id in log records
        include_span_id: Whether to include span_id in log records
        trace_id_key: Key name for trace ID in log records
        span_id_key: Key name for span ID in log records

    Returns:
        Configured StructlogTraceProcessor instance
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError(
            "structlog is not available - cannot create structlog processor"
        )

    return StructlogTraceProcessor(
        include_trace_id=include_trace_id,
        include_span_id=include_span_id,
        trace_id_key=trace_id_key,
        span_id_key=span_id_key,
    )


def create_structlog_baggage_processor(
    prefix_filter: str | None = None,
    exclude_keys: set[str] | None = None,
) -> StructlogBaggageProcessor:
    """
    Factory function to create a structlog baggage processor.

    Args:
        prefix_filter: Optional prefix to filter baggage keys
        exclude_keys: Optional set of baggage keys to exclude

    Returns:
        Configured StructlogBaggageProcessor instance
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError(
            "structlog is not available - cannot create structlog processor"
        )

    return StructlogBaggageProcessor(
        prefix_filter=prefix_filter,
        exclude_keys=exclude_keys,
    )


def create_structlog_full_context_processor(
    include_trace_id: bool = True,
    include_span_id: bool = True,
    include_baggage: bool = True,
    baggage_prefix_filter: str | None = None,
    baggage_exclude_keys: set[str] | None = None,
    trace_id_key: str = "trace_id",
    span_id_key: str = "span_id",
) -> StructlogCompositeProcessor:
    """
    Factory function to create a full context structlog processor (trace + baggage).

    Args:
        include_trace_id: Whether to include trace_id in log records
        include_span_id: Whether to include span_id in log records
        include_baggage: Whether to include baggage context in log records
        baggage_prefix_filter: Optional prefix to filter baggage keys
        baggage_exclude_keys: Optional set of baggage keys to exclude
        trace_id_key: Key name for trace ID in log records
        span_id_key: Key name for span ID in log records

    Returns:
        Configured StructlogCompositeProcessor instance
    """
    if not STRUCTLOG_AVAILABLE:
        raise ImportError(
            "structlog is not available - cannot create structlog processor"
        )

    injectors: list[Any] = []

    # Add trace context injector if needed
    if include_trace_id or include_span_id:
        injectors.append(
            TraceContextInjector(
                include_trace_id=include_trace_id,
                include_span_id=include_span_id,
                trace_id_key=trace_id_key,
                span_id_key=span_id_key,
            )
        )

    # Add baggage context injector if needed
    if include_baggage:
        injectors.append(
            BaggageContextInjector(
                prefix_filter=baggage_prefix_filter,
                exclude_keys=baggage_exclude_keys,
            )
        )

    return StructlogCompositeProcessor(injectors)
