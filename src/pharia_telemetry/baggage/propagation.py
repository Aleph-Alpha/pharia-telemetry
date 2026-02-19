"""
OpenTelemetry baggage and span context propagation utilities.

This module provides helpers for propagating trace context across service boundaries
using OpenTelemetry baggage and setting span attributes for observability.

NOTE: If your application uses Pydantic Logfire, these utilities are generally not needed.
Pydantic Logfire provides built-in baggage convenience functions and automatically handles
baggage propagation. These are primarily for applications that don't use Pydantic Logfire.
"""

import functools
import logging
from typing import Any, Callable, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

try:
    from opentelemetry import baggage, trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - baggage propagation will be disabled")


P = ParamSpec("P")
R = TypeVar("R")


def _require_otel(
    fallback: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that guards a function behind OpenTelemetry availability.

    When OTel is not installed, the decorated function returns the fallback value
    and logs a debug message. When OTel is available but the function raises, the
    exception is logged and the fallback is returned.

    Args:
        fallback: Value to return when OTel is unavailable or an error occurs.
            Must be a valid instance of the decorated function's return type.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not OTEL_AVAILABLE:
                logger.debug("OpenTelemetry not available - skipping %s", fn.__name__)
                return fallback  # type: ignore[no-any-return]

            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error("Failed in %s: %s", fn.__name__, e)
                return fallback  # type: ignore[no-any-return]

        return wrapper

    return decorator


@_require_otel(fallback=None)
def set_baggage_span_attributes(
    span: Optional["trace.Span"] = None,
) -> None:
    """
    Set baggage items as span attributes for searchability.

    NOTE: This function is typically optional as baggage attributes can be automatically
    added to all spans by using the BaggageSpanProcessor. This function provides
    explicit/manual control when needed.

    Values are automatically retrieved from OpenTelemetry baggage and set 1:1 as span attributes.

    Args:
        span: Target span (uses current span if None)
    """
    target_span = span or trace.get_current_span()
    if not target_span:
        logger.warning("No span available to set attributes")
        return

    baggage_items = baggage.get_all()
    for key, value in baggage_items.items():
        if value is not None:
            target_span.set_attribute(key, str(value))
            logger.debug("Set span attribute %s from baggage: %s", key, value)


@_require_otel(fallback=None)
def set_gen_ai_span_attributes(
    operation_name: str,
    agent_id: str,
    span: Optional["trace.Span"] = None,
    conversation_id: str | None = None,
    model_name: str | None = None,
) -> None:
    """
    Set GenAI semantic attributes on a span for AI operations.

    This function sets OpenTelemetry GenAI semantic convention attributes
    on spans to enable proper categorization and monitoring of AI operations.

    Args:
        operation_name: The operation name (e.g., "chat", "execute_tool")
        agent_id: The agent identifier (e.g., "aa_qa_chat")
        span: Target span (uses current span if None)
        conversation_id: Optional conversation ID
        model_name: Optional model name
    """
    target_span = span or trace.get_current_span()
    if not target_span:
        logger.warning("No span available to set gen_ai attributes")
        return

    from pharia_telemetry.sem_conv.gen_ai import GenAI

    target_span.set_attribute(GenAI.OPERATION_NAME, operation_name)
    target_span.set_attribute(GenAI.AGENT_ID, agent_id)

    if conversation_id:
        target_span.set_attribute(GenAI.CONVERSATION_ID, str(conversation_id))

    if model_name:
        target_span.set_attribute(GenAI.REQUEST_MODEL, model_name)

    logger.debug(
        "Set gen_ai span attributes - operation: %s, agent: %s",
        operation_name,
        agent_id,
    )


@_require_otel(fallback=None)
def set_baggage_item(key: str, value: str) -> None:
    """
    Set a baggage item in the current context.

    Args:
        key: Baggage key
        value: Baggage value
    """
    baggage.set_baggage(key, value)
    logger.debug("Set baggage %s: %s", key, value)


@_require_otel(fallback=None)
def get_baggage_item(key: str) -> str | None:
    """
    Get a baggage item from the current context.

    Args:
        key: Baggage key

    Returns:
        Baggage value or None if not found
    """
    value = baggage.get_baggage(key)
    logger.debug("Retrieved baggage %s: %s", key, value)
    return str(value) if value is not None else None


@_require_otel(fallback={})
def get_all_baggage() -> dict[str, str]:
    """
    Get all baggage items from the current context.

    Returns:
        Dictionary of all baggage items
    """
    items = baggage.get_all()
    logger.debug("Retrieved all baggage: %s", items)
    return {k: str(v) if v is not None else "" for k, v in items.items()}
