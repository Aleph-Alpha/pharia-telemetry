"""
OpenTelemetry baggage and span context propagation utilities.

This module provides helpers for propagating trace context across service boundaries
using OpenTelemetry baggage and setting span attributes for observability.

NOTE: If your application uses Pydantic Logfire, these utilities are generally not needed.
Pydantic Logfire provides built-in baggage convenience functions and automatically handles
baggage propagation. These are primarily for applications that don't use Pydantic Logfire.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import baggage, trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - baggage propagation will be disabled")


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
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - skipping span attributes")
        return

    try:
        target_span = span or trace.get_current_span()
        if not target_span:
            logger.warning("No span available to set attributes")
            return

        # Get all baggage items and set them directly as span attributes
        baggage_items = baggage.get_all()
        for key, value in baggage_items.items():
            if value is not None:
                target_span.set_attribute(key, str(value))
                logger.debug("Set span attribute %s from baggage: %s", key, value)

    except Exception as e:
        logger.error("Failed to set baggage span attributes: %s", e)


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
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - skipping gen_ai span attributes")
        return

    try:
        target_span = span or trace.get_current_span()
        if not target_span:
            logger.warning("No span available to set gen_ai attributes")
            return

        # Import here to avoid circular imports
        from pharia_telemetry.sem_conv.gen_ai import GenAI

        # Set required GenAI semantic convention attributes
        target_span.set_attribute(GenAI.OPERATION_NAME, operation_name)
        target_span.set_attribute(GenAI.AGENT_ID, agent_id)

        # Set optional attributes
        if conversation_id:
            target_span.set_attribute(GenAI.CONVERSATION_ID, str(conversation_id))

        if model_name:
            target_span.set_attribute(GenAI.REQUEST_MODEL, model_name)

        logger.debug(
            "Set gen_ai span attributes - operation: %s, agent: %s",
            operation_name,
            agent_id,
        )

    except Exception as e:
        logger.error("Failed to set gen_ai span attributes: %s", e)


def set_baggage_item(key: str, value: str) -> None:
    """
    Set a baggage item in the current context.

    Args:
        key: Baggage key
        value: Baggage value
    """
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - skipping baggage setting")
        return

    try:
        baggage.set_baggage(key, value)
        logger.debug("Set baggage %s: %s", key, value)
    except Exception as e:
        logger.error("Failed to set baggage item %s: %s", key, e)


def get_baggage_item(key: str) -> str | None:
    """
    Get a baggage item from the current context.

    Args:
        key: Baggage key

    Returns:
        Baggage value or None if not found
    """
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - returning None for baggage")
        return None

    try:
        value = baggage.get_baggage(key)
        logger.debug("Retrieved baggage %s: %s", key, value)
        # Ensure we return a string or None
        return str(value) if value is not None else None
    except Exception as e:
        logger.error("Failed to get baggage item %s: %s", key, e)
        return None


def get_all_baggage() -> dict[str, str]:
    """
    Get all baggage items from the current context.

    Returns:
        Dictionary of all baggage items
    """
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - returning empty baggage")
        return {}

    try:
        items = baggage.get_all()
        logger.debug("Retrieved all baggage: %s", items)
        # Convert all values to strings to ensure type safety
        return {k: str(v) if v is not None else "" for k, v in items.items()}
    except Exception as e:
        logger.error("Failed to get all baggage items: %s", e)
        return {}
