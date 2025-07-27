"""
GenAI convenience functions for OpenTelemetry semantic conventions.

This module provides high-level convenience functions for creating and managing
GenAI spans following OpenTelemetry semantic conventions.

Example usage:
    ```python
    from pharia_telemetry.gen_ai import setChatSpan, GenAI, set_genai_span_usage

    # All constants and functions available from single import
    with setChatSpan(
        model="gpt-4",
        system=GenAI.Values.System.OPENAI,
        agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
        conversation_id="conv-123"
    ) as span:
        # Your AI operation here
        response = call_ai_model()

        # Record token usage
        set_genai_span_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
    ```

Based on:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from pharia_telemetry.constants.gen_ai import GenAI
from pharia_telemetry.utils.setup import get_tracer, setup_basic_tracing

logger = logging.getLogger(__name__)

try:
    # Check if OpenTelemetry is available
    import opentelemetry  # noqa: F401

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - GenAI functions will be limited")


def setup_telemetry(
    service_name: str,
    service_version: Optional[str] = None,
    environment: Optional[str] = None,
    enable_baggage_processor: bool = True,
    enable_console_exporter: bool = False,
) -> bool:
    """
    Set up OpenTelemetry tracing with sensible defaults for Pharia services.

    This is the main entry point for most applications - it configures everything
    needed for distributed tracing and context propagation.

    Args:
        service_name: Name of the service (required)
        service_version: Version of the service (optional)
        environment: Environment name (dev, staging, prod) (optional)
        enable_baggage_processor: Whether to enable automatic baggage->span attributes (default: True)
        enable_console_exporter: Whether to enable console output for debugging (default: False)

    Returns:
        True if setup succeeded, False if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        return False

    try:
        additional_attributes = {}
        if environment:
            additional_attributes["deployment.environment"] = environment

        tracer_provider = setup_basic_tracing(
            service_name=service_name,
            service_version=service_version,
            enable_baggage_processor=enable_baggage_processor,
            enable_console_exporter=enable_console_exporter,
            additional_resource_attributes=additional_attributes,
        )
        return tracer_provider is not None
    except Exception as e:
        logger.error("Failed to setup telemetry: %s", e)
        return False


# =============================================================================
# GenAI Span Creation Functions
# =============================================================================


@contextmanager
def create_genai_span(
    operation_name: str,
    agent_name: str,
    *,
    model: Optional[str] = None,
    system: str = GenAI.Values.System.OPENAI,
    span_name: Optional[str] = None,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Create a GenAI span following OpenTelemetry semantic conventions.

    This function creates a properly formatted GenAI span with all the required
    and recommended attributes according to OpenTelemetry GenAI semantic conventions.

    Args:
        operation_name: The type of GenAI operation (e.g., "chat", "generate_content")
        model: The model being used (e.g., "gpt-4", "claude-3")
        system: The GenAI system (default: "openai")
        span_name: Custom span name (default: f"gen_ai {operation_name}")
        conversation_id: Unique conversation identifier
        agent_name: Display name of the agent
        agent_id: Unique identifier for the agent
        additional_attributes: Additional span attributes

    Yields:
        Span: The created OpenTelemetry span

    Example:
        ```python
        from pharia_telemetry import create_genai_span
        from pharia_telemetry.constants.gen_ai import GenAI

        with create_genai_span(
            operation_name=GenAI.Values.OperationName.CHAT,
            model="gpt-4",
            system=GenAI.Values.System.OPENAI,
            conversation_id="conv-123",
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT
        ) as span:
            # Your GenAI operation here
            response = call_ai_model()

            # Set usage information
            set_genai_span_usage(input_tokens=100, output_tokens=50)
        ```
    """
    if not OTEL_AVAILABLE:
        yield None
        return

    tracer = get_tracer()
    if not tracer:
        yield None
        return

    # Build span name
    if span_name is None:
        span_name = f"{operation_name} {agent_name}"

    # Build attributes
    attributes = {
        GenAI.OPERATION_NAME: operation_name,
        GenAI.SYSTEM: system,
    }

    # Add optional attributes
    if model:
        attributes[GenAI.REQUEST_MODEL] = model
    if conversation_id:
        attributes[GenAI.CONVERSATION_ID] = conversation_id
    if agent_name:
        attributes[GenAI.AGENT_NAME] = agent_name
    if agent_id:
        attributes[GenAI.AGENT_ID] = agent_id

    # Add any additional attributes
    if additional_attributes:
        attributes.update(additional_attributes)

    # Create and yield the span
    with tracer.start_as_current_span(span_name, attributes=attributes) as span:
        yield span


def setChatSpan(
    *,
    model: Optional[str] = None,
    system: str = GenAI.Values.System.OPENAI,
    conversation_id: Optional[str] = None,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    agent_name: Optional[str] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function for chat operations.

    Args:
        model: The model being used (e.g., "gpt-4", "claude-3")
        system: The GenAI system (default: "openai")
        conversation_id: Unique conversation identifier
        agent_id: Agent identifier (default: QA_CHAT)
        agent_name: Display name of the agent
        additional_attributes: Additional span attributes

    Returns:
        Context manager for the GenAI chat span
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.CHAT,
        agent_name=agent_name or "chat_agent",
        model=model,
        system=system,
        conversation_id=conversation_id,
        agent_id=agent_id,
        additional_attributes=additional_attributes,
    )


def setToolExecutionSpan(
    tool_name: str,
    *,
    tool_description: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function for tool execution operations.

    Args:
        tool_name: Name of the tool being executed
        tool_description: Description of the tool
        tool_call_id: Unique identifier for this tool call
        conversation_id: Unique conversation identifier
        agent_id: Agent identifier (default: QA_CHAT)
        additional_attributes: Additional span attributes

    Returns:
        Context manager for the GenAI tool execution span
    """
    tool_attributes = {
        GenAI.TOOL_NAME: tool_name,
    }

    if tool_description:
        tool_attributes[GenAI.TOOL_DESCRIPTION] = tool_description
    if tool_call_id:
        tool_attributes[GenAI.TOOL_CALL_ID] = tool_call_id

    if additional_attributes:
        tool_attributes.update(additional_attributes)

    return create_genai_span(
        operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
        agent_name=tool_name,  # Use tool name as agent name
        span_name=f"execute_tool {tool_name}",
        conversation_id=conversation_id,
        agent_id=agent_id,
        additional_attributes=tool_attributes,
    )


def setAgentCreationSpan(
    *,
    agent_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function for agent creation operations.

    Args:
        agent_name: Name of the agent being created
        agent_id: Unique identifier for the agent
        conversation_id: Unique conversation identifier
        additional_attributes: Additional span attributes

    Returns:
        Context manager for the GenAI agent creation span
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.CREATE_AGENT,
        agent_name=agent_name or "unknown_agent",
        agent_id=agent_id or GenAI.Values.PhariaAgentId.AGENT_CREATION,
        conversation_id=conversation_id,
        additional_attributes=additional_attributes,
    )


def setEmbeddingsSpan(
    *,
    model: Optional[str] = None,
    system: str = GenAI.Values.System.OPENAI,
    encoding_format: Optional[str] = None,
    dimensions: Optional[int] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function for embeddings operations.

    Args:
        model: The model being used (e.g., "text-embedding-3-small")
        system: The GenAI system (default: "openai")
        encoding_format: Format of the embeddings (e.g., "float")
        dimensions: Number of dimensions in the embeddings
        additional_attributes: Additional span attributes

    Returns:
        Context manager for the GenAI embeddings span
    """
    embeddings_attributes = {}

    if encoding_format:
        embeddings_attributes["gen_ai.request.encoding_format"] = encoding_format
    if dimensions:
        embeddings_attributes["gen_ai.request.dimensions"] = str(dimensions)

    if additional_attributes:
        embeddings_attributes.update(additional_attributes)

    return create_genai_span(
        operation_name=GenAI.Values.OperationName.EMBEDDINGS,
        agent_name="embeddings_agent",
        model=model,
        system=system,
        additional_attributes=embeddings_attributes,
    )


def setAgentInvocationSpan(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    system: str = GenAI.Values.System.PHARIA_AI,
    conversation_id: Optional[str] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function for agent invocation operations.

    Args:
        agent_id: Agent identifier (default: AGENTIC_CHAT)
        agent_name: Display name of the agent
        model: The model being used
        system: The GenAI system (default: "pharia_ai")
        conversation_id: Unique conversation identifier
        additional_attributes: Additional span attributes

    Returns:
        Context manager for the GenAI agent invocation span
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.INVOKE_AGENT,
        agent_name=agent_name or "invoked_agent",
        model=model,
        system=system,
        conversation_id=conversation_id,
        agent_id=agent_id,
        additional_attributes=additional_attributes,
    )


# =============================================================================
# GenAI Span Attribute Functions
# =============================================================================


def set_genai_span_usage(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    """
    Set token usage attributes on the current GenAI span.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens (computed automatically if not provided)

    Example:
        ```python
        with setChatSpan(model="gpt-4"):
            response = call_ai_model()
            set_genai_span_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        ```
    """
    if not OTEL_AVAILABLE:
        return

    from opentelemetry import trace

    span = trace.get_current_span()

    if span and span.is_recording():
        if input_tokens is not None:
            span.set_attribute(GenAI.USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens is not None:
            span.set_attribute(GenAI.USAGE_OUTPUT_TOKENS, output_tokens)

        # Calculate total if not provided
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

        if total_tokens is not None:
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)


def set_genai_span_response(
    response_id: Optional[str] = None,
    finish_reasons: Optional[List[str]] = None,
    model: Optional[str] = None,
    system_fingerprint: Optional[str] = None,
) -> None:
    """
    Set response attributes on the current GenAI span.

    Args:
        response_id: Unique identifier for the response
        finish_reasons: List of finish reasons (e.g., ["stop", "length"])
        model: The actual model used for the response
        system_fingerprint: System fingerprint (OpenAI-specific)

    Example:
        ```python
        with setChatSpan(model="gpt-4"):
            response = call_ai_model()
            set_genai_span_response(
                response_id=response.id,
                finish_reasons=["stop"],
                model=response.model
            )
        ```
    """
    if not OTEL_AVAILABLE:
        return

    from opentelemetry import trace

    span = trace.get_current_span()

    if span and span.is_recording():
        if response_id:
            span.set_attribute(GenAI.RESPONSE_ID, response_id)
        if finish_reasons:
            span.set_attribute(GenAI.RESPONSE_FINISH_REASONS, finish_reasons)
        if model:
            span.set_attribute(GenAI.RESPONSE_MODEL, model)
        if system_fingerprint:
            span.set_attribute(
                GenAI.OPENAI_RESPONSE_SYSTEM_FINGERPRINT, system_fingerprint
            )


# =============================================================================
# Context and Logging Integration
# =============================================================================


def add_context_to_logs(context: Dict[str, str]) -> Dict[str, str]:
    """
    Add context from baggage to structured logging.

    Args:
        context: Dictionary of context key-value pairs to add

    Returns:
        Enhanced context dictionary with injected values
    """
    if not OTEL_AVAILABLE:
        return context

    try:
        from pharia_telemetry.logging import create_context_injector

        injector = create_context_injector()
        return injector.inject(context)
    except Exception as e:
        logger.error("Failed to add context to logs: %s", e)
        return context


def get_current_context() -> Dict[str, str]:
    """
    Get the current OpenTelemetry context as a dictionary.

    Returns:
        Dictionary containing current baggage and trace context
    """
    if not OTEL_AVAILABLE:
        return {}

    try:
        from pharia_telemetry.baggage import get_all_baggage

        return get_all_baggage() or {}
    except Exception as e:
        logger.error("Failed to get current context: %s", e)
        return {}


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Alias for consistency with TypeScript version
setGenAiSpan = create_genai_span

# =============================================================================
# Export Constants for API Convenience
# =============================================================================

# Re-export GenAI constants for easy access
__all__ = [
    # Main functions
    "setup_telemetry",
    "create_genai_span",
    "setChatSpan",
    "setToolExecutionSpan",
    "setAgentCreationSpan",
    "setEmbeddingsSpan",
    "setAgentInvocationSpan",
    "set_genai_span_usage",
    "set_genai_span_response",
    "add_context_to_logs",
    "get_current_context",
    # Backwards compatibility
    "setGenAiSpan",
    # Constants
    "GenAI",
]
