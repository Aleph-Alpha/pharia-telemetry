"""
GenAI convenience functions for OpenTelemetry semantic conventions.

This module provides simple, clean functions for creating GenAI spans following
OpenTelemetry semantic conventions. Provides convenience functions with sensible defaults.

NOTE: If your application uses Pydantic Logfire, these GenAI utilities are generally not needed.
Pydantic Logfire provides built-in support for AI/LLM operations with automatic instrumentation
and span creation. These are primarily for applications that don't use Pydantic Logfire.

Example usage (for non-Pydantic Logfire applications):
    ```python
    from pharia_telemetry.gen_ai import create_chat_span, DataContext, GenAI

    # Simple chat span with data context
    data_context = DataContext(
        collections=["docs", "knowledge_base"],
        dataset_ids=["training_data", "eval_data"],
        namespaces=["pharia", "public"],
        indexes=["vector_index", "text_index"]
    )

    with create_chat_span(
        conversation_id="conv-123",
        model="gpt-4",
        data_context=data_context,
        extra_attributes={"custom_field": "value"}
    ) as span:
        # Your AI operation here
        response = call_ai_model()
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
    ```

Based on:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

from opentelemetry.trace import NonRecordingSpan, Span, SpanKind
from opentelemetry.trace.span import INVALID_SPAN_CONTEXT

from pharia_telemetry.sem_conv.gen_ai import GenAI
from pharia_telemetry.setup.setup import get_tracer

logger = logging.getLogger(__name__)

# Type variables for generic functions
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


@dataclass
class DataContext:
    """
    Data context for GenAI operations containing information about collections,
    datasets, namespaces, and indexes that the operation operates on.
    """

    collections: Optional[List[str]] = None
    dataset_ids: Optional[List[str]] = None
    namespaces: Optional[List[str]] = None
    indexes: Optional[List[str]] = None

    def to_attributes(self) -> Dict[str, Any]:
        """Convert DataContext to OpenTelemetry span attributes."""
        attributes = {}

        if self.collections:
            attributes["pharia.data.collections"] = self.collections
        if self.dataset_ids:
            attributes["pharia.data.dataset.ids"] = self.dataset_ids
        if self.namespaces:
            attributes["pharia.data.namespaces"] = self.namespaces
        if self.indexes:
            attributes["pharia.data.indexes"] = self.indexes

        return attributes


# =============================================================================
# Utility Functions
# =============================================================================


def _is_async_context() -> bool:
    """
    Detect if we're running in an async context.

    Returns:
        bool: True if running in an async context, False otherwise
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        return loop is not None
    except RuntimeError:
        # No event loop running
        return False


def _build_span_name_and_attributes(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    """
    Build span name and attributes according to OpenTelemetry GenAI semantic conventions.

    This shared logic is used by both sync and async span creation functions.

    Returns:
        Tuple of (span_name, attributes_dict)
    """
    # Build span name according to OpenTelemetry GenAI semantic conventions
    if operation_name == GenAI.Values.OperationName.CREATE_AGENT:
        span_name = f"create_agent {agent_name or agent_id or 'unknown'}"
    elif operation_name == GenAI.Values.OperationName.INVOKE_AGENT:
        span_name = f"invoke_agent {agent_name or agent_id or 'unknown'}"
    elif operation_name == GenAI.Values.OperationName.EXECUTE_TOOL:
        span_name = f"execute_tool {tool_name or 'unknown'}"
    else:
        # For inference operations: "{operation} {model}"
        span_name = f"{operation_name} {model or 'unknown'}"

    # Build core attributes
    attributes = {
        GenAI.OPERATION_NAME: operation_name,
    }

    # Add essential attributes if provided
    if agent_id:
        attributes[GenAI.AGENT_ID] = agent_id
    if agent_name:
        attributes[GenAI.AGENT_NAME] = agent_name
    if model:
        attributes[GenAI.REQUEST_MODEL] = model
    if conversation_id:
        attributes[GenAI.CONVERSATION_ID] = conversation_id
    if tool_name:
        attributes[GenAI.TOOL_NAME] = tool_name

    # Add data context attributes
    if data_context:
        attributes.update(data_context.to_attributes())

    # Add any additional attributes for flexibility
    if additional_attributes:
        attributes.update(additional_attributes)

    return span_name, attributes


# =============================================================================
# Core GenAI Span Creation
# =============================================================================


@contextmanager
def create_genai_span_sync(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    span_kind: Optional[SpanKind] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """
    Create a GenAI span following OpenTelemetry semantic conventions.

    Args:
        operation_name: The type of GenAI operation (use GenAI.Values.OperationName constants)
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        model: The model being used (e.g., "gpt-4", "claude-3")
        conversation_id: Unique conversation identifier
        tool_name: Name of the tool (for tool operations)
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        span_kind: OpenTelemetry span kind (default: CLIENT)
        additional_attributes: Additional span attributes for flexibility

    Yields:
        Span: The OpenTelemetry span or no-op span

    Example:
        ```python
        with create_genai_span(
            GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            model="gpt-4",
            conversation_id="conv-123"
        ) as span:
            # Your GenAI operation here
            response = call_ai_model()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```
    """
    tracer = get_tracer()
    if not tracer:
        # Create a proper NonRecordingSpan with context when no tracer is available
        yield NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return

    # Set default span kind if not provided
    if span_kind is None:
        span_kind = SpanKind.CLIENT

    # Use shared logic to build span name and attributes
    span_name, attributes = _build_span_name_and_attributes(
        operation_name=operation_name,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        tool_name=tool_name,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )

    # Create and yield the span
    with tracer.start_as_current_span(
        span_name, kind=span_kind, attributes=attributes
    ) as span:
        yield span


@asynccontextmanager
async def create_genai_span_async(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    span_kind: Optional[SpanKind] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Span, None]:
    """
    Async version of create_genai_span following OpenTelemetry semantic conventions.

    Args:
        operation_name: The type of GenAI operation (use GenAI.Values.OperationName constants)
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        model: The model being used (e.g., "gpt-4", "claude-3")
        conversation_id: Unique conversation identifier
        tool_name: Name of the tool (for tool operations)
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        span_kind: OpenTelemetry span kind (default: CLIENT)
        additional_attributes: Additional span attributes for flexibility

    Yields:
        Span: The OpenTelemetry span or no-op span

    Example:
        ```python
        async with acreate_genai_span(
            GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            model="gpt-4",
            conversation_id="conv-123"
        ) as span:
            # Your async GenAI operation here
            response = await call_ai_model_async()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```
    """
    tracer = get_tracer()
    if not tracer:
        # Create a proper NonRecordingSpan with context when no tracer is available
        yield NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return

    # Set default span kind if not provided
    if span_kind is None:
        span_kind = SpanKind.CLIENT

    # Use shared logic to build span name and attributes
    span_name, attributes = _build_span_name_and_attributes(
        operation_name=operation_name,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        tool_name=tool_name,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )

    # Create and yield the span
    with tracer.start_as_current_span(
        span_name, kind=span_kind, attributes=attributes
    ) as span:
        yield span


def create_genai_span(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    span_kind: Optional[SpanKind] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create a GenAI span that automatically detects sync/async context.

    This is the default GenAI span creation function. It automatically detects whether
    it's being called from a synchronous or asynchronous context and returns the
    appropriate context manager.

    Args:
        operation_name: The type of GenAI operation (use GenAI.Values.OperationName constants)
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        model: The model being used (e.g., "gpt-4", "claude-3")
        conversation_id: Unique conversation identifier
        tool_name: Name of the tool (for tool operations)
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        span_kind: OpenTelemetry span kind (default: CLIENT)
        additional_attributes: Additional span attributes for flexibility

    Returns:
        Either a sync or async context manager for the GenAI span (auto-detected)

    Example:
        ```python
        # Works in synchronous context
        with create_genai_span(
            GenAI.Values.OperationName.CHAT,
            model="gpt-4"
        ) as span:
            response = call_ai_model()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```

        ```python
        # Also works in asynchronous context
        async with create_genai_span(
            GenAI.Values.OperationName.CHAT,
            model="gpt-4"
        ) as span:
            response = await call_ai_model_async()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```
    """
    # Detect if we're in an async context
    if _is_async_context():
        return create_genai_span_async(
            operation_name=operation_name,
            agent_id=agent_id,
            agent_name=agent_name,
            model=model,
            conversation_id=conversation_id,
            tool_name=tool_name,
            data_context=data_context,
            span_kind=span_kind,
            additional_attributes=additional_attributes,
        )
    else:
        return create_genai_span_sync(
            operation_name=operation_name,
            agent_id=agent_id,
            agent_name=agent_name,
            model=model,
            conversation_id=conversation_id,
            tool_name=tool_name,
            data_context=data_context,
            span_kind=span_kind,
            additional_attributes=additional_attributes,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_chat_span(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create a chat span with sensible defaults that auto-detects sync/async context.

    Args:
        agent_id: Agent identifier (default: QA_CHAT)
        agent_name: Display name of the agent
        model: The model being used
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Either a sync or async context manager for the GenAI chat span (auto-detected)

    Example:
        ```python
        # Works in both sync and async contexts
        with create_chat_span(
            conversation_id="conv-123",
            model="gpt-4",
            data_context=DataContext(collections=["knowledge_base"])
        ) as span:
            # Your chat operation here
            response = call_ai_model()  # or await call_ai_model_async()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.CHAT,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_embeddings_span(
    *,
    model: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create an embeddings span that auto-detects sync/async context.

    Args:
        model: The model being used (e.g., "text-embedding-3-small")
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Either a sync or async context manager for the GenAI embeddings span (auto-detected)

    Example:
        ```python
        # Works in both sync and async contexts
        with create_embeddings_span(
            model="luminous-embed",
            data_context=DataContext(collections=["documents"], indexes=["vector_index"])
        ) as span:
            # Your embeddings operation here
            embeddings = get_embeddings(text)  # or await get_embeddings_async(text)
            span.set_attribute("gen_ai.usage.input_tokens", 50)
        ```
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.EMBEDDINGS,
        agent_name="embeddings_agent",
        model=model,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_tool_execution_span(
    tool_name: str,
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create a tool execution span with sensible defaults that auto-detects sync/async context.

    Args:
        tool_name: Name of the tool being executed (required)
        agent_id: Agent identifier (default: QA_CHAT)
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Either a sync or async context manager for the GenAI tool execution span (auto-detected)

    Example:
        ```python
        # Works in both sync and async contexts
        with create_tool_execution_span(
            "web_search",
            conversation_id="conv-123",
            data_context=DataContext(collections=["web_results"], namespaces=["search"])
        ) as span:
            # Your tool execution here
            result = execute_tool(tool_name, args)  # or await execute_tool_async(tool_name, args)
            span.set_attribute("tool.call_id", "call_123")
        ```
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
        tool_name=tool_name,
        agent_id=agent_id,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_agent_creation_span(
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create an agent creation span that auto-detects sync/async context.

    Args:
        agent_id: Unique identifier for the agent
        agent_name: Name of the agent being created
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Either a sync or async context manager for the GenAI agent creation span (auto-detected)

    Example:
        ```python
        # Works in both sync and async contexts
        with create_agent_creation_span(
            agent_name="Customer Support Agent",
            agent_id="new_agent_123"
        ) as span:
            # Your agent creation here
            agent = create_agent(config)  # or await create_agent_async(config)
            span.set_attribute("gen_ai.agent.type", "support")
        ```
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.CREATE_AGENT,
        agent_id=agent_id or GenAI.Values.PhariaAgentId.AGENT_CREATION,
        agent_name=agent_name,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_agent_invocation_span(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Union[ContextManager[Span], AsyncContextManager[Span]]:
    """
    Create an agent invocation span with sensible defaults that auto-detects sync/async context.

    Args:
        agent_id: Agent identifier (default: AGENTIC_CHAT)
        agent_name: Display name of the agent
        model: The model being used
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Either a sync or async context manager for the GenAI agent invocation span (auto-detected)

    Example:
        ```python
        # Works in both sync and async contexts
        with create_agent_invocation_span(
            agent_name="QA Assistant",
            model="luminous-supreme",
            conversation_id="conv-123"
        ) as span:
            # Your agent invocation here
            response = agent.process(message)  # or await agent.process_async(message)
            span.set_attribute("gen_ai.usage.input_tokens", 100)
        ```
    """
    return create_genai_span(
        operation_name=GenAI.Values.OperationName.INVOKE_AGENT,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


# =============================================================================
# Async Convenience Functions
# =============================================================================


def acreate_chat_span(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncContextManager[Span]:
    """
    Create an async chat span with sensible defaults.

    Args:
        agent_id: Agent identifier (default: QA_CHAT)
        agent_name: Display name of the agent
        model: The model being used
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Async context manager for the GenAI chat span

    Example:
        ```python
        async with acreate_chat_span(
            conversation_id="conv-123",
            model="gpt-4",
            data_context=DataContext(collections=["knowledge_base"])
        ) as span:
            # Your async chat operation here
            response = await call_ai_model_async()
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        ```
    """
    return create_genai_span_async(
        operation_name=GenAI.Values.OperationName.CHAT,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def acreate_embeddings_span(
    *,
    model: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncContextManager[Span]:
    """
    Create an async embeddings span.

    Args:
        model: The model being used (e.g., "text-embedding-3-small")
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Async context manager for the GenAI embeddings span

    Example:
        ```python
        async with acreate_embeddings_span(
            model="luminous-embed",
            data_context=DataContext(collections=["documents"], indexes=["vector_index"])
        ) as span:
            # Your async embeddings operation here
            embeddings = await get_embeddings_async(text)
            span.set_attribute("gen_ai.usage.input_tokens", 50)
        ```
    """
    return create_genai_span_async(
        operation_name=GenAI.Values.OperationName.EMBEDDINGS,
        agent_name="embeddings_agent",
        model=model,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def acreate_tool_execution_span(
    tool_name: str,
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncContextManager[Span]:
    """
    Create an async tool execution span with sensible defaults.

    Args:
        tool_name: Name of the tool being executed (required)
        agent_id: Agent identifier (default: QA_CHAT)
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Async context manager for the GenAI tool execution span

    Example:
        ```python
        async with acreate_tool_execution_span(
            "web_search",
            conversation_id="conv-123",
            data_context=DataContext(collections=["web_results"], namespaces=["search"])
        ) as span:
            # Your async tool execution here
            result = await execute_tool_async(tool_name, args)
            span.set_attribute("tool.call_id", "call_123")
        ```
    """
    return create_genai_span_async(
        operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
        tool_name=tool_name,
        agent_id=agent_id,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def acreate_agent_creation_span(
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncContextManager[Span]:
    """
    Create an async agent creation span.

    Args:
        agent_id: Unique identifier for the agent
        agent_name: Name of the agent being created
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Async context manager for the GenAI agent creation span

    Example:
        ```python
        async with acreate_agent_creation_span(
            agent_name="Customer Support Agent",
            agent_id="new_agent_123"
        ) as span:
            # Your async agent creation here
            agent = await create_agent_async(config)
            span.set_attribute("gen_ai.agent.type", "support")
        ```
    """
    return create_genai_span_async(
        operation_name=GenAI.Values.OperationName.CREATE_AGENT,
        agent_id=agent_id or GenAI.Values.PhariaAgentId.AGENT_CREATION,
        agent_name=agent_name,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def acreate_agent_invocation_span(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncContextManager[Span]:
    """
    Create an async agent invocation span with sensible defaults.

    Args:
        agent_id: Agent identifier (default: AGENTIC_CHAT)
        agent_name: Display name of the agent
        model: The model being used
        conversation_id: Unique conversation identifier
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        additional_attributes: Additional span attributes

    Returns:
        Async context manager for the GenAI agent invocation span

    Example:
        ```python
        async with acreate_agent_invocation_span(
            agent_name="QA Assistant",
            model="luminous-supreme",
            conversation_id="conv-123"
        ) as span:
            # Your async agent invocation here
            response = await agent.process_async(message)
            span.set_attribute("gen_ai.usage.input_tokens", 100)
        ```
    """
    return create_genai_span_async(
        operation_name=GenAI.Values.OperationName.INVOKE_AGENT,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


# =============================================================================
# Decorators for Function Instrumentation
# =============================================================================


def genai_span(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    span_kind: Optional[SpanKind] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically instrument synchronous functions with GenAI spans.

    Args:
        operation_name: The type of GenAI operation (use GenAI.Values.OperationName constants)
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        model: The model being used (e.g., "gpt-4", "claude-3")
        conversation_id: Unique conversation identifier
        tool_name: Name of the tool (for tool operations)
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        span_kind: OpenTelemetry span kind (default: CLIENT)
        additional_attributes: Additional span attributes for flexibility

    Returns:
        Decorated function with automatic GenAI span instrumentation

    Example:
        ```python
        @genai_span(
            GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            model="gpt-4"
        )
        def chat_with_ai(message: str) -> str:
            response = call_ai_model(message)
            set_genai_span_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return response.content
        ```
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with create_genai_span_sync(
                operation_name=operation_name,
                agent_id=agent_id,
                agent_name=agent_name,
                model=model,
                conversation_id=conversation_id,
                tool_name=tool_name,
                data_context=data_context,
                span_kind=span_kind,
                additional_attributes=additional_attributes,
            ):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def agenai_span(
    operation_name: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data_context: Optional[DataContext] = None,
    span_kind: Optional[SpanKind] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to automatically instrument async functions with GenAI spans.

    Args:
        operation_name: The type of GenAI operation (use GenAI.Values.OperationName constants)
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        model: The model being used (e.g., "gpt-4", "claude-3")
        conversation_id: Unique conversation identifier
        tool_name: Name of the tool (for tool operations)
        data_context: DataContext containing collections, datasets, namespaces, and indexes
        span_kind: OpenTelemetry span kind (default: CLIENT)
        additional_attributes: Additional span attributes for flexibility

    Returns:
        Decorated async function with automatic GenAI span instrumentation

    Example:
        ```python
        @agenai_span(
            GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            model="gpt-4"
        )
        async def chat_with_ai_async(message: str) -> str:
            response = await call_ai_model_async(message)
            set_genai_span_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return response.content
        ```
    """

    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with create_genai_span_async(
                operation_name=operation_name,
                agent_id=agent_id,
                agent_name=agent_name,
                model=model,
                conversation_id=conversation_id,
                tool_name=tool_name,
                data_context=data_context,
                span_kind=span_kind,
                additional_attributes=additional_attributes,
            ):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Span Attribute Setting Functions
# =============================================================================


def set_genai_span_usage(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    """
    Set usage information on the current GenAI span.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens used (auto-calculated if not provided)

    Example:
        ```python
        with create_chat_span(model="gpt-4") as span:
            response = call_ai_model()
            set_genai_span_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        ```
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return

        if input_tokens is not None:
            span.set_attribute(GenAI.USAGE_INPUT_TOKENS, input_tokens)

        if output_tokens is not None:
            span.set_attribute(GenAI.USAGE_OUTPUT_TOKENS, output_tokens)

        # Auto-calculate total if not provided but both input and output are available
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

        if total_tokens is not None:
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

    except Exception as e:
        logger.warning("Failed to set GenAI span usage: %s", e)


def set_genai_span_response(
    response_id: Optional[str] = None,
    model: Optional[str] = None,
    finish_reasons: Optional[List[str]] = None,
    system_fingerprint: Optional[str] = None,
) -> None:
    """
    Set response information on the current GenAI span.

    Args:
        response_id: Unique response identifier
        model: Model used for the response
        finish_reasons: Reasons why the generation finished
        system_fingerprint: System fingerprint for reproducibility

    Example:
        ```python
        with create_chat_span(model="gpt-4") as span:
            response = call_ai_model()
            set_genai_span_response(
                response_id=response.id,
                model=response.model,
                finish_reasons=response.choices[0].finish_reason,
                system_fingerprint=response.system_fingerprint,
            )
        ```
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return

        if response_id is not None:
            span.set_attribute(GenAI.RESPONSE_ID, response_id)

        if model is not None:
            span.set_attribute(GenAI.RESPONSE_MODEL, model)

        if finish_reasons is not None:
            span.set_attribute(GenAI.RESPONSE_FINISH_REASONS, finish_reasons)

        if system_fingerprint is not None:
            span.set_attribute(
                GenAI.OPENAI_RESPONSE_SYSTEM_FINGERPRINT, system_fingerprint
            )

    except Exception as e:
        logger.warning("Failed to set GenAI span response: %s", e)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core functions (smart by default)
    "create_genai_span",
    # Explicit sync/async functions (for advanced use)
    "create_genai_span_sync",
    "create_genai_span_async",
    # Convenience span functions (smart by default)
    "create_chat_span",
    "create_embeddings_span",
    "create_tool_execution_span",
    "create_agent_creation_span",
    "create_agent_invocation_span",
    # Async convenience span functions (explicit async)
    "acreate_chat_span",
    "acreate_embeddings_span",
    "acreate_tool_execution_span",
    "acreate_agent_creation_span",
    "acreate_agent_invocation_span",
    # Decorators
    "genai_span",
    "agenai_span",
    # Span attribute setting functions
    "set_genai_span_usage",
    "set_genai_span_response",
    # Data structures
    "DataContext",
    # Constants
    "GenAI",
]
