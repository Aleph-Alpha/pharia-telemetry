"""
GenAI convenience functions for OpenTelemetry semantic conventions.

This module provides simple, clean functions for creating GenAI spans following
OpenTelemetry semantic conventions. Provides convenience functions with sensible defaults.

TYPE SAFETY: Functions like `create_genai_span()` use runtime detection to return either sync
or async context managers. While functional, this can cause type checking issues. For better
type safety, use the explicit sync/async versions:
- Use `create_genai_span_sync()` in synchronous contexts
- Use `create_genai_span_async()` in asynchronous contexts

Example usage:
    ```python
    from pharia_telemetry.sem_conv.gen_ai import create_chat_span, DataContext, GenAI

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
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    TypeVar,
)

from opentelemetry.trace import NonRecordingSpan, Span, SpanKind
from opentelemetry.trace.span import INVALID_SPAN_CONTEXT

from pharia_telemetry.setup.setup import get_tracer

logger = logging.getLogger(__name__)


# GenAI semantic convention constants
# Based on https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

# Core GenAI attributes
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

# OpenAI specific attributes
GEN_AI_OPENAI_REQUEST_SERVICE_TIER = "gen_ai.openai.request.service_tier"
GEN_AI_OPENAI_RESPONSE_SERVICE_TIER = "gen_ai.openai.response.service_tier"
GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "gen_ai.openai.response.system_fingerprint"

# Operation and system attributes
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"

# Request attributes
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"

# Response attributes
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

# Output and token attributes
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"

# Tool attributes
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# Usage and token attributes
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

try:
    # Check if OpenTelemetry is available
    import opentelemetry  # noqa: F401
    from opentelemetry.trace import SpanKind

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - GenAI functions will be limited")


# Type variables for generic functions
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


@dataclass
class DataContext:
    """
    Data context for GenAI operations containing information about collections,
    datasets, namespaces, and indexes that the operation operates on.
    """

    collections: list[str] | None = None
    dataset_ids: list[str] | None = None
    namespaces: list[str] | None = None
    indexes: list[str] | None = None

    def to_attributes(self) -> dict[str, Any]:
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
    Detect whether we are executing inside an actual asyncio Task.

    This is intentionally conservative: some environments (e.g., pytest with
    asyncio strict mode) may have a running loop in the thread even when the
    current call site is synchronous. We therefore only treat the context as
    async when a current Task exists.
    """
    try:
        return asyncio.current_task() is not None
    except RuntimeError:
        return False


def _build_span_name_and_attributes(
    operation_name: str,
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
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
# OpenTelemetry GenAI Constants
# =============================================================================


class GenAI:
    """
    OpenTelemetry GenAI semantic convention attribute keys.

    These constants provide standardized attribute keys for GenAI operations
    following the OpenTelemetry semantic conventions specification.
    """

    # =========================================================================
    # Core GenAI Attributes
    # =========================================================================

    # Agent attributes
    AGENT_ID: str = GEN_AI_AGENT_ID
    AGENT_NAME: str = GEN_AI_AGENT_NAME

    # Conversation attributes
    CONVERSATION_ID: str = GEN_AI_CONVERSATION_ID

    # Operation attributes
    OPERATION_NAME: str = GEN_AI_OPERATION_NAME
    SYSTEM: str = GEN_AI_SYSTEM

    # Model attributes
    REQUEST_MODEL: str = GEN_AI_REQUEST_MODEL
    RESPONSE_MODEL: str = GEN_AI_RESPONSE_MODEL

    # =========================================================================
    # Request Attributes
    # =========================================================================

    REQUEST_TEMPERATURE: str = GEN_AI_REQUEST_TEMPERATURE
    REQUEST_TOP_P: str = GEN_AI_REQUEST_TOP_P
    REQUEST_MAX_TOKENS: str = GEN_AI_REQUEST_MAX_TOKENS
    REQUEST_SEED: str = GEN_AI_REQUEST_SEED

    # =========================================================================
    # Response Attributes
    # =========================================================================

    RESPONSE_ID: str = GEN_AI_RESPONSE_ID
    RESPONSE_FINISH_REASONS: str = GEN_AI_RESPONSE_FINISH_REASONS
    OUTPUT_TYPE: str = GEN_AI_OUTPUT_TYPE

    # =========================================================================
    # Usage and Token Attributes
    # =========================================================================

    USAGE_INPUT_TOKENS: str = GEN_AI_USAGE_INPUT_TOKENS
    USAGE_OUTPUT_TOKENS: str = GEN_AI_USAGE_OUTPUT_TOKENS
    TOKEN_TYPE: str = GEN_AI_TOKEN_TYPE

    # =========================================================================
    # Tool Attributes
    # =========================================================================

    TOOL_NAME: str = GEN_AI_TOOL_NAME
    TOOL_DESCRIPTION: str = GEN_AI_TOOL_DESCRIPTION
    TOOL_CALL_ID: str = GEN_AI_TOOL_CALL_ID

    # =========================================================================
    # OpenAI Specific Attributes
    # =========================================================================

    OPENAI_REQUEST_SERVICE_TIER: str = GEN_AI_OPENAI_REQUEST_SERVICE_TIER
    OPENAI_RESPONSE_SERVICE_TIER: str = GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT: str = GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT

    # =========================================================================
    # Standard Values
    # =========================================================================

    class Values:
        """Standard values for GenAI attributes following OpenTelemetry conventions."""

        class OperationName:
            """
            Standard GenAI operation names.

            Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
            """

            CHAT: str = "chat"
            GENERATE_CONTENT: str = "generate_content"
            EXECUTE_TOOL: str = "execute_tool"
            CREATE_AGENT: str = "create_agent"
            EMBEDDINGS: str = "embeddings"
            INVOKE_AGENT: str = "invoke_agent"
            TEXT_COMPLETION: str = "text_completion"
            IMAGE_GENERATION: str = "image_generation"
            AUDIO_GENERATION: str = "audio_generation"
            VIDEO_GENERATION: str = "video_generation"
            CODE_GENERATION: str = "code_generation"
            DATA_ANALYSIS: str = "data_analysis"
            DATA_VISUALIZATION: str = "data_visualization"

        class System:
            """
            Standard GenAI system identifiers.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            ANTHROPIC: str = "anthropic"
            AWS_BEDROCK: str = "aws.bedrock"
            AZURE_AI_INFERENCE: str = "azure.ai.inference"
            AZURE_AI_OPENAI: str = "azure.ai.openai"
            COHERE: str = "cohere"
            DEEPSEEK: str = "deepseek"
            GCP_GEMINI: str = "gcp.gemini"
            GCP_GEN_AI: str = "gcp.gen_ai"
            GCP_VERTEX_AI: str = "gcp.vertex_ai"
            GROQ: str = "groq"
            IBM_WATSONX_AI: str = "ibm.watsonx.ai"
            MISTRAL_AI: str = "mistral_ai"
            OPENAI: str = "openai"
            PERPLEXITY: str = "perplexity"
            XAI: str = "xai"

            # Pharia-specific systems
            PHARIA_AI: str = "pharia_ai"
            CUSTOM: str = "_OTHER"

        class OutputType:
            """
            Standard output type values.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            IMAGE: str = "image"
            JSON: str = "json"
            SPEECH: str = "speech"
            TEXT: str = "text"

        class TokenType:
            """
            Standard token type values.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            INPUT: str = "input"
            OUTPUT: str = "output"

        class PhariaAgentId:
            """Standard GenAI agent IDs for Pharia services."""

            QA_CHAT: str = "pharia_qa_chat"
            AGENTIC_CHAT: str = "pharia_agentic_chat"
            TRANSLATION: str = "pharia_translation"
            TRANSCRIPTION: str = "pharia_transcription"
            EASY_LANGUAGE: str = "pharia_easy_language"
            SIGN_LANGUAGE: str = "pharia_sign_language"
            FILE_UPLOAD: str = "pharia_file_upload"
            DOCUMENT_PROCESSING: str = "pharia_document_processing"
            AGENT_CREATION: str = "pharia_agent_creation"
            CUSTOM_AGENT: str = "pharia_custom_agent"


# =============================================================================
# Core GenAI Span Creation
# =============================================================================


@contextmanager
def create_genai_span_sync(
    operation_name: str,
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    span_kind: SpanKind | None = None,
    additional_attributes: dict[str, Any] | None = None,
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
    # If OpenTelemetry is not available (as detected in this module), return a no-op span
    if not OTEL_AVAILABLE:
        yield NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return

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
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    span_kind: SpanKind | None = None,
    additional_attributes: dict[str, Any] | None = None,
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
    # If OpenTelemetry is not available (as detected in this module), return a no-op span
    if not OTEL_AVAILABLE:
        yield NonRecordingSpan(INVALID_SPAN_CONTEXT)
        return

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
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    span_kind: SpanKind | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
    """
    Create a GenAI span that automatically detects sync/async context.

    This is the default GenAI span creation function. It automatically detects whether
    it's being called from a synchronous or asynchronous context and returns the
    appropriate context manager.

    **Type Safety Note**: While this function provides automatic context detection,
    type checkers may have difficulty with the Union return type. For better type safety,
    consider using the explicit sync/async versions:
    - `create_genai_span_sync()` for synchronous contexts
    - `create_genai_span_async()` for asynchronous contexts

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

        ```python
        # For better type safety, use explicit versions:
        with create_genai_span_sync(...) as span:  # Always returns ContextManager
            pass
        async with create_genai_span_async(...) as span:  # Always returns AsyncContextManager
            pass
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
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
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
    model: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
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
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
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
    agent_id: str | None = None,
    agent_name: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
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
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span] | AbstractAsyncContextManager[Span]:
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
# Sync Convenience Functions (type-safe for mypy)
# =============================================================================


def create_chat_span_sync(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span]:
    """
    Create a chat span (sync-only) with sensible defaults.

    This variant returns a non-Union sync context manager to keep static type
    checkers (e.g., mypy) happy when used inside a normal `with` block.
    """
    return create_genai_span_sync(
        operation_name=GenAI.Values.OperationName.CHAT,
        agent_id=agent_id,
        agent_name=agent_name,
        model=model,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_embeddings_span_sync(
    *,
    model: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span]:
    """
    Create an embeddings span (sync-only) with sensible defaults.
    """
    return create_genai_span_sync(
        operation_name=GenAI.Values.OperationName.EMBEDDINGS,
        agent_name="embeddings_agent",
        model=model,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_tool_execution_span_sync(
    tool_name: str,
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span]:
    """
    Create a tool execution span (sync-only) with sensible defaults.
    """
    return create_genai_span_sync(
        operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
        tool_name=tool_name,
        agent_id=agent_id,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_agent_creation_span_sync(
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span]:
    """
    Create an agent creation span (sync-only) with sensible defaults.
    """
    return create_genai_span_sync(
        operation_name=GenAI.Values.OperationName.CREATE_AGENT,
        agent_id=agent_id or GenAI.Values.PhariaAgentId.AGENT_CREATION,
        agent_name=agent_name,
        conversation_id=conversation_id,
        data_context=data_context,
        additional_attributes=additional_attributes,
    )


def create_agent_invocation_span_sync(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span]:
    """
    Create an agent invocation span (sync-only) with sensible defaults.
    """
    return create_genai_span_sync(
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


def create_chat_span_async(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Span]:
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
        async with create_chat_span_async(
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


def create_embeddings_span_async(
    *,
    model: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Span]:
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
        async with create_embeddings_span_async(
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


def create_tool_execution_span_async(
    tool_name: str,
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.QA_CHAT,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Span]:
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
        async with create_tool_execution_span_async(
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


def create_agent_creation_span_async(
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Span]:
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
        async with create_agent_creation_span_async(
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


def create_agent_invocation_span_async(
    *,
    agent_id: str = GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    data_context: DataContext | None = None,
    additional_attributes: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Span]:
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
        async with create_agent_invocation_span_async(
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
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    span_kind: SpanKind | None = None,
    additional_attributes: dict[str, Any] | None = None,
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
    agent_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
    tool_name: str | None = None,
    data_context: DataContext | None = None,
    span_kind: SpanKind | None = None,
    additional_attributes: dict[str, Any] | None = None,
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
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
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
    response_id: str | None = None,
    model: str | None = None,
    finish_reasons: list[str] | None = None,
    system_fingerprint: str | None = None,
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
    # Sync convenience span functions (explicit sync)
    "create_chat_span_sync",
    "create_embeddings_span_sync",
    "create_tool_execution_span_sync",
    "create_agent_creation_span_sync",
    "create_agent_invocation_span_sync",
    # Async convenience span functions (explicit async)
    "create_chat_span_async",
    "create_embeddings_span_async",
    "create_tool_execution_span_async",
    "create_agent_creation_span_async",
    "create_agent_invocation_span_async",
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
