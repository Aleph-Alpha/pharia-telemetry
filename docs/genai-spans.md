# GenAI Spans Guide

This guide explains how to use the pharia-telemetry library to create GenAI spans following OpenTelemetry semantic conventions.

## Overview

The pharia-telemetry library provides convenient functions for creating GenAI spans that follow the [OpenTelemetry Semantic Conventions for GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/). These functions automatically set the correct attributes and span names according to the specification.

## Quick Start

```python
from pharia_telemetry import (
    setup_telemetry,
    setChatSpan,
    set_genai_span_usage,
    set_genai_span_response,
)
from pharia_telemetry.sem_conv.gen_ai import GenAI

# Setup telemetry
setup_telemetry("my-service")

# Create a chat completion span
with setChatSpan(
    model="llama-3.1-8B",
    system=GenAI.Values.System.PHARIAAI,
    conversation_id="conv_12345",
) as span:
    if span:
        # Your GenAI operation here
        response = phariaai_client.chat.completions.create(...)

        # Set usage information
        set_genai_span_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        # Set response information
        set_genai_span_response(
            response_id=response.id,
            model=response.model,
            finish_reasons=[choice.finish_reason for choice in response.choices],
        )
```

## Core Functions

### General GenAI Functions

#### `create_genai_span()`

Creates a GenAI span with proper semantic conventions.

**Parameters:**
- `operation_name` (str): The GenAI operation being performed (e.g., "chat", "embeddings", "execute_tool")
- `model` (str, optional): The GenAI model name (e.g., "llama-3.1-8B", "llama-3.1-8B-instruct")
- `system` (str, optional): The GenAI system (default: "phariaai")
- `span_name` (str, optional): Custom span name (auto-generated if not provided)
- `conversation_id` (str, optional): Unique identifier for conversation/session tracking
- `agent_name` (str, optional): Human-readable name of the GenAI agent
- `agent_id` (str, optional): Unique identifier of the GenAI agent
- `additional_attributes` (dict, optional): Additional span attributes

**Returns:** Context manager yielding OpenTelemetry span object (or None if unavailable)

### Operation-Specific Convenience Functions

#### `setChatSpan()`

Creates a span specifically for chat operations.

```python
with setChatSpan(
    model="llama-3.1-8B",
    system=GenAI.Values.System.PHARIAAI,
    conversation_id="conv_123",
    agent_name="Customer Support Assistant"
) as span:
    # Chat completion logic
    pass
```

#### `setToolExecutionSpan()`

Creates a span for tool execution operations.

```python
with setToolExecutionSpan(
    tool_name="calculator",
    model="llama-3.1-8B",
    tool_description="Performs mathematical calculations",
    tool_call_id="call_abc123"
) as span:
    # Tool execution logic
    pass
```

#### `setAgentCreationSpan()`

Creates a span for agent creation operations.

```python
with setAgentCreationSpan(
    agent_name="Data Analysis Agent",
    agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
    model="llama-3.1-8B"
) as span:
    # Agent creation logic
    pass
```

#### `setEmbeddingsSpan()`

Creates a span for embeddings generation.

```python
with setEmbeddingsSpan(
    model="llama-3.1-8B-embeddings",
    system=GenAI.Values.System.PHARIAAI
) as span:
    # Embeddings generation logic
    pass
```

#### `setAgentInvocationSpan()`

Creates a span for agent invocation operations.

```python
with setAgentInvocationSpan(
    agent_name="Research Assistant",
    agent_id=GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
    model="llama-3.1-8B",
    conversation_id="conv_456"
) as span:
    # Agent invocation logic
    pass
```

### Helper Functions

#### `set_genai_span_usage()`

Sets token usage attributes on the current GenAI span.

**Parameters:**
- `input_tokens` (int, optional): Number of input tokens used
- `output_tokens` (int, optional): Number of output tokens generated
- `total_tokens` (int, optional): Total number of tokens used (auto-calculated if not provided)

#### `set_genai_span_response()`

Sets response attributes on the current GenAI span.

**Parameters:**
- `response_id` (str, optional): Unique identifier for the response
- `model` (str, optional): The actual model that generated the response
- `finish_reasons` (list[str], optional): Array of reasons the model stopped generating tokens
- `system_fingerprint` (str, optional): System fingerprint for the response

## Available Constants

The library provides constants for commonly used values:

### Operation Names

```python
# Available in GenAI.Values.OperationName
CHAT = "chat"
EMBEDDINGS = "embeddings"
EXECUTE_TOOL = "execute_tool"
INVOKE_AGENT = "invoke_agent"
CREATE_AGENT = "create_agent"
TEXT_COMPLETION = "text_completion"
GENERATE_CONTENT = "generate_content"
```

### System Identifiers

```python
# Available in GenAI.Values.System
PHARIAAI = "phariaai"
OPENAI = "openai"
ANTHROPIC = "anthropic"
AWS_BEDROCK = "aws.bedrock"
AZURE_AI_OPENAI = "azure.ai.openai"
GCP_GEMINI = "gcp.gemini"
HUGGING_FACE = "huggingface"
ALEPH_ALPHA = "aleph_alpha"
```

### Pharia Agent IDs

```python
# Available in GenAI.Values.PhariaAgentId
QA_CHAT = "pharia_qa_chat"
AGENTIC_CHAT = "pharia_agentic_chat"
TRANSLATION = "pharia_translation"
TRANSCRIPTION = "pharia_transcription"
DATA_ANALYSIS = "pharia_data_analysis"
RESEARCH_ASSISTANT = "pharia_research_assistant"
```

## Integration Examples

### PHARIA AI Integration

```python
import phariaai_client
from pharia_telemetry import setChatSpan, set_genai_span_usage, set_genai_span_response
from pharia_telemetry.sem_conv.gen_ai import GenAI

def chat_with_phariaai(messages, model="llama-3.1-8B"):
    with setChatSpan(
        model=model,
        system=GenAI.Values.System.PHARIAAI,
        conversation_id="conv_12345"
    ) as span:
        response = phariaai_client.chat.completions.create(
            model=model,
            messages=messages,
        )

        if span:
            # Set usage information
            set_genai_span_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            # Set response information
            set_genai_span_response(
                response_id=response.id,
                model=response.model,
                finish_reasons=[choice.finish_reason for choice in response.choices],
            )

        return response
```

### Agent Framework Integration

```python
from pharia_telemetry import setAgentInvocationSpan
from pharia_telemetry.sem_conv.gen_ai import GenAI

class MathTutorAgent:
    def invoke(self, query: str, conversation_id: str):
        with setAgentInvocationSpan(
            agent_name="Math Tutor",
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            model="llama-3.1-8B",
            conversation_id=conversation_id,
        ) as span:
            # Agent logic here
            result = self._process_query(query)

            if span:
                # Add any additional agent-specific attributes
                span.set_attribute("agent.query_type", "mathematical")
                span.set_attribute("agent.difficulty_level", "intermediate")

            return result
```

### Tool Execution Integration

```python
from pharia_telemetry import setToolExecutionSpan

class CalculatorTool:
    def execute(self, expression: str, call_id: str):
        with setToolExecutionSpan(
            tool_name="calculator",
            tool_description="Performs mathematical calculations",
            tool_call_id=call_id,
            model="llama-3.1-8B"
        ) as span:
            try:
                result = eval(expression)  # In production, use safe evaluation

                if span:
                    span.set_attribute("tool.expression", expression)
                    span.set_attribute("tool.result", str(result))
                    span.set_attribute("tool.success", True)

                return result

            except Exception as e:
                if span:
                    span.set_attribute("tool.error", str(e))
                    span.set_attribute("tool.success", False)
                raise
```

### Embeddings Integration

```python
from pharia_telemetry import setEmbeddingsSpan
from pharia_telemetry.sem_conv.gen_ai import GenAI

def generate_embeddings(texts: list[str]):
    with setEmbeddingsSpan(
        model="llama-3.1-8B-embeddings",
        system=GenAI.Values.System.PHARIAAI
    ) as span:
        embeddings = phariaai_client.embeddings.create(
            model="llama-3.1-8B-embeddings",
            input=texts
        )

        if span:
            span.set_attribute("embeddings.input_count", len(texts))
            span.set_attribute("embeddings.total_tokens", embeddings.usage.total_tokens)
            span.set_attribute("embeddings.dimensions", len(embeddings.data[0].embedding))

        return embeddings
```

## Error Handling

The GenAI convenience functions are designed to gracefully handle cases where OpenTelemetry is not available:

```python
with setChatSpan(model="llama-3.1-8B", system=GenAI.Values.System.PHARIAAI) as span:
    if span:
        # OpenTelemetry is available, span was created
        span.set_attribute("custom.attribute", "value")
    else:
        # OpenTelemetry is not available, span is None
        # Your code should continue normally
        pass

    # Your GenAI operation logic here (works regardless of span availability)
    result = perform_genai_operation()
    return result
```

## Best Practices

1. **Always check for span availability**: Use `if span:` before setting attributes
2. **Use constants**: Prefer using the provided constants over hardcoded strings
3. **Set usage information**: Always set token usage when available for cost tracking
4. **Include conversation IDs**: Use conversation IDs for proper session tracking
5. **Set response metadata**: Include response IDs and finish reasons for debugging
6. **Handle errors gracefully**: The functions are designed to not break your application if telemetry fails
7. **Use operation-specific functions**: Prefer `setChatSpan()` over `create_genai_span()` for chat operations

## Complete Example

See [examples/comprehensive_genai_example.py](../examples/comprehensive_genai_example.py) for a complete working example demonstrating all the GenAI span features.

## Module Information

The GenAI functionality is provided by the `pharia_telemetry.sem_conv.gen_ai` module, but all functions are re-exported through the main `pharia_telemetry` package for convenience.

## Legacy Function Migration

If you were using the old API, here's how to migrate:

### Before (Old API)
```python
from pharia_telemetry import create_genai_span
from pharia_telemetry.sem_conv.baggage import Spans

with create_genai_span(
    operation_name=Spans.Values.GenAiOperationName.CHAT,
    model="llama-3.1-8B",
    system=GenAI.Values.System.PHARIAAI
) as span:
    pass
```

### After (New API)
```python
from pharia_telemetry import setChatSpan
from pharia_telemetry.sem_conv.gen_ai import GenAI

with setChatSpan(
    model="llama-3.1-8B",
    system=GenAI.Values.System.PHARIAAI
) as span:
    pass
```
