# Manual Instrumentation Guide

This guide covers when and how to use manual instrumentation with `pharia-telemetry` for scenarios where auto-instrumentation doesn't work well.

## When to Use Manual Instrumentation

| Scenario | Why Manual? | Solution |
|----------|-------------|----------|
| **SSE Streaming** | Context loss during streaming | Manual span management during streaming lifecycle |
| **HTTP/2 Services** | Limited auto-instrumentation support | Manual middleware-based tracing |
| **High Performance** | Reduce auto-instrumentation overhead | Selective manual tracing of critical paths |
| **Custom Logic** | Domain-specific span attributes | Custom spans with business context |
| **Complex Workflows** | Multi-step process tracing | Manual span hierarchy management |

## Foundation Setup

Always start with the `pharia-telemetry` foundation:

```python
from pharia_telemetry import setup_telemetry

# Foundation provides propagators, baggage processor, exporters
setup_telemetry("my-service", service_version="1.0.0")

# Then add manual instrumentation as needed
```

## Server-Sent Events (SSE) Streaming

### The Problem

Auto-instrumentation breaks with SSE streaming:

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)  # ❌ Causes context loss

@app.get("/stream")
async def stream_chat():
    async def generate():
        for chunk in chat_chunks:
            yield f"data: {chunk}\n\n"  # Context lost here!

    return StreamingResponse(generate(), media_type="text/plain")
```

**Issues:**
- OpenTelemetry context detached too early
- Spans closed before streaming completes
- Performance overhead on each chunk

### The Solution: Manual Span Management

```python
from pharia_telemetry import constants, set_baggage_item
from pharia_telemetry.utils import get_tracer
from fastapi.responses import StreamingResponse

tracer = get_tracer(__name__)

@app.get("/stream/{user_id}")
async def stream_chat(user_id: str):
    # Set context early
    set_baggage_item(constants.Baggage.USER_ID, user_id)
    set_baggage_item(constants.Baggage.USER_INTENT, "chat_streaming")

    with tracer.start_span("sse_chat_stream") as span:
        # Add custom attributes
        span.set_attribute("stream.type", "chat")
        span.set_attribute("stream.user_id", user_id)

        async def generate():
            chunk_count = 0
            try:
                async for chunk in chat_service.stream_response(user_id):
                    chunk_count += 1

                    # Add events for debugging (sparingly - performance impact)
                    if chunk_count % 10 == 0:  # Every 10th chunk
                        span.add_event("chunk_milestone", {
                            "chunk_count": chunk_count,
                            "timestamp": time.time()
                        })

                    yield f"data: {chunk}\n\n"

                # Mark successful completion
                span.set_attribute("stream.chunks_sent", chunk_count)
                span.set_status(trace.Status(trace.StatusCode.OK))

            except Exception as e:
                # Handle streaming errors
                span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    f"Streaming failed: {str(e)}"
                ))
                span.set_attribute("stream.error", str(e))
                span.set_attribute("stream.chunks_sent", chunk_count)
                raise

        return StreamingResponse(generate(), media_type="text/plain")
```

### Advanced SSE Pattern with Child Spans

```python
@app.get("/stream/{conversation_id}")
async def stream_ai_response(conversation_id: str):
    set_baggage_item(constants.Baggage.CHAT_QA_CONVERSATION_ID, conversation_id)

    with tracer.start_span("ai_streaming_response") as parent_span:
        parent_span.set_attribute("conversation.id", conversation_id)

        async def generate():
            # Create child span for initial processing
            with tracer.start_span("ai_query_processing", parent=parent_span) as query_span:
                query_span.set_attribute("ai.model", "llama-3.1-8B")
                prompt = await prepare_prompt(conversation_id)
                query_span.set_attribute("ai.prompt_length", len(prompt))

            # Create child span for streaming
            with tracer.start_span("ai_token_streaming", parent=parent_span) as stream_span:
                token_count = 0
                async for token in ai_model.stream(prompt):
                    token_count += 1
                    yield f"data: {token}\n\n"

                stream_span.set_attribute("ai.tokens_generated", token_count)

        return StreamingResponse(generate(), media_type="text/plain")
```

## HTTP/2 Services with Hypercorn

### The Problem

Auto-instrumentation has limited HTTP/2 support:

```python
# ❌ Limited HTTP/2 compatibility
FastAPIInstrumentor.instrument_app(app)

# Running with: hypercorn app:app --bind 0.0.0.0:8000 --http2
```

### The Solution: Manual Middleware

```python
from pharia_telemetry import constants, set_baggage_item
from pharia_telemetry.utils import get_tracer
from fastapi import Request, Response
import time

tracer = get_tracer(__name__)

@app.middleware("http")
async def manual_tracing_middleware(request: Request, call_next):
    # Extract user context from headers
    user_id = request.headers.get("x-user-id")
    session_id = request.headers.get("x-session-id")

    # Start span for HTTP request
    span_name = f"{request.method} {request.url.path}"

    with tracer.start_span(span_name) as span:
        # Set standard HTTP attributes
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.scheme", request.url.scheme)
        span.set_attribute("http.server_name", request.url.hostname)

        # Set user context in baggage (flows to downstream services)
        if user_id:
            set_baggage_item(constants.Baggage.USER_ID, user_id)
            span.set_attribute("app.user.id", user_id)

        if session_id:
            set_baggage_item(constants.Baggage.SESSION_ID, session_id)
            span.set_attribute("app.session.id", session_id)

        # Record start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Set response attributes
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.response_size", len(response.body) if hasattr(response, 'body') else 0)

            # Set status based on HTTP status code
            if response.status_code >= 400:
                span.set_status(trace.Status(trace.StatusCode.ERROR))

            return response

        except Exception as e:
            # Handle request errors
            span.set_status(trace.Status(
                trace.StatusCode.ERROR,
                f"Request failed: {str(e)}"
            ))
            span.set_attribute("error.message", str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise

        finally:
            # Record request duration
            duration = time.time() - start_time
            span.set_attribute("http.duration_ms", duration * 1000)
```

## Custom Span Management

### Basic Custom Spans

```python
from pharia_telemetry import get_tracer, set_baggage_span_attributes

tracer = get_tracer(__name__)

async def process_user_request(user_id: str, request_data: dict):
    with tracer.start_span("user_request_processing") as span:
        # Automatically add all baggage as span attributes
        set_baggage_span_attributes(span)

        # Add custom business attributes
        span.set_attribute("request.type", request_data.get("type"))
        span.set_attribute("request.priority", request_data.get("priority", "normal"))
        span.set_attribute("user.id", user_id)

        # Your business logic
        result = await perform_processing(request_data)

        # Add result attributes
        span.set_attribute("result.items_count", len(result))
        span.set_attribute("result.status", "success")

        return result
```

### GenAI Operations Tracing

```python
from pharia_telemetry import (
    constants,
    set_baggage_item,
    create_genai_span,
    set_genai_span_usage,
)

async def generate_ai_response(conversation_id: str, prompt: str, user_id: str):
    # Set context
    set_baggage_item(constants.Baggage.USER_ID, user_id)
    set_baggage_item(constants.Baggage.CHAT_QA_CONVERSATION_ID, conversation_id)

    # Use new GenAI convenience function
    with create_genai_span(
        operation_name=constants.GenAI.Values.OperationName.CHAT,
        agent_id=constants.GenAI.Values.AgentId.QA_CHAT,
        model="llama-3.1-8B",
        conversation_id=conversation_id,
    ) as span:
        if span:
            # Add custom attributes
            span.set_attribute("ai.prompt_length", len(prompt))
            span.set_attribute("ai.temperature", 0.7)
        span.set_attribute("ai.max_tokens", 1000)

        try:
            # Call AI model
            response = await ai_model.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )

            # Add response attributes
            span.set_attribute("ai.response_length", len(response))
            span.set_attribute("ai.tokens_used", response.usage.total_tokens)
            span.set_attribute("ai.finish_reason", response.finish_reason)

            return response.text

        except Exception as e:
            span.set_status(trace.Status(
                trace.StatusCode.ERROR,
                f"AI generation failed: {str(e)}"
            ))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
```

### Complex Workflow Tracing

```python
async def complex_data_pipeline(user_id: str, dataset_id: str):
    set_baggage_item(constants.Baggage.USER_ID, user_id)
    set_baggage_item(constants.Baggage.DATA_DATASET_IDS, dataset_id)

    with tracer.start_span("data_pipeline") as parent_span:
        parent_span.set_attribute("pipeline.dataset_id", dataset_id)
        parent_span.set_attribute("pipeline.user_id", user_id)

        # Step 1: Data extraction
        with tracer.start_span("data_extraction", parent=parent_span) as extract_span:
            extract_span.set_attribute("extraction.source", "database")
            raw_data = await extract_data(dataset_id)
            extract_span.set_attribute("extraction.records_count", len(raw_data))

        # Step 2: Data transformation
        with tracer.start_span("data_transformation", parent=parent_span) as transform_span:
            transform_span.set_attribute("transformation.type", "normalization")
            processed_data = await transform_data(raw_data)
            transform_span.set_attribute("transformation.records_processed", len(processed_data))
            transform_span.set_attribute("transformation.records_filtered", len(raw_data) - len(processed_data))

        # Step 3: Data loading
        with tracer.start_span("data_loading", parent=parent_span) as load_span:
            load_span.set_attribute("loading.destination", "vector_store")
            result = await load_data(processed_data)
            load_span.set_attribute("loading.records_loaded", result.loaded_count)
            load_span.set_attribute("loading.batch_id", result.batch_id)

        # Set pipeline summary
        parent_span.set_attribute("pipeline.total_records", len(raw_data))
        parent_span.set_attribute("pipeline.processed_records", len(processed_data))
        parent_span.set_attribute("pipeline.loaded_records", result.loaded_count)
        parent_span.set_attribute("pipeline.batch_id", result.batch_id)

        return result
```

## Performance Optimization

### Selective Instrumentation

```python
# Only trace critical paths
TRACE_ENABLED_PATHS = {"/api/critical", "/api/important"}

@app.middleware("http")
async def selective_tracing_middleware(request: Request, call_next):
    should_trace = any(request.url.path.startswith(path) for path in TRACE_ENABLED_PATHS)

    if should_trace:
        # Full tracing for critical paths
        with tracer.start_span(f"{request.method} {request.url.path}") as span:
            # ... full tracing logic
            response = await call_next(request)
            return response
    else:
        # No tracing overhead for non-critical paths
        return await call_next(request)
```

### Conditional Span Events

```python
import os

DEBUG_TRACING = os.getenv("DEBUG_TRACING", "false").lower() == "true"

async def process_with_optional_events(data):
    with tracer.start_span("data_processing") as span:
        span.set_attribute("data.size", len(data))

        for i, item in enumerate(data):
            result = await process_item(item)

            # Only add events in debug mode
            if DEBUG_TRACING and i % 100 == 0:
                span.add_event("processing_checkpoint", {
                    "items_processed": i,
                    "current_item_id": item.id
                })

        span.set_attribute("processing.items_processed", len(data))
```

### Sampling for High-Volume Operations

```python
import random

SAMPLE_RATE = 0.1  # Trace 10% of operations

async def high_volume_operation(item_id: str):
    should_trace = random.random() < SAMPLE_RATE

    if should_trace:
        with tracer.start_span("high_volume_operation") as span:
            span.set_attribute("item.id", item_id)
            span.set_attribute("sampled", True)
            return await perform_operation(item_id)
    else:
        # No tracing overhead
        return await perform_operation(item_id)
```

## Error Handling and Status

### Proper Error Handling

```python
async def operation_with_error_handling():
    with tracer.start_span("risky_operation") as span:
        try:
            result = await risky_operation()
            span.set_status(trace.Status(trace.StatusCode.OK))
            return result

        except ValidationError as e:
            # Client error - not a service failure
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"Validation failed: {str(e)}"))
            span.set_attribute("error.type", "validation")
            span.set_attribute("error.message", str(e))
            raise

        except DatabaseError as e:
            # Service error - infrastructure issue
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"Database error: {str(e)}"))
            span.set_attribute("error.type", "database")
            span.set_attribute("error.message", str(e))
            span.set_attribute("error.retryable", True)
            raise

        except Exception as e:
            # Unknown error
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"Unknown error: {str(e)}"))
            span.set_attribute("error.type", "unknown")
            span.set_attribute("error.message", str(e))
            raise
```

## Testing Manual Instrumentation

### Unit Testing with Spans

```python
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

@pytest.fixture
def tracer_provider():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider, exporter

async def test_manual_instrumentation(tracer_provider):
    provider, exporter = tracer_provider
    tracer = provider.get_tracer(__name__)

    # Your function using manual instrumentation
    with tracer.start_span("test_operation") as span:
        span.set_attribute("test.value", "success")

    # Verify span was created correctly
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_operation"
    assert spans[0].attributes["test.value"] == "success"
```

## Next Steps

- **Learn about context** → [Baggage & Context Guide](baggage-and-context.md)
- **Configure production** → [Configuration Guide](configuration.md)
- **Real-world examples** → [Integration Examples](integration-examples.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
