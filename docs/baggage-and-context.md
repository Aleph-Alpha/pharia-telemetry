# Baggage & Context Propagation Guide

This guide covers how to use `pharia-telemetry` for context propagation across services using OpenTelemetry baggage.

## What is Baggage?

**Baggage** is OpenTelemetry's mechanism for propagating context (key-value pairs) across service boundaries. Unlike span attributes (which only exist on individual spans), baggage travels with requests through your entire distributed system.

### Key Benefits

- 🔄 **Automatic Propagation**: Set once, flows everywhere
- 🔍 **Searchable Context**: User context in every span and log
- 📊 **Correlation**: Connect distributed operations to users/sessions
- 🎯 **Debugging**: Trace issues back to specific users or workflows

## Standardized Baggage Keys

`pharia-telemetry` provides standardized keys for consistent context across all Pharia services:

### User-Level Context

```python
from pharia_telemetry import constants

# User identification
constants.Baggage.USER_ID        # "app.user.id"
constants.Baggage.SESSION_ID     # "app.session.id"
constants.Baggage.USER_INTENT    # "app.user.intent"
```

### Conversation Context

```python
# Chat system context
constants.Baggage.CHAT_QA_CONVERSATION_ID      # "aa.chat.qa.conversation.id"
constants.Baggage.CHAT_AGENT_CONVERSATION_ID   # "aa.chat.agent.conversation.id"
```

### Data Context

```python
# Data processing context
constants.Baggage.DATA_NAMESPACES      # "aa.data.namespaces"
constants.Baggage.DATA_COLLECTIONS     # "aa.data.collections"
constants.Baggage.DATA_DATASET_IDS     # "aa.data.dataset.ids"
```

### Pharia-Specific Context

```python
# Feature and policy context
constants.Baggage.FEATURE_FLAGS            # "pharia.feature.flags"
constants.Baggage.FEATURE_SET              # "pharia.feature.set"
constants.Baggage.PHARIA_USER_INTENT       # "pharia.user.intent"
constants.Baggage.PRIVACY_POLICY_PROMPTS   # "pharia.privacy.policy.prompts"
```

### Standardized Values

```python
from pharia_telemetry import constants

# User intent values
constants.Baggage.Values.UserIntent.QA_CHAT           # "pharia_qa_chat"
constants.Baggage.Values.UserIntent.AGENTIC_CHAT      # "pharia_agentic_chat"
constants.Baggage.Values.UserIntent.TRANSLATION       # "pharia_translation"
constants.Baggage.Values.UserIntent.TRANSCRIPTION     # "pharia_transcription"

# Feature flags
constants.Baggage.Values.FeatureFlags.ADVANCED_SEARCH    # "advanced_search"
constants.Baggage.Values.FeatureFlags.EXPERIMENTAL_UI    # "experimental_ui"
constants.Baggage.Values.FeatureFlags.BETA_FEATURES      # "beta_features"

# Feature sets
constants.Baggage.Values.FeatureSet.BASIC        # "basic"
constants.Baggage.Values.FeatureSet.PREMIUM      # "premium"
constants.Baggage.Values.FeatureSet.ENTERPRISE   # "enterprise"
```

## Basic Usage

### Setting Context

```python
from pharia_telemetry import constants, set_baggage_item

# Set user context (flows to ALL downstream operations)
set_baggage_item(constants.Baggage.USER_ID, "user-12345")
set_baggage_item(constants.Baggage.SESSION_ID, "session-67890")
set_baggage_item(constants.Baggage.USER_INTENT, constants.Baggage.Values.UserIntent.QA_CHAT)

# Set conversation context
set_baggage_item(constants.Baggage.CHAT_QA_CONVERSATION_ID, "conv-abc123")

# Set Pharia-specific context
set_baggage_item(constants.Baggage.FEATURE_FLAGS, "advanced_search,beta_features")
```

### Retrieving Context

```python
from pharia_telemetry import get_baggage_item, get_current_context

# Get specific context values
user_id = get_baggage_item(constants.Baggage.USER_ID)
session_id = get_baggage_item(constants.Baggage.SESSION_ID)

# Get all context at once (using the new convenience function)
all_context = get_current_context()
print(all_context)
# {
#   'trace_id': 'abc123...',
#   'span_id': 'def456...',
#   'baggage': {'app.user.id': 'user-12345', 'app.session.id': 'session-67890', ...}
# }
```

## Service Integration Patterns

### API Gateway Pattern

Set context at the entry point and it flows everywhere:

```python
from fastapi import FastAPI, Request, Header
from pharia_telemetry import setup_basic_tracing, BaggageKeys, set_baggage_item

app = FastAPI()

@app.on_event("startup")
async def setup():
    setup_basic_tracing("api-gateway", service_version="1.0.0")

@app.middleware("http")
async def set_user_context(request: Request, call_next):
    # Extract user info from headers/JWT
    user_id = request.headers.get("x-user-id")
    session_id = request.headers.get("x-session-id")

    if user_id:
        set_baggage_item(BaggageKeys.USER_ID, user_id)
    if session_id:
        set_baggage_item(BaggageKeys.SESSION_ID, session_id)

    # All downstream operations now have user context
    response = await call_next(request)
    return response

@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    # Context automatically available to downstream services
    async with httpx.AsyncClient() as client:
        # This HTTP call automatically includes user context in headers
        profile = await client.get(f"http://profile-service/users/{user_id}")
        preferences = await client.get(f"http://preference-service/users/{user_id}")

    return {"profile": profile.json(), "preferences": preferences.json()}
```

### Microservice Communication

Context automatically flows between services:

```python
# Service A: User Service
@app.get("/users/{user_id}")
async def get_user_info(user_id: str):
    # Set context in the first service
    set_baggage_item(BaggageKeys.USER_ID, user_id)
    set_baggage_item(BaggageKeys.USER_INTENT, "profile_lookup")

    # Call downstream services - context flows automatically
    async with httpx.AsyncClient() as client:
        profile = await client.get(f"http://profile-service/users/{user_id}")

    return profile.json()
```

```python
# Service B: Profile Service
@app.get("/users/{user_id}")
async def get_profile(user_id: str):
    # Context is automatically available from Service A
    current_user = get_baggage_item(BaggageKeys.USER_ID)
    intent = get_baggage_item(BaggageKeys.USER_INTENT)

    # Log with automatic context inclusion
    logger.info(f"Fetching profile for user: {current_user}, intent: {intent}")

    # Database query includes user context in spans automatically
    profile = await database.fetch_user_profile(user_id)
    return profile
```

### Background Task Processing

Preserve context in async tasks:

```python
from pharia_telemetry import get_all_baggage, set_baggage_item

async def schedule_background_task(task_data):
    # Capture current context
    current_context = get_all_baggage()

    # Schedule task with context
    await task_queue.enqueue(
        process_task,
        task_data=task_data,
        context=current_context
    )

async def process_task(task_data: dict, context: dict):
    # Restore context in background worker
    for key, value in context.items():
        set_baggage_item(key, value)

    # Now all operations have the original user context
    user_id = get_baggage_item(BaggageKeys.USER_ID)
    logger.info(f"Processing background task for user: {user_id}")

    # Process with full context
    result = await perform_processing(task_data)
    return result
```

## Advanced Patterns

### Conversation Flow Tracking

Track context across complex conversation workflows:

```python
async def handle_qa_chat_message(message: str, conversation_id: str, user_id: str):
    # Set conversation context
    set_baggage_item(BaggageKeys.USER_ID, user_id)
    set_baggage_item(BaggageKeys.CHAT_QA_CONVERSATION_ID, conversation_id)
    set_baggage_item(BaggageKeys.USER_INTENT, BaggageValues.UserIntent.AA_QA_CHAT)

    # All downstream operations automatically tagged with conversation context

    # 1. Retrieve conversation history (includes conversation_id in spans)
    history = await conversation_service.get_history(conversation_id)

    # 2. Process with AI service (includes user + conversation context)
    response = await ai_service.generate_response(message, history)

    # 3. Save to database (includes all context in spans)
    await conversation_service.save_message(conversation_id, message, response)

    return response
```

### Data Pipeline Context

Track data processing across multiple stages:

```python
async def process_dataset(dataset_id: str, user_id: str, collections: list[str]):
    # Set data processing context
    set_baggage_item(BaggageKeys.USER_ID, user_id)
    set_baggage_item(BaggageKeys.DATA_DATASET_IDS, dataset_id)
    set_baggage_item(BaggageKeys.DATA_COLLECTIONS, ",".join(collections))

    # All pipeline stages automatically include data context

    # Stage 1: Data extraction
    raw_data = await data_service.extract(dataset_id)

    # Stage 2: Data transformation
    processed_data = await transform_service.process(raw_data)

    # Stage 3: Data loading
    result = await vector_service.load(processed_data, collections)

    return result
```

### Multi-Service Correlation

Connect operations across different service types:

```python
# Chat Service
async def start_deep_research(query: str, user_id: str):
    research_id = generate_research_id()

    # Set research context
    set_baggage_item(BaggageKeys.USER_ID, user_id)
    set_baggage_item("research.id", research_id)
    set_baggage_item("research.query", query)

    # Trigger multiple services
    await search_service.perform_search(query)
    await analysis_service.analyze_results()
    await summarization_service.create_summary()

    return research_id

# All services automatically get research context in their spans and logs
```

## Automatic Span Attributes

When using `BaggageSpanProcessor` (enabled by default in `setup_basic_tracing`), all baggage automatically becomes searchable span attributes:

### Configuration

```python
from pharia_telemetry import setup_basic_tracing

# BaggageSpanProcessor automatically included
setup_basic_tracing("my-service", enable_baggage_processor=True)
```

### Result

Every span automatically includes baggage as attributes:

```json
{
  "trace_id": "abc123...",
  "span_id": "def456...",
  "name": "database_query",
  "attributes": {
    "db.statement": "SELECT * FROM users WHERE id = ?",
    "app.user.id": "user-12345",           ← From baggage
    "app.session.id": "session-67890",     ← From baggage
    "app.user.intent": "aa_qa_chat",       ← From baggage
    "aa.chat.qa.conversation.id": "conv-123" ← From baggage
  }
}
```

## Context Filtering

### Prefix Filtering

Only include specific baggage keys in spans/logs:

```python
from pharia_telemetry import BaggageSpanProcessor

# Only include app.* keys in spans
processor = BaggageSpanProcessor(prefix_filter="app.")
provider.add_span_processor(processor)
```

### Logging with Filtering

```python
from pharia_telemetry import TraceCorrelationProcessor

# Only include app.* keys in logs
processor = TraceCorrelationProcessor(
    include_baggage=True,
    baggage_prefix_filter="app."
)

structlog.configure(processors=[processor, ...])
```

## Best Practices

### ✅ Do's

1. **Set Context Early**: Set baggage at service entry points
```python
@app.middleware("http")
async def set_context(request: Request, call_next):
    set_baggage_item(BaggageKeys.USER_ID, extract_user_id(request))
    return await call_next(request)
```

2. **Use Standardized Keys**: Always use `BaggageKeys` constants
```python
# ✅ Good
set_baggage_item(BaggageKeys.USER_ID, user_id)

# ❌ Bad
set_baggage_item("user_id", user_id)  # Non-standard key
```

3. **Keep Values Small**: Baggage is sent with every request
```python
# ✅ Good
set_baggage_item(BaggageKeys.USER_ID, "user-123")

# ❌ Bad
set_baggage_item(BaggageKeys.USER_ID, json.dumps(full_user_object))
```

4. **Use Semantic Values**: Use standardized values when available
```python
# ✅ Good
set_baggage_item(BaggageKeys.USER_INTENT, BaggageValues.UserIntent.AA_QA_CHAT)

# ❌ Bad
set_baggage_item(BaggageKeys.USER_INTENT, "some_custom_value")
```

### ❌ Don'ts

1. **Don't Set Large Values**: Baggage travels with every request
```python
# ❌ Bad - too much data
user_profile = await get_full_user_profile(user_id)
set_baggage_item("user.profile", json.dumps(user_profile))
```

2. **Don't Use Sensitive Data**: Baggage may be logged/traced
```python
# ❌ Bad - sensitive information
set_baggage_item("user.password", password)
set_baggage_item("user.credit_card", cc_number)
```

3. **Don't Create Custom Keys Without Documentation**: Use standard keys
```python
# ❌ Bad - custom undocumented key
set_baggage_item("my.custom.key", value)
```

## Debugging Context Issues

### Check Current Context

```python
from pharia_telemetry import get_all_baggage

def debug_current_context():
    context = get_all_baggage()
    print("Current baggage context:")
    for key, value in context.items():
        print(f"  {key}: {value}")
```

### Trace Context Flow

```python
import structlog
from pharia_telemetry import create_trace_correlation_processor

# Enable debug logging to see context in logs
structlog.configure(
    processors=[
        create_trace_correlation_processor(),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)

async def trace_context_flow():
    logger.info("Before setting context")

    set_baggage_item(BaggageKeys.USER_ID, "test-user")
    logger.info("After setting user_id")

    await downstream_service.call()
    logger.info("After downstream call")
```

### Verify Propagation

```python
# Service A
async def call_service_b():
    set_baggage_item(BaggageKeys.USER_ID, "test-123")

    async with httpx.AsyncClient() as client:
        response = await client.get("http://service-b/test")

    return response

# Service B
@app.get("/test")
async def test_endpoint():
    user_id = get_baggage_item(BaggageKeys.USER_ID)
    return {"received_user_id": user_id}  # Should be "test-123"
```

## Legacy Compatibility

The library maintains backward compatibility:

```python
# Modern API (recommended)
from pharia_telemetry import BaggageKeys, BaggageValues

# Legacy API (still supported)
from pharia_telemetry import AppBaggageKeys, AppBaggageValues

# Both work identically
assert BaggageKeys.USER_ID == AppBaggageKeys.USER_ID
```

## Next Steps

- **Setup structured logging** → [Structured Logging Guide](structured-logging.md)
- **Configure for production** → [Configuration Guide](configuration.md)
- **See real examples** → [Integration Examples](integration-examples.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
