# Getting Started with pharia-telemetry

Welcome to `pharia-telemetry`! This guide will get you up and running with observability in your Pharia service in just a few minutes.

## What is pharia-telemetry?

`pharia-telemetry` is a **clean, minimal foundation library** that provides:
- **Context Propagation**: User and session data flows automatically between services
- **Structured Logging**: Logs include trace IDs and user context automatically
- **OpenTelemetry Setup**: Simple, high-level configuration for distributed tracing
- **Standardized Constants**: Clean, namespaced constants across all Pharia services

**Key Concept**: `pharia-telemetry` handles the foundation with minimal API surface, you add framework-specific instrumentation.

## Installation

```bash
# Basic installation
pip install pharia-telemetry

# With structlog support (for advanced structured logging)
pip install pharia-telemetry[structlog]
```

## Your First Instrumented Service

### 1. Basic Setup

```python
from pharia_telemetry import setup_telemetry

# One line to enable distributed tracing
setup_telemetry("my-service", service_version="1.0.0")

# That's it! Your service now:
# - Sends traces to OTLP endpoint (if configured)
# - Propagates context between services
# - Adds baggage as span attributes automatically
```

### 2. Add Context Propagation

```python
from pharia_telemetry import constants, set_baggage_item

# Set user context - this flows to ALL downstream operations
set_baggage_item(constants.Baggage.USER_ID, "user-123")
set_baggage_item(constants.Baggage.SESSION_ID, "session-456")

# Now every span and log will include this context!
```

### 3. Enable Structured Logging

```python
import structlog
from pharia_telemetry import add_context_to_logs

# Get a structured logging processor
injector = add_context_to_logs("structlog")

structlog.configure(
    processors=[
        injector,  # Adds trace_id + user context
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)
logger.info("User logged in")  # Includes trace_id and user_id automatically
```

## Complete FastAPI Example

Here's a complete example showing everything together:

```python
from fastapi import FastAPI
from pharia_telemetry import (
    setup_telemetry,
    constants,
    set_baggage_item,
    add_context_to_logs,
)
import structlog

app = FastAPI()

# Setup observability on startup
@app.on_event("startup")
async def setup_observability():
    # 1. Setup pharia-telemetry foundation
    setup_telemetry("user-service", service_version="1.0.0")

    # 2. Configure structured logging
    injector = add_context_to_logs("structlog")
    structlog.configure(
        processors=[
            injector,
            structlog.processors.JSONRenderer(),
        ]
    )

logger = structlog.get_logger(__name__)

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    # Set context that flows everywhere
    set_baggage_item(constants.Baggage.USER_ID, user_id)

    # Log with automatic context
    logger.info("Fetching user profile")

    # Your business logic here
    return {"user_id": user_id, "name": "John Doe"}
```

**What this gives you:**
- ✅ Every HTTP request is automatically traced
- ✅ User context flows to all downstream services
- ✅ Logs include trace_id and user_id automatically
- ✅ Spans include user context as searchable attributes

## Understanding the Foundation

### What pharia-telemetry Provides

```python
setup_telemetry()  # Sets up:
# ├── OpenTelemetry propagators (W3C TraceContext + Baggage)
# ├── Tracer provider with resource attributes
# ├── OTLP exporter (if endpoint configured)
# ├── Baggage span processor (adds baggage to spans)
# └── Console exporter (for development)
```

### What You Add (Framework-Specific)

```python
# For web frameworks
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)

# For databases
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
SQLAlchemyInstrumentor().instrument()

# For HTTP clients
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
HTTPXClientInstrumentor().instrument()
```

## Environment Configuration

Configure via environment variables:

```bash
# Required: Where to send traces
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otlp-endpoint.com"

# Optional: Authentication
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"

# Optional: Service metadata
export OTEL_SERVICE_NAME="user-service"
export ENVIRONMENT="production"
```

## Observability in Action

### Traces Include Context
```json
{
  "trace_id": "abc123...",
  "span_id": "def456...",
  "name": "GET /users/123",
  "attributes": {
    "http.method": "GET",
    "http.url": "/users/123",
    "app.user.id": "user-123",        ← From pharia-telemetry
    "app.session.id": "session-456"   ← From pharia-telemetry
  }
}
```

### Logs Include Context
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "message": "Fetching user profile",
  "trace_id": "abc123...",          ← From pharia-telemetry
  "span_id": "def456...",           ← From pharia-telemetry
  "app.user.id": "user-123",        ← From pharia-telemetry
  "app.session.id": "session-456"   ← From pharia-telemetry
}
```

## Clean API Design

pharia-telemetry features a **dramatically simplified API** with only **8 core exports**:

```python
from pharia_telemetry import (
    # High-level convenience (90% of users)
    setup_telemetry,           # One-function setup
    add_context_to_logs,       # Easy logging integration
    get_current_context,       # Manual context access

    # GenAI convenience functions
    create_genai_span,         # Create GenAI spans
    set_genai_span_usage,      # Set token usage
    set_genai_span_response,   # Set response attributes

    # Essential utilities (advanced users)
    set_baggage_item,          # Set context for propagation
    get_baggage_item,          # Get propagated context
    create_context_injector,   # Custom logging integration

    # Namespaced constants
    constants,                 # Access via constants.Baggage.USER_ID
)
```

## Clean Constants Structure

```python
from pharia_telemetry import constants

# General telemetry constants
user_id = constants.Baggage.USER_ID                    # "app.user.id"
qa_intent = constants.Baggage.Values.UserIntent.QA_CHAT  # "pharia_qa_chat"

# GenAI constants in separate module
model = constants.GenAI.REQUEST_MODEL                  # "gen_ai.request.model"
chat_op = constants.GenAI.Values.OperationName.CHAT    # "chat"
openai = constants.GenAI.Values.System.OPENAI          # "openai"
```

## Next Steps

Now that you have the basics working:

1. **Add Auto-Instrumentation** → [Auto-Instrumentation Guide](auto-instrumentation.md)
2. **Learn About Context** → [Baggage & Context Guide](baggage-and-context.md)
3. **Advanced Logging** → [Structured Logging Guide](structured-logging.md)
4. **GenAI Applications** → [GenAI Spans Guide](genai-spans.md)
5. **Real Examples** → [Integration Examples](integration-examples.md)

## Common Gotchas

### ❌ Not Setting OTLP Endpoint
```python
setup_telemetry("my-service")  # Won't send traces anywhere!
```

**Fix**: Set environment variable:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-endpoint.com"
```

### ❌ Setting Context Too Late
```python
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    result = await database.get_user(user_id)  # No context yet!
    set_baggage_item(constants.Baggage.USER_ID, user_id)  # Too late
```

**Fix**: Set context early:
```python
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    set_baggage_item(constants.Baggage.USER_ID, user_id)  # Set first
    result = await database.get_user(user_id)  # Now has context
```

### ❌ Missing Framework Instrumentation
```python
setup_telemetry("my-service")  # Only foundation
# HTTP requests, database calls not traced
```

**Fix**: Add framework-specific instrumentation:
```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)  # Now HTTP requests traced
```

## Getting Help

- 🔍 **Check examples**: [Integration Examples](integration-examples.md)
- 🐛 **Having issues?**: [Troubleshooting Guide](troubleshooting.md)
- 💬 **Need help?**: [GitHub Discussions](https://github.com/aleph-alpha/pharia-telemetry/discussions)
