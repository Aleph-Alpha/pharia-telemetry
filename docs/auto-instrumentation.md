# Auto-Instrumentation Guide

This guide covers adding OpenTelemetry auto-instrumentation to your `pharia-telemetry` foundation.

## Overview

`pharia-telemetry` provides the **foundation** (propagators, baggage, exporters), and you add **framework-specific auto-instrumentation** for automatic tracing of:
- HTTP requests and responses
- Database queries
- HTTP client calls
- Background tasks

## Available Auto-Instrumentations

| Library | Package | Purpose | Compatibility |
|---------|---------|---------|---------------|
| **FastAPI** | `opentelemetry-instrumentation-fastapi` | Web framework HTTP requests | ✅ Standard REST APIs<br/>⚠️ Issues with SSE/WebSockets |
| **SQLAlchemy** | `opentelemetry-instrumentation-sqlalchemy` | Database queries and operations | ✅ All SQL operations |
| **HTTPX** | `opentelemetry-instrumentation-httpx` | Async HTTP client requests | ✅ All HTTP/1.1 calls |
| **Requests** | `opentelemetry-instrumentation-requests` | HTTP client requests | ✅ All HTTP/1.1 calls |
| **URLLib3** | `opentelemetry-instrumentation-urllib3` | Low-level HTTP client | ✅ All HTTP/1.1 calls |
| **AsyncPG** | `opentelemetry-instrumentation-asyncpg` | PostgreSQL async operations | ✅ All PostgreSQL calls |
| **Psycopg** | `opentelemetry-instrumentation-psycopg` | PostgreSQL operations | ✅ All PostgreSQL calls |
| **Logging** | `opentelemetry-instrumentation-logging` | Standard library logging correlation | ✅ Standard `logging` module |

### Logging Instrumentation Options

There are **two approaches** for adding trace correlation to logs:

#### 1. **OpenTelemetry Logging Instrumentation** (Preferred)
```python
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Automatically adds trace_id and span_id to standard logging
LoggingInstrumentor().instrument()

import logging
logger = logging.getLogger(__name__)
logger.info("This message will include trace_id and span_id")
```

**Use when:**
- Using Python's standard `logging` module
- Want automatic instrumentation with minimal setup
- Don't need baggage context in logs

#### 2. **pharia-telemetry Structured Logging** (For Advanced Use Cases)
```python
import structlog
from pharia_telemetry import add_context_to_logs

# For structured logging with baggage context
injector = add_context_to_logs("structlog")
structlog.configure(
    processors=[
        injector,  # Adds trace_id + baggage
        structlog.processors.JSONRenderer(),
    ]
)
```

**Use when:**
- Using structured logging (structlog)
- Need baggage context (user_id, session_id) in logs
- Want comprehensive correlation data

**Recommendation**: Use OpenTelemetry's logging instrumentation for standard logging, and pharia-telemetry's processor for structured logging with baggage.

## Installation

```bash
# Install pharia-telemetry foundation
pip install pharia-telemetry

# Install instrumentations for your specific stack
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-instrumentation-sqlalchemy
pip install opentelemetry-instrumentation-httpx
pip install opentelemetry-instrumentation-requests
pip install opentelemetry-instrumentation-logging  # For standard logging
```

## Standard Setup Pattern

### 1. Foundation First

```python
from pharia_telemetry import setup_telemetry

# Always setup pharia-telemetry foundation first
setup_telemetry("my-service", service_version="1.0.0")
```

### 2. Add Framework Instrumentations

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Add framework-specific auto-instrumentation
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=database_engine)
HTTPXClientInstrumentor().instrument()
```

## Complete FastAPI Service Example

```python
from fastapi import FastAPI
from sqlalchemy import create_engine
from pharia_telemetry import setup_telemetry, constants, set_baggage_item

# Import instrumentors
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

app = FastAPI()

@app.on_event("startup")
async def setup_observability():
    # 1. Setup pharia-telemetry foundation (propagators, baggage, exporters)
    setup_telemetry("user-service", service_version="1.2.3")

    # 2. Add framework-specific auto-instrumentation
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()  # Standard logging correlation

    # 3. If you have a database
    engine = create_engine("postgresql://user:pass@localhost/db")
    SQLAlchemyInstrumentor().instrument(engine=engine)

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    # Set context using pharia-telemetry
    set_baggage_item(constants.Baggage.USER_ID, user_id)

    # All operations are now automatically traced:
    # - HTTP request (FastAPI instrumentation)
    # - Database queries (SQLAlchemy instrumentation)
    # - HTTP client calls (HTTPX instrumentation)
    # - User context flows through all operations (pharia-telemetry baggage)

    async with httpx.AsyncClient() as client:
        profile = await client.get(f"http://profile-service/users/{user_id}")

    user_data = await database.fetch_user(user_id)

    return {"user_id": user_id, "profile": profile.json(), "data": user_data}

### Logging Output Comparison

With both logging approaches set up, here's what you get:

#### Standard Logging (via LoggingInstrumentor)
```python
import logging
logger = logging.getLogger(__name__)
logger.info("User profile fetched", extra={"user_id": user_id})
```

**Output:**
```json
{
  "message": "User profile fetched",
  "user_id": "user-123",
  "otelTraceID": "abc123def456...",     ← Added by LoggingInstrumentor
  "otelSpanID": "789ghi012jkl...",      ← Added by LoggingInstrumentor
}
```

#### Structured Logging (via pharia-telemetry)
```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("User profile fetched", user_id=user_id)
```

**Output:**
```json
{
  "event": "User profile fetched",
  "user_id": "user-123",
  "trace_id": "abc123def456...",        ← Added by TraceCorrelationProcessor
  "span_id": "789ghi012jkl...",         ← Added by TraceCorrelationProcessor
  "app.user.id": "user-123",            ← Added from baggage
  "app.session.id": "session-456"       ← Added from baggage
}
```

**Key Difference**: pharia-telemetry's processor includes **baggage context** automatically.
```

## Framework-Specific Setup

### FastAPI
```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)
# Traces: HTTP requests, responses, middleware
```

### SQLAlchemy
```python
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Option 1: Instrument specific engine
SQLAlchemyInstrumentor().instrument(engine=my_engine)

# Option 2: Instrument all engines
SQLAlchemyInstrumentor().instrument()
# Traces: SQL queries, connection pooling, transactions
```

### HTTPX (Async HTTP Client)
```python
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

HTTPXClientInstrumentor().instrument()
# Traces: All HTTPX async HTTP calls
```

### Requests (Sync HTTP Client)
```python
from opentelemetry.instrumentation.requests import RequestsInstrumentor

RequestsInstrumentor().instrument()
# Traces: All requests library HTTP calls
```

### PostgreSQL (AsyncPG)
```python
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

AsyncPGInstrumentor().instrument()
# Traces: Async PostgreSQL operations
```

### Logging
```python
from opentelemetry.instrumentation.logging import LoggingInstrumentor

LoggingInstrumentor().instrument()
# Automatically adds trace_id and span_id to log records
```

## What Gets Traced

### HTTP Requests (FastAPI)
```json
{
  "name": "GET /users/123",
  "attributes": {
    "http.method": "GET",
    "http.url": "/users/123",
    "http.status_code": 200,
    "app.user.id": "user-123"  ← From pharia-telemetry baggage
  }
}
```

### Database Queries (SQLAlchemy)
```json
{
  "name": "SELECT users",
  "attributes": {
    "db.statement": "SELECT * FROM users WHERE id = ?",
    "db.system": "postgresql",
    "app.user.id": "user-123"  ← From pharia-telemetry baggage
  }
}
```

### HTTP Client Calls (HTTPX)
```json
{
  "name": "GET http://profile-service/users/123",
  "attributes": {
    "http.method": "GET",
    "http.url": "http://profile-service/users/123",
    "http.status_code": 200,
    "app.user.id": "user-123"  ← From pharia-telemetry baggage
  }
}
```

## ⚠️ Auto-Instrumentation Limitations

### Server-Sent Events (SSE) Issues

**Problem**: FastAPI with SSE streaming breaks auto-instrumentation
```python
# ❌ This causes context loss during streaming
FastAPIInstrumentor.instrument_app(app)

@app.get("/stream")
async def stream_data():
    async def generate():
        for item in data:
            yield f"data: {item}\n\n"  # Context lost here!

    return StreamingResponse(generate(), media_type="text/plain")
```

**Issues**:
- Context detached too early during exception handling
- Spans closed prematurely for streaming responses
- Performance overhead on each streamed chunk

**Solution**: Use manual instrumentation → [Manual Instrumentation Guide](manual-instrumentation.md)

### HTTP/2 Compatibility Issues

**Problem**: Limited HTTP/2 support with Hypercorn
```python
# ❌ Auto-instrumentation doesn't work well with HTTP/2
FastAPIInstrumentor.instrument_app(app)  # Limited HTTP/2 support

# Running with: hypercorn app:app --bind 0.0.0.0:8000 --http2
```

**Issues**:
- Protocol incompatibility with HTTP/2 features
- Server compatibility problems with Hypercorn
- Performance degradation for streaming endpoints

**Solution**: Use manual instrumentation → [Manual Instrumentation Guide](manual-instrumentation.md)

## Decision Matrix: Auto vs Manual

| Use Case | Recommendation | Reason |
|----------|---------------|---------|
| **Standard REST APIs** | ✅ **Auto-instrumentation** | Works reliably for typical HTTP patterns |
| **CRUD applications** | ✅ **Auto-instrumentation** | Perfect for database + HTTP operations |
| **Background tasks** | ✅ **Auto-instrumentation** | Standard async patterns work well |
| **SSE/WebSocket streaming** | ⚠️ **Manual instrumentation** | Context propagation issues with streams |
| **HTTP/2 services** | ⚠️ **Manual instrumentation** | Limited protocol support |
| **High-performance services** | ⚠️ **Manual instrumentation** | Reduce overhead, optimize hot paths |
| **Complex business logic** | ⚠️ **Manual instrumentation** | Custom span attributes and events |

## Performance Considerations

### Auto-Instrumentation Overhead
- **HTTP requests**: ~1-5ms per request
- **Database queries**: ~0.1-1ms per query
- **HTTP client calls**: ~0.5-2ms per call

### Optimization Tips
```python
# 1. Use sampling for high-traffic services
setup_basic_tracing("my-service", sample_rate=0.1)  # Sample 10%

# 2. Disable console exporter in production
setup_basic_tracing("my-service", enable_console_exporter=False)

# 3. Batch span exports (done by default)
# BatchSpanProcessor used automatically for better performance
```

## Debugging Auto-Instrumentation

### Check What's Instrumented
```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Check if already instrumented
if FastAPIInstrumentor().is_instrumented_by_opentelemetry:
    print("FastAPI already instrumented")
else:
    FastAPIInstrumentor.instrument_app(app)
```

### Enable Debug Logging
```python
import logging

# Enable OpenTelemetry debug logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Setup tracing
setup_basic_tracing("my-service")
```

### Verify Traces are Sent
```bash
# Enable console exporter to see traces locally
export OTEL_TRACES_EXPORTER="console"
```

## Common Issues

### ❌ Double Instrumentation
```python
# This causes problems:
FastAPIInstrumentor.instrument_app(app)
FastAPIInstrumentor.instrument_app(app)  # Don't do this!
```

### ❌ Wrong Order
```python
# Wrong order - instrumentation before foundation
FastAPIInstrumentor.instrument_app(app)  # No propagators set up yet!
setup_basic_tracing("my-service")  # Too late
```

**Fix**: Foundation first, then instrumentation:
```python
setup_basic_tracing("my-service")  # Foundation first
FastAPIInstrumentor.instrument_app(app)  # Then instrumentation
```

### ❌ Missing Dependencies
```python
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
# ModuleNotFoundError: No module named 'opentelemetry.instrumentation.sqlalchemy'
```

**Fix**: Install the instrumentation package:
```bash
pip install opentelemetry-instrumentation-sqlalchemy
```

## Next Steps

- **Manual control needed?** → [Manual Instrumentation Guide](manual-instrumentation.md)
- **Learn about context** → [Baggage & Context Guide](baggage-and-context.md)
- **Real-world examples** → [Integration Examples](integration-examples.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
