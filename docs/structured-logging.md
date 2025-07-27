# Structured Logging Guide

This guide covers how to use `pharia-telemetry` for structured logging with automatic trace correlation and context inclusion.

## Overview

This guide covers **structured logging** with `structlog` using `pharia-telemetry`'s `TraceCorrelationProcessor`.

### When to Use This Approach

Use `pharia-telemetry`'s structured logging when you need:
- **Structured JSON logs** with `structlog`
- **Baggage context** (user_id, session_id) in every log record
- **Comprehensive correlation** across distributed services

### Alternative: Standard Logging

For **standard Python logging**, use OpenTelemetry's official instrumentation instead:
```python
from opentelemetry.instrumentation.logging import LoggingInstrumentor
LoggingInstrumentor().instrument()  # Adds trace_id and span_id to standard logs
```

### What pharia-telemetry Provides

`pharia-telemetry`'s `TraceCorrelationProcessor` automatically includes:
- **Trace ID and Span ID**: For correlating logs with traces
- **Baggage Context**: User and session context in every log record
- **Service Information**: Consistent service identification
- **Error Context**: Automatic error tracking in logs

## Basic Setup

### Quick Start with structlog

```python
import structlog
from pharia_telemetry import setup_telemetry, add_context_to_logs

# 1. Setup telemetry foundation
setup_telemetry("my-service", service_version="1.0.0")

# 2. Configure structured logging
injector = add_context_to_logs("structlog")
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        injector,  # Adds trace_id + baggage
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# 3. Use structured logging
logger = structlog.get_logger(__name__)
logger.info("Service started", service="my-service", version="1.0.0")
```

### Result

Your logs automatically include trace and context information:

```json
{
  "timestamp": "2024-01-01T12:00:00.123456Z",
  "level": "info",
  "event": "Service started",
  "service": "my-service",
  "version": "1.0.0",
  "trace_id": "abc123def456...",           ← Automatic
  "span_id": "789ghi012jkl...",            ← Automatic
  "app.user.id": "user-12345",             ← From baggage
  "app.session.id": "session-67890"        ← From baggage
}
```

## Advanced Configuration

### Custom Trace Correlation Processor

```python
from pharia_telemetry import TraceCorrelationProcessor

# Full control over what's included
processor = TraceCorrelationProcessor(
    include_trace_id=True,           # Include trace_id field
    include_span_id=True,            # Include span_id field
    include_baggage=True,            # Include all baggage as fields
    baggage_prefix_filter="app.",    # Only include app.* baggage keys
    trace_id_key="trace_id",         # Custom field name for trace ID
    span_id_key="span_id",           # Custom field name for span ID
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(),
        processor,  # Your custom processor
        structlog.processors.JSONRenderer(),
    ]
)
```

### Multiple Environment Setup

```python
import os
import structlog
from pharia_telemetry import create_trace_correlation_processor

def setup_logging():
    processors = [
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
    ]

    # Add trace correlation in production/staging
    if os.getenv("ENVIRONMENT") in ["production", "staging"]:
        processors.append(create_trace_correlation_processor())

    # Different formatting for different environments
    if os.getenv("ENVIRONMENT") == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
```

## Integration Patterns

### FastAPI Service with Structured Logging

```python
from fastapi import FastAPI, Request
import structlog
from pharia_telemetry import (
    setup_basic_tracing,
    create_trace_correlation_processor,
    BaggageKeys,
    set_baggage_item,
)

app = FastAPI()

@app.on_event("startup")
async def setup_observability():
    # Setup telemetry
    setup_basic_tracing("user-api", service_version="1.2.3")

    # Setup structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(),
            create_trace_correlation_processor(),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

logger = structlog.get_logger(__name__)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Set context
    user_id = request.headers.get("x-user-id")
    if user_id:
        set_baggage_item(BaggageKeys.USER_ID, user_id)

    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
    )

    response = await call_next(request)

    # Log request completion
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
    )

    return response

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    # All logs automatically include trace_id and user context
    logger.info("Fetching user profile", user_id=user_id)

    try:
        profile = await user_service.get_profile(user_id)
        logger.info("Profile fetched successfully", profile_size=len(profile))
        return profile

    except UserNotFoundError:
        logger.warning("User not found", user_id=user_id)
        raise HTTPException(status_code=404, detail="User not found")

    except Exception as e:
        logger.error("Failed to fetch user profile", error=str(e), user_id=user_id)
        raise
```

### Background Task Processing

```python
import structlog
from pharia_telemetry import (
    setup_basic_tracing,
    create_trace_correlation_processor,
    get_all_baggage,
    set_baggage_item,
    BaggageKeys,
)

# Setup
setup_basic_tracing("task-processor")
structlog.configure(
    processors=[
        create_trace_correlation_processor(),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)

async def process_user_data(user_id: str, task_data: dict, context: dict = None):
    # Restore context if provided
    if context:
        for key, value in context.items():
            set_baggage_item(key, value)
    else:
        # Set basic context
        set_baggage_item(BaggageKeys.USER_ID, user_id)

    # Start processing with automatic context in logs
    logger.info("Starting data processing", task_type=task_data.get("type"))

    try:
        # Processing steps with detailed logging
        logger.info("Validating input data", data_size=len(task_data))
        validated_data = await validate_data(task_data)

        logger.info("Processing data", validation_passed=True)
        result = await process_data(validated_data)

        logger.info(
            "Processing completed successfully",
            items_processed=result.items_count,
            processing_time_ms=result.duration_ms,
        )

        return result

    except ValidationError as e:
        logger.error(
            "Data validation failed",
            error=str(e),
            validation_errors=e.errors if hasattr(e, 'errors') else None,
        )
        raise

    except Exception as e:
        logger.error(
            "Processing failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
```

### Database Operations with Context

```python
import structlog
from pharia_telemetry import get_baggage_item, BaggageKeys

logger = structlog.get_logger(__name__)

class UserRepository:
    async def get_user_by_id(self, user_id: str):
        # Logs automatically include user context from baggage
        logger.info("Querying user by ID", user_id=user_id)

        try:
            user = await self.db.fetch_one(
                "SELECT * FROM users WHERE id = ?", user_id
            )

            if user:
                logger.info("User found in database", user_id=user_id)
                return user
            else:
                logger.warning("User not found in database", user_id=user_id)
                return None

        except Exception as e:
            logger.error(
                "Database query failed",
                user_id=user_id,
                error=str(e),
                query="SELECT * FROM users WHERE id = ?",
            )
            raise

    async def create_user(self, user_data: dict):
        # Get current user context for audit logging
        current_user = get_baggage_item(BaggageKeys.USER_ID)

        logger.info(
            "Creating new user",
            email=user_data.get("email"),
            created_by=current_user,
        )

        try:
            user_id = await self.db.execute(
                "INSERT INTO users (email, name) VALUES (?, ?)",
                user_data["email"],
                user_data["name"],
            )

            logger.info(
                "User created successfully",
                new_user_id=user_id,
                email=user_data.get("email"),
                created_by=current_user,
            )

            return user_id

        except Exception as e:
            logger.error(
                "Failed to create user",
                email=user_data.get("email"),
                error=str(e),
                created_by=current_user,
            )
            raise
```

## Error Handling and Debugging

### Exception Logging with Context

```python
import structlog
import traceback
from pharia_telemetry import get_all_baggage

logger = structlog.get_logger(__name__)

async def risky_operation(data: dict):
    logger.info("Starting risky operation", data_type=data.get("type"))

    try:
        result = await perform_operation(data)
        logger.info("Operation completed successfully", result_size=len(result))
        return result

    except ValidationError as e:
        logger.warning(
            "Validation failed",
            error=str(e),
            validation_errors=getattr(e, 'errors', None),
            input_data=data,
        )
        raise

    except ConnectionError as e:
        logger.error(
            "Connection failed",
            error=str(e),
            retry_count=getattr(e, 'retry_count', 0),
            endpoint=getattr(e, 'endpoint', None),
        )
        raise

    except Exception as e:
        # Include full context for unknown errors
        current_context = get_all_baggage()

        logger.error(
            "Unknown error occurred",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            context=current_context,
            input_data=data,
        )
        raise
```

### Debug Logging

```python
import os
import structlog

# Enable debug logging based on environment
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

logger = structlog.get_logger(__name__)

async def complex_operation(data):
    logger.info("Starting complex operation")

    for i, item in enumerate(data):
        if DEBUG_LOGGING:
            logger.debug(
                "Processing item",
                item_index=i,
                item_id=getattr(item, 'id', None),
                item_type=type(item).__name__,
            )

        result = await process_item(item)

        if DEBUG_LOGGING and i % 100 == 0:
            logger.debug(f"Processed {i} items so far")

    logger.info("Complex operation completed", total_items=len(data))
```

## Log Analysis and Observability

### Structured Queries

With structured logging, you can easily query logs:

```bash
# Find all logs for a specific user
jq '.["app.user.id"] == "user-12345"' logs.jsonl

# Find all errors for a specific trace
jq '.level == "error" and .trace_id == "abc123def456"' logs.jsonl

# Find all logs for a specific conversation
jq '.["aa.chat.qa.conversation.id"] == "conv-789"' logs.jsonl

# Performance analysis - find slow operations
jq '.processing_time_ms > 1000' logs.jsonl
```

### Log Correlation with Traces

```json
{
  "timestamp": "2024-01-01T12:00:00.123456Z",
  "level": "error",
  "event": "Database query failed",
  "trace_id": "abc123def456...",           ← Use this to find related spans
  "span_id": "789ghi012jkl...",            ← Exact span with this error
  "app.user.id": "user-12345",             ← User affected
  "query": "SELECT * FROM users WHERE id = ?",
  "error": "Connection timeout",
  "retry_count": 3
}
```

Use the `trace_id` to find all related operations in your tracing system, and `span_id` to find the exact span where this log occurred.

## Best Practices

### ✅ Do's

1. **Use Structured Fields**: Don't put everything in the message
```python
# ✅ Good - structured fields
logger.info("User logged in", user_id=user_id, login_method="oauth")

# ❌ Bad - unstructured message
logger.info(f"User {user_id} logged in using oauth")
```

2. **Include Relevant Context**: Add business-specific fields
```python
# ✅ Good - business context
logger.info(
    "Payment processed",
    amount=payment.amount,
    currency=payment.currency,
    payment_method=payment.method,
    merchant_id=payment.merchant_id,
)
```

3. **Use Consistent Field Names**: Standardize across services
```python
# ✅ Good - consistent field names
logger.info("API call", endpoint="/users", method="GET", status_code=200)
logger.info("API call", endpoint="/orders", method="POST", status_code=201)
```

4. **Log at Appropriate Levels**:
```python
logger.debug("Detailed debugging info")     # Development debugging
logger.info("Normal operations")            # Key business events
logger.warning("Recoverable issues")        # Things that need attention
logger.error("Errors requiring action")     # Failures that need fixing
```

### ❌ Don'ts

1. **Don't Log Sensitive Data**:
```python
# ❌ Bad - sensitive data in logs
logger.info("User authenticated", password=user.password)
logger.info("Payment processed", credit_card=payment.cc_number)

# ✅ Good - safe logging
logger.info("User authenticated", user_id=user.id)
logger.info("Payment processed", payment_id=payment.id, last_four=payment.cc_last_four)
```

2. **Don't Use Unstructured Messages for Data**:
```python
# ❌ Bad - hard to parse
logger.info(f"Processed {count} items in {duration}ms with {errors} errors")

# ✅ Good - structured
logger.info("Processing completed", items_count=count, duration_ms=duration, error_count=errors)
```

3. **Don't Over-Log in Hot Paths**:
```python
# ❌ Bad - too much logging in loops
for item in large_dataset:
    logger.info("Processing item", item_id=item.id)  # Too much!

# ✅ Good - periodic logging
for i, item in enumerate(large_dataset):
    if i % 1000 == 0:
        logger.info("Processing progress", items_processed=i)
```

## Testing Structured Logging

### Unit Testing

```python
import pytest
import structlog
from structlog.testing import LogCapture

def test_user_service_logging():
    cap = LogCapture()
    structlog.configure(processors=[cap])
    logger = structlog.get_logger()

    # Your code that logs
    user_service = UserService(logger)
    user_service.create_user({"name": "John", "email": "john@example.com"})

    # Assert logging
    assert len(cap.entries) == 1
    log_entry = cap.entries[0]
    assert log_entry["event"] == "User created"
    assert log_entry["email"] == "john@example.com"
```

### Integration Testing

```python
import json
import tempfile
from pathlib import Path

def test_logging_integration():
    # Setup logging to file
    log_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    structlog.configure(
        processors=[
            create_trace_correlation_processor(),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=lambda: structlog.WriteLogger(log_file),
    )

    # Your application code
    logger = structlog.get_logger()
    logger.info("Test message", test_field="test_value")

    # Verify log output
    log_file.close()
    with open(log_file.name) as f:
        log_data = json.loads(f.read().strip())

    assert log_data["event"] == "Test message"
    assert log_data["test_field"] == "test_value"

    Path(log_file.name).unlink()  # Cleanup
```

## Next Steps

- **Configure for production** → [Configuration Guide](configuration.md)
- **See real examples** → [Integration Examples](integration-examples.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
- **Learn about context** → [Baggage & Context Guide](baggage-and-context.md)
