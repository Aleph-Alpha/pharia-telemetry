# Logging Context Injection Usage Examples

This document provides comprehensive examples of using the modular context injection API in pharia-telemetry.

## Overview

The logging module provides framework-agnostic injectors that add OpenTelemetry context to log records:

- **`TraceContextInjector`**: Adds trace_id and span_id
- **`BaggageContextInjector`**: Adds baggage context (user.id, session.id, etc.)
- **`CompositeContextInjector`**: Combines multiple injectors
- **`StructlogProcessor`**: Structlog-specific processor using the injectors

## Framework-Agnostic Usage

### Individual Injectors

```python
from pharia_telemetry.logging import TraceContextInjector, BaggageContextInjector

# Create injectors with custom configuration
trace_injector = TraceContextInjector(
    include_span_id=False,
    trace_id_key="request_id"
)

baggage_injector = BaggageContextInjector(
    prefix_filter="app.",  # Only include app.* baggage
    exclude_keys={"sensitive.token"}  # Exclude specific keys
)

# Use with any logging framework
log_record = {"message": "Processing request", "level": "info"}

# Apply injectors individually
enriched = trace_injector.inject(log_record)
enriched = baggage_injector.inject(enriched)

print(enriched)
# Output: {"message": "Processing request", "level": "info", "request_id": "abc123..."}
```

### Composite Injector

```python
from pharia_telemetry.logging import CompositeContextInjector

# Combine multiple injectors
composite = CompositeContextInjector([trace_injector, baggage_injector])

# Single injection call
enriched = composite.inject({"message": "Composite example"})
```

### Factory Functions

```python
from pharia_telemetry.logging import (
    create_trace_injector,
    create_baggage_injector,
    create_full_context_injector
)

# Individual injectors
trace_injector = create_trace_injector(
    include_span_id=False,
    trace_id_key="correlation_id"
)

baggage_injector = create_baggage_injector(
    prefix_filter="app.",
    exclude_keys={"internal.secret"}
)

# Full context injector (combines trace + baggage)
full_injector = create_full_context_injector(
    include_trace_id=True,
    include_span_id=False,
    include_baggage=True,
    baggage_prefix_filter="app.",
    trace_id_key="correlation_id"
)

log_dict = {"message": "Factory example"}
enriched = full_injector.inject(log_dict)
```

## Framework-Specific Usage

### Structlog

```python
import structlog
from pharia_telemetry.logging import create_full_context_injector

# Create a simple structlog processor using the injectors
class ContextProcessor:
    def __init__(self, **kwargs):
        self.injector = create_full_context_injector(**kwargs)

    def __call__(self, logger, method_name, event_dict):
        return self.injector.inject(event_dict)

# Configure structlog with context injection
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(),
        ContextProcessor(
            include_span_id=False,
            baggage_prefix_filter="app."
        ),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger(__name__)
logger.info("Request processed", user_action="login")
# Automatically includes trace_id and app.* baggage
```

### Standard Logging with Custom Formatter

```python
import logging
from pharia_telemetry.logging import create_full_context_injector

class ContextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.injector = create_full_context_injector(
            baggage_prefix_filter="app."
        )

    def format(self, record):
        # Convert LogRecord to dict
        log_dict = {
            "message": record.getMessage(),
            "level": record.levelname,
            "module": record.module,
            "timestamp": record.created,
        }

        # Inject context
        enriched = self.injector.inject(log_dict)

        # Convert back to formatted string
        return str(enriched)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ContextFormatter())

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.info("Standard logging example")
```

### FastAPI with Custom Middleware

```python
from fastapi import FastAPI, Request
from pharia_telemetry.logging import create_full_context_injector
import logging
import json

app = FastAPI()
injector = create_full_context_injector(baggage_prefix_filter="app.")

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Create base log entry
    log_entry = {
        "method": request.method,
        "url": str(request.url),
        "timestamp": time.time(),
    }

    # Inject OpenTelemetry context
    enriched_log = injector.inject(log_entry)

    # Log the enriched entry
    logging.info(json.dumps(enriched_log))

    response = await call_next(request)
    return response
```

## Advanced Configuration Examples

### Custom Key Names

```python
from pharia_telemetry.logging import TraceContextInjector

# Use custom key names for different systems
injector = TraceContextInjector(
    trace_id_key="correlation_id",
    span_id_key="operation_id"
)

result = injector.inject({"message": "Custom keys"})
# Output: {"message": "Custom keys", "correlation_id": "...", "operation_id": "..."}
```

### Selective Context Inclusion

```python
from pharia_telemetry.logging import create_full_context_injector

# Only include trace ID and specific baggage
injector = create_full_context_injector(
    include_trace_id=True,
    include_span_id=False,  # Exclude span ID
    include_baggage=True,
    baggage_prefix_filter="app.user.",  # Only user-related baggage
    baggage_exclude_keys={"app.user.password_hash"}  # Exclude sensitive data
)
```

### Multiple Environment Configuration

```python
import os
from pharia_telemetry.logging import create_full_context_injector

def create_injector_for_env():
    if os.getenv("ENVIRONMENT") == "production":
        # Production: minimal context for performance
        return create_full_context_injector(
            include_span_id=False,
            baggage_prefix_filter="app.user."
        )
    else:
        # Development: full context for debugging
        return create_full_context_injector(
            include_trace_id=True,
            include_span_id=True,
            include_baggage=True
        )

injector = create_injector_for_env()
```

## Migration from Legacy API

### Before (TraceCorrelationProcessor)
```python
# Old API (deprecated)
from pharia_telemetry.logging.correlation import TraceCorrelationProcessor

processor = TraceCorrelationProcessor(
    include_baggage=True,
    baggage_prefix_filter="app."
)
```

### After (New Modular API)
```python
# New API: Framework-agnostic
from pharia_telemetry.logging import create_full_context_injector

injector = create_full_context_injector(
    include_baggage=True,
    baggage_prefix_filter="app."
)

# For structlog specifically, create a simple processor:
class ContextProcessor:
    def __init__(self, **kwargs):
        self.injector = create_full_context_injector(**kwargs)

    def __call__(self, logger, method_name, event_dict):
        return self.injector.inject(event_dict)

processor = ContextProcessor(
    include_baggage=True,
    baggage_prefix_filter="app."
)
```

## Performance Considerations

### Lazy Initialization

```python
from pharia_telemetry.logging import TraceContextInjector, BaggageContextInjector

class OptimizedLogger:
    def __init__(self):
        self._trace_injector = None
        self._baggage_injector = None

    @property
    def trace_injector(self):
        if self._trace_injector is None:
            self._trace_injector = TraceContextInjector()
        return self._trace_injector

    def log_with_context(self, message):
        log_dict = {"message": message}
        return self.trace_injector.inject(log_dict)
```

### Conditional Context Injection

```python
from pharia_telemetry.logging import create_full_context_injector
import os

# Only inject context in specific environments
INJECT_CONTEXT = os.getenv("ENABLE_TRACING", "false").lower() == "true"
injector = create_full_context_injector() if INJECT_CONTEXT else None

def log_message(message):
    log_dict = {"message": message}

    if injector:
        log_dict = injector.inject(log_dict)

    print(log_dict)
```

## Error Handling

The injectors are designed to fail gracefully:

```python
from pharia_telemetry.logging import create_full_context_injector

injector = create_full_context_injector()

# Even if OpenTelemetry is not configured, injection works safely
log_dict = {"message": "Safe example"}
result = injector.inject(log_dict)
# Returns original dict if no context available
```

## Testing

```python
import pytest
from unittest.mock import patch
from pharia_telemetry.logging import TraceContextInjector

def test_trace_injection():
    injector = TraceContextInjector()

    with patch("pharia_telemetry.logging.injectors.trace.get_current_span") as mock_span:
        # Mock OpenTelemetry span
        mock_span.return_value = None  # No active span

        result = injector.inject({"message": "test"})
        assert result == {"message": "test"}  # No trace context added
```

This modular approach provides maximum flexibility while maintaining clean separation of concerns between trace context, baggage context, and framework-specific implementations.
