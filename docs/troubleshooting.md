# Troubleshooting Guide

This guide helps you diagnose and fix common issues with `pharia-telemetry`.

## Quick Diagnostics

### Check Basic Setup

```python
from pharia_telemetry import setup_basic_tracing, get_tracer
import os

def diagnose_setup():
    print("=== pharia-telemetry Diagnostics ===")

    # Check environment variables
    print("\n1. Environment Variables:")
    otel_vars = {k: v for k, v in os.environ.items() if k.startswith("OTEL_")}
    if otel_vars:
        for key, value in otel_vars.items():
            # Hide sensitive values
            display_value = "***HIDDEN***" if "TOKEN" in key or "KEY" in key else value
            print(f"   {key}: {display_value}")
    else:
        print("   ❌ No OTEL_ environment variables found")

    # Test basic setup
    print("\n2. Basic Setup Test:")
    try:
        setup_basic_tracing("diagnostic-test")
        print("   ✅ setup_basic_tracing() succeeded")
    except Exception as e:
        print(f"   ❌ setup_basic_tracing() failed: {e}")
        return

    # Test tracer creation
    print("\n3. Tracer Test:")
    try:
        tracer = get_tracer(__name__)
        print("   ✅ get_tracer() succeeded")
    except Exception as e:
        print(f"   ❌ get_tracer() failed: {e}")
        return

    # Test span creation
    print("\n4. Span Creation Test:")
    try:
        with tracer.start_span("test_span") as span:
            span.set_attribute("test", True)
        print("   ✅ Span creation succeeded")
    except Exception as e:
        print(f"   ❌ Span creation failed: {e}")

    print("\n=== Diagnosis Complete ===")

if __name__ == "__main__":
    diagnose_setup()
```

## Common Issues

### 1. No Traces Appearing

#### Symptoms
- Application runs without errors
- No traces visible in tracing backend
- Console exporter shows nothing

#### Diagnosis
```python
import os
from pharia_telemetry import setup_basic_tracing

# Enable debug logging
import logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Force console exporter
os.environ["OTEL_TRACES_EXPORTER"] = "console"

setup_basic_tracing("debug-test", enable_console_exporter=True)

# Create test span
from pharia_telemetry import get_tracer
tracer = get_tracer(__name__)
with tracer.start_span("debug_span"):
    print("If you see span output below, setup is working")
```

#### Common Causes & Solutions

**Missing OTLP Endpoint**
```bash
# Problem: No endpoint configured
# Solution: Set the endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-otlp-endpoint:4318/v1/traces"
```

**Wrong Endpoint URL**
```bash
# Problem: Incorrect URL format
❌ export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:14268"        # Wrong port
❌ export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4317"         # gRPC port
✅ export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4318/v1/traces"  # HTTP port
```

**Network Connectivity**
```python
# Test endpoint connectivity
import httpx
import asyncio

async def test_otlp_endpoint():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        print("❌ OTEL_EXPORTER_OTLP_ENDPOINT not set")
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json={"test": "data"})
            print(f"✅ Endpoint reachable: {response.status_code}")
    except Exception as e:
        print(f"❌ Endpoint unreachable: {e}")

asyncio.run(test_otlp_endpoint())
```

**Sampling Issues**
```python
# Problem: Sampling rate too low
# Solution: Increase sampling for debugging
setup_basic_tracing("debug-test", sample_rate=1.0)  # 100% sampling
```

### 2. Context Not Propagating

#### Symptoms
- Traces created but no connection between services
- Baggage values not appearing in downstream services
- Each service shows isolated traces

#### Diagnosis
```python
from pharia_telemetry import (
    BaggageKeys,
    set_baggage_item,
    get_baggage_item,
    get_all_baggage
)

def test_context_propagation():
    print("=== Context Propagation Test ===")

    # Set test context
    set_baggage_item(BaggageKeys.USER_ID, "test-user-123")
    set_baggage_item("test.key", "test-value")

    # Check if context is set
    user_id = get_baggage_item(BaggageKeys.USER_ID)
    test_value = get_baggage_item("test.key")
    all_baggage = get_all_baggage()

    print(f"User ID: {user_id}")
    print(f"Test value: {test_value}")
    print(f"All baggage: {all_baggage}")

    if user_id and test_value:
        print("✅ Baggage working locally")
    else:
        print("❌ Baggage not working locally")

test_context_propagation()
```

#### Common Causes & Solutions

**Missing Propagators Setup**
```python
# Problem: Propagators not configured
# Solution: Ensure setup_basic_tracing() is called
from pharia_telemetry import setup_basic_tracing

# This sets up propagators automatically
setup_basic_tracing("my-service")
```

**Wrong HTTP Client**
```python
# Problem: HTTP client doesn't support propagation
❌ import requests  # Doesn't automatically propagate
✅ import httpx     # Use with OpenTelemetry instrumentation

# Make sure HTTPX is instrumented
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
HTTPXClientInstrumentor().instrument()
```

**Missing Headers in HTTP Calls**
```python
# Manual header extraction (if needed)
from opentelemetry import propagate

async def make_request_with_context():
    headers = {}
    propagate.inject(headers)  # Adds trace context to headers

    async with httpx.AsyncClient() as client:
        response = await client.get("http://downstream", headers=headers)

    return response
```

### 3. Logs Missing Trace Context

#### Symptoms
- Structured logs don't include `trace_id` or `span_id`
- Baggage values missing from log records
- Unable to correlate logs with traces

#### Diagnosis
```python
import structlog
from pharia_telemetry import create_trace_correlation_processor, get_tracer

# Setup logging with trace correlation
structlog.configure(
    processors=[
        create_trace_correlation_processor(),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)
tracer = get_tracer(__name__)

# Test logging with span
with tracer.start_span("test_logging"):
    logger.info("Test log message", test_field="test_value")
    # Check if output includes trace_id and span_id
```

#### Common Causes & Solutions

**Wrong Logging Approach**
```python
# Problem: Using wrong instrumentation for your logging library

# For standard logging - use OpenTelemetry instrumentation
✅ from opentelemetry.instrumentation.logging import LoggingInstrumentor
LoggingInstrumentor().instrument()

# For structlog - use pharia-telemetry processor
✅ from pharia_telemetry import create_trace_correlation_processor
structlog.configure(
    processors=[
        create_trace_correlation_processor(),
        structlog.processors.JSONRenderer(),
    ]
)
```

**Missing Trace Correlation Processor (structlog only)**
```python
# Problem: No trace correlation processor for structlog
❌ structlog.configure(processors=[structlog.processors.JSONRenderer()])

# Solution: Add trace correlation processor
✅ from pharia_telemetry import create_trace_correlation_processor
structlog.configure(
    processors=[
        create_trace_correlation_processor(),
        structlog.processors.JSONRenderer(),
    ]
)
```

**Wrong Processor Order**
```python
# Problem: Processors in wrong order
❌ structlog.configure(
    processors=[
        structlog.processors.JSONRenderer(),  # This should be last
        create_trace_correlation_processor(),
    ]
)

# Solution: Trace processor before renderer
✅ structlog.configure(
    processors=[
        create_trace_correlation_processor(),  # Add context first
        structlog.processors.JSONRenderer(),   # Render last
    ]
)
```

### 4. Performance Issues

#### Symptoms
- High latency in requests
- High memory usage
- Application slowdowns

#### Diagnosis
```python
import time
import psutil
import os
from pharia_telemetry import setup_basic_tracing, get_tracer

def performance_test():
    process = psutil.Process(os.getpid())

    print("=== Performance Test ===")
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    # Test with different sampling rates
    for sample_rate in [1.0, 0.1, 0.01]:
        start_time = time.time()
        memory_before = process.memory_info().rss

        setup_basic_tracing(f"perf-test-{sample_rate}", sample_rate=sample_rate)
        tracer = get_tracer(__name__)

        # Create many spans
        for i in range(1000):
            with tracer.start_span(f"test_span_{i}"):
                time.sleep(0.001)  # Simulate work

        duration = time.time() - start_time
        memory_after = process.memory_info().rss
        memory_diff = (memory_after - memory_before) / 1024 / 1024

        print(f"Sample rate {sample_rate}: {duration:.2f}s, +{memory_diff:.1f}MB")

performance_test()
```

#### Common Causes & Solutions

**High Sampling Rate in Production**
```python
# Problem: 100% sampling in high-traffic production
❌ setup_basic_tracing("prod-service", sample_rate=1.0)  # Too much overhead

# Solution: Lower sampling rate
✅ setup_basic_tracing("prod-service", sample_rate=0.1)   # 10% sampling
```

**Unnecessary Console Exporter**
```python
# Problem: Console exporter enabled in production
❌ setup_basic_tracing("prod-service", enable_console_exporter=True)

# Solution: Disable console exporter
✅ setup_basic_tracing("prod-service", enable_console_exporter=False)
```

**Too Many Span Events**
```python
# Problem: Adding events in hot loops
❌ for item in large_list:
    span.add_event("processing_item", {"item_id": item.id})  # Too many events

# Solution: Sample events
✅ for i, item in enumerate(large_list):
    if i % 100 == 0:  # Only every 100th item
        span.add_event("processing_checkpoint", {"items_processed": i})
```

### 5. Auto-Instrumentation Issues

#### Symptoms
- Manual spans work but no automatic HTTP/database traces
- Missing spans for framework operations
- Double instrumentation errors

#### Diagnosis
```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def check_instrumentation():
    print("=== Instrumentation Status ===")

    # Check FastAPI instrumentation
    if hasattr(FastAPIInstrumentor(), '_instrumented'):
        print("✅ FastAPI instrumentation active")
    else:
        print("❌ FastAPI not instrumented")

    # Check HTTPX instrumentation
    try:
        # This will show if HTTPX is instrumented
        import httpx
        client = httpx.Client()
        if hasattr(client, '_otel_'):
            print("✅ HTTPX instrumentation active")
        else:
            print("❌ HTTPX not instrumented")
        client.close()
    except Exception as e:
        print(f"❌ HTTPX instrumentation error: {e}")

check_instrumentation()
```

#### Common Causes & Solutions

**Wrong Setup Order**
```python
# Problem: Instrumentation before telemetry setup
❌ FastAPIInstrumentor.instrument_app(app)  # Too early
setup_basic_tracing("my-service")

# Solution: Setup telemetry first
✅ setup_basic_tracing("my-service")        # Foundation first
FastAPIInstrumentor.instrument_app(app)   # Then instrumentation
```

**Double Instrumentation**
```python
# Problem: Instrumenting twice
❌ FastAPIInstrumentor.instrument_app(app)
FastAPIInstrumentor.instrument_app(app)  # Error!

# Solution: Check if already instrumented
✅ if not getattr(app, '_otel_instrumented', False):
    FastAPIInstrumentor.instrument_app(app)
    app._otel_instrumented = True
```

**Missing Dependencies**
```bash
# Problem: Instrumentation package not installed
❌ from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# ModuleNotFoundError: No module named 'opentelemetry.instrumentation.fastapi'

# Solution: Install instrumentation packages
✅ pip install opentelemetry-instrumentation-fastapi
```

### 6. Kubernetes/Container Issues

#### Symptoms
- Works locally but not in Kubernetes
- No traces from containerized services
- Network connectivity errors

#### Diagnosis

**Check Pod Configuration**
```bash
# Check environment variables in pod
kubectl exec -it my-pod -- env | grep OTEL

# Check network connectivity
kubectl exec -it my-pod -- curl -v http://jaeger-collector:4318/v1/traces

# Check logs for errors
kubectl logs my-pod | grep -i otel
```

**Test in Container**
```python
# Add to your application for debugging
import socket
import os

def debug_container_networking():
    print("=== Container Network Debug ===")
    print(f"Hostname: {socket.gethostname()}")
    print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")

    # Test DNS resolution
    try:
        import socket
        endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', '').replace('http://', '').split('/')[0]
        if ':' in endpoint:
            host, port = endpoint.split(':')
            ip = socket.gethostbyname(host)
            print(f"✅ DNS resolution: {host} -> {ip}")
        else:
            print("❌ Cannot parse endpoint")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")

debug_container_networking()
```

#### Common Causes & Solutions

**Wrong Service URLs**
```yaml
# Problem: localhost URLs in Kubernetes
❌ OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318/v1/traces"

# Solution: Use Kubernetes service names
✅ OTEL_EXPORTER_OTLP_ENDPOINT: "http://jaeger-collector:4318/v1/traces"
```

**Missing Network Policies**
```yaml
# Check if network policies block traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-otel-traffic
spec:
  podSelector:
    matchLabels:
      app: my-service
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: jaeger-collector
    ports:
    - port: 4318
```

**Resource Limits**
```yaml
# Problem: Too restrictive resource limits
❌ resources:
    limits:
      memory: "64Mi"   # Too low for telemetry overhead

# Solution: Adequate resources
✅ resources:
    limits:
      memory: "512Mi"
    requests:
      memory: "256Mi"
```

## Debug Configuration

### Enable Maximum Debugging

```python
import logging
import os
from pharia_telemetry import setup_basic_tracing

# 1. Enable all OpenTelemetry debug logging
logging.basicConfig(level=logging.DEBUG)
for logger_name in ['opentelemetry', 'opentelemetry.sdk', 'opentelemetry.exporter']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# 2. Force console exporter
os.environ["OTEL_TRACES_EXPORTER"] = "console"

# 3. Enable baggage processor debug
os.environ["DEBUG_TRACING"] = "true"

# 4. Setup with maximum visibility
setup_basic_tracing(
    "debug-service",
    sample_rate=1.0,  # 100% sampling
    enable_console_exporter=True,
    enable_baggage_processor=True,
)

# 5. Test everything
from pharia_telemetry import get_tracer, set_baggage_item, BaggageKeys

tracer = get_tracer(__name__)
set_baggage_item(BaggageKeys.USER_ID, "debug-user")

with tracer.start_span("debug_span") as span:
    span.set_attribute("debug", True)
    span.add_event("Debug event")
    print("Debug span created - check console output above")
```

### Validate Complete Setup

```python
from pharia_telemetry import *
import structlog
import httpx
import asyncio

async def full_integration_test():
    print("=== Full Integration Test ===")

    # 1. Setup everything
    setup_basic_tracing("integration-test")

    structlog.configure(
        processors=[
            create_trace_correlation_processor(),
            structlog.processors.JSONRenderer(),
        ]
    )

    # 2. Test all components
    logger = structlog.get_logger(__name__)
    tracer = get_tracer(__name__)

    with tracer.start_span("integration_test") as span:
        # Test baggage
        set_baggage_item(BaggageKeys.USER_ID, "integration-user")

        # Test logging
        logger.info("Integration test message")

        # Test HTTP context propagation
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://httpbin.org/headers")
                headers = response.json().get("headers", {})

                if "traceparent" in headers:
                    print("✅ HTTP context propagation working")
                else:
                    print("❌ HTTP context propagation failed")

            except Exception as e:
                print(f"❌ HTTP test failed: {e}")

        # Test GenAI attributes
        set_gen_ai_span_attributes(
            operation_name=SpanAttributeValues.GenAiOperationName.CHAT,
            agent_id=SpanAttributeValues.GenAiAgentId.QA_CHAT,
        )

        span.set_attribute("integration.test", "completed")

    print("✅ Integration test completed")

if __name__ == "__main__":
    asyncio.run(full_integration_test())
```

## Getting Help

### Gather Debug Information

```python
def gather_debug_info():
    """Gather all relevant information for support"""
    import sys
    import platform
    import pkg_resources

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_variables": {
            k: v for k, v in os.environ.items()
            if k.startswith(("OTEL_", "ENVIRONMENT"))
        },
        "installed_packages": {
            pkg.project_name: pkg.version
            for pkg in pkg_resources.working_set
            if "opentelemetry" in pkg.project_name.lower() or "pharia" in pkg.project_name.lower()
        }
    }

    print("=== Debug Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")

    return info

gather_debug_info()
```

### Support Channels

- 📧 **Email**: conrad.poepke@aleph-alpha.com
- 🐛 **GitHub Issues**: [pharia-telemetry/issues](https://github.com/aleph-alpha/pharia-telemetry/issues)
- 💬 **Internal Slack**: #platform-engineering

### Include in Support Requests

1. **Debug information** (from `gather_debug_info()`)
2. **Error messages** (full stack traces)
3. **Configuration** (environment variables, setup code)
4. **Expected vs actual behavior**
5. **Steps to reproduce**

## Next Steps

- **Review configuration** → [Configuration Guide](configuration.md)
- **Check examples** → [Integration Examples](integration-examples.md)
- **Learn basics** → [Getting Started Guide](getting-started.md)
