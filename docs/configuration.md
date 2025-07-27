# Configuration Guide

This guide covers how to configure `pharia-telemetry` for different environments, from development to production.

## Environment Variables

### Core Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Yes | None | OTLP endpoint URL for traces |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | None | Authentication headers |
| `OTEL_SERVICE_NAME` | No | None | Service name (can be set in code) |
| `ENVIRONMENT` | No | `"development"` | Deployment environment |
| `HOSTNAME` | No | System hostname | Service instance identifier |

### Development/Debug Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `"INFO"` | Enable debug logging and console exporter |
| `OTEL_TRACES_EXPORTER` | `"otlp"` | Set to `"console"` to print traces locally |
| `DEBUG_TRACING` | `"false"` | Enable verbose tracing debug information |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SAMPLE_RATE` | `"1.0"` | Sampling rate (0.0 to 1.0) |
| `OTEL_BSP_MAX_QUEUE_SIZE` | `"2048"` | Batch span processor queue size |
| `OTEL_BSP_EXPORT_TIMEOUT` | `"30000"` | Export timeout in milliseconds |
| `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | `"512"` | Maximum batch size for exports |

## Basic Setup Examples

### Development Environment

```bash
# Minimal development setup
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"
export ENVIRONMENT="development"
export LOG_LEVEL="DEBUG"
export OTEL_TRACES_EXPORTER="console"  # Also print traces to console
```

```python
from pharia_telemetry import setup_basic_tracing

# Simple development setup
setup_basic_tracing("my-service-dev", service_version="dev")
```

### Staging Environment

```bash
# Staging configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="https://staging-otel.example.com/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer staging-token"
export ENVIRONMENT="staging"
export OTEL_SERVICE_NAME="my-service"
export OTEL_SAMPLE_RATE="0.5"  # Sample 50% in staging
```

```python
from pharia_telemetry import setup_basic_tracing

setup_basic_tracing(
    service_name="my-service",
    service_version="1.2.3",
    enable_console_exporter=False,  # Disable console in staging
)
```

### Production Environment

```bash
# Production configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.example.com/v1/traces"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer prod-token"
export ENVIRONMENT="production"
export OTEL_SERVICE_NAME="my-service"
export OTEL_SAMPLE_RATE="0.1"  # Sample 10% in production
export HOSTNAME="server-01"
```

```python
from pharia_telemetry import setup_basic_tracing

setup_basic_tracing(
    service_name="my-service",
    service_version="1.2.3",
    enable_console_exporter=False,
    enable_baggage_processor=True,
)
```

## Advanced Configuration

### Custom Resource Attributes

```python
from pharia_telemetry import setup_basic_tracing

setup_basic_tracing(
    service_name="user-service",
    service_version="1.2.3",
    deployment_environment="production",
    additional_resource_attributes={
        "service.namespace": "user-management",
        "deployment.region": "us-west-2",
        "deployment.cluster": "prod-cluster-01",
        "service.owner": "platform-team",
        "cost.center": "engineering",
    }
)
```

### Manual Setup with Full Control

```python
from pharia_telemetry import (
    setup_otel_propagators,
    create_tracer_provider,
    add_baggage_span_processor,
    add_otlp_exporter,
    add_console_exporter,
)

# 1. Setup propagators
setup_otel_propagators()

# 2. Create tracer provider with custom configuration
provider = create_tracer_provider(
    service_name="my-service",
    service_version="1.2.3",
    deployment_environment="production",
    additional_resource_attributes={
        "service.namespace": "api",
        "deployment.region": "eu-central-1",
    }
)

# 3. Add span processors
add_baggage_span_processor(provider, prefix_filter="app.")

# 4. Add exporters
add_otlp_exporter(
    provider,
    endpoint="https://otel.example.com/v1/traces",
    headers={"Authorization": "Bearer your-token"},
)

# Development only
if os.getenv("ENVIRONMENT") == "development":
    add_console_exporter(provider)
```

### Sampling Configuration

```python
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBased,
    ALWAYS_ON,
    ALWAYS_OFF,
)
from pharia_telemetry import create_tracer_provider

# Simple ratio-based sampling
provider = create_tracer_provider(
    service_name="my-service",
    sampler=TraceIdRatioBased(rate=0.1),  # 10% sampling
)

# Parent-based sampling (respects upstream sampling decisions)
provider = create_tracer_provider(
    service_name="my-service",
    sampler=ParentBased(root=TraceIdRatioBased(rate=0.1)),
)

# Environment-based sampling
sampling_rate = {
    "development": 1.0,
    "staging": 0.5,
    "production": 0.1,
}.get(os.getenv("ENVIRONMENT", "development"), 1.0)

provider = create_tracer_provider(
    service_name="my-service",
    sampler=TraceIdRatioBased(rate=sampling_rate),
)
```

## Service-Specific Configuration

### FastAPI Configuration

```python
from fastapi import FastAPI
from pharia_telemetry import setup_basic_tracing, create_trace_correlation_processor
import structlog
import os

app = FastAPI()

@app.on_event("startup")
async def configure_observability():
    # Environment-specific configuration
    environment = os.getenv("ENVIRONMENT", "development")
    service_version = os.getenv("SERVICE_VERSION", "dev")

    # Setup telemetry
    setup_basic_tracing(
        service_name="api-service",
        service_version=service_version,
        deployment_environment=environment,
        enable_console_exporter=(environment == "development"),
        additional_resource_attributes={
            "service.type": "api",
            "framework": "fastapi",
        }
    )

    # Setup structured logging
    processors = [structlog.processors.TimeStamper()]

    if environment in ["staging", "production"]:
        processors.append(create_trace_correlation_processor())

    if environment == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": os.getenv("ENVIRONMENT")}
```

### Background Worker Configuration

```python
import asyncio
import signal
import os
from pharia_telemetry import setup_basic_tracing

class BackgroundWorker:
    def __init__(self):
        self.running = False
        self.setup_observability()

    def setup_observability(self):
        environment = os.getenv("ENVIRONMENT", "development")

        # Higher sampling for workers (they process fewer requests)
        sample_rate = {
            "development": 1.0,
            "staging": 1.0,
            "production": 0.5,  # Still high for background tasks
        }.get(environment, 1.0)

        setup_basic_tracing(
            service_name="background-worker",
            service_version=os.getenv("SERVICE_VERSION", "dev"),
            deployment_environment=environment,
            sample_rate=sample_rate,
            additional_resource_attributes={
                "service.type": "worker",
                "worker.queue": os.getenv("QUEUE_NAME", "default"),
            }
        )

    async def start(self):
        self.running = True

        # Setup graceful shutdown
        def signal_handler(signum, frame):
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        while self.running:
            await self.process_tasks()
            await asyncio.sleep(1)

    async def process_tasks(self):
        # Your task processing logic here
        pass

if __name__ == "__main__":
    worker = BackgroundWorker()
    asyncio.run(worker.start())
```

## Container and Kubernetes Configuration

### Docker Environment

```dockerfile
# Dockerfile
FROM python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set default environment variables
ENV ENVIRONMENT=production
ENV OTEL_SERVICE_NAME=my-service
ENV OTEL_SAMPLE_RATE=0.1

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "my_service"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-service
  template:
    metadata:
      labels:
        app: my-service
    spec:
      containers:
      - name: my-service
        image: my-service:1.2.3
        env:
        # OpenTelemetry Configuration
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:4318/v1/traces"
        - name: OTEL_SERVICE_NAME
          value: "my-service"
        - name: ENVIRONMENT
          value: "production"
        - name: OTEL_SAMPLE_RATE
          value: "0.1"

        # Service Configuration
        - name: SERVICE_VERSION
          value: "1.2.3"
        - name: HOSTNAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        # Authentication (from secrets)
        - name: OTEL_EXPORTER_OTLP_HEADERS
          valueFrom:
            secretKeyRef:
              name: otel-secrets
              key: auth-header

        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-config
data:
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://jaeger-collector:4318/v1/traces"
  OTEL_SAMPLE_RATE: "0.1"
  ENVIRONMENT: "production"

---
apiVersion: v1
kind: Secret
metadata:
  name: otel-secrets
type: Opaque
data:
  auth-header: QXV0aG9yaXphdGlvbjogQmVhcmVyIHlvdXItdG9rZW4=  # Base64 encoded
```

## Monitoring and Alerting Configuration

### Health Check Endpoint

```python
from fastapi import FastAPI, HTTPException
from pharia_telemetry import get_tracer
import asyncio

app = FastAPI()
tracer = get_tracer(__name__)

@app.get("/health")
async def health_check():
    """Health check that includes telemetry status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("SERVICE_VERSION", "unknown"),
        "environment": os.getenv("ENVIRONMENT", "unknown"),
    }

    # Check if telemetry is working
    try:
        with tracer.start_span("health_check_trace") as span:
            span.set_attribute("health.check", True)
            health_status["telemetry"] = "operational"
    except Exception as e:
        health_status["telemetry"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check dependencies
    try:
        # Example: database connectivity
        await check_database_connection()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

async def check_database_connection():
    # Your database check logic
    pass
```

### Metrics Endpoint

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_DURATION.observe(time.time() - start_time)

    return response
```

## Performance Tuning

### High-Throughput Services

```python
from pharia_telemetry import setup_basic_tracing

# Configuration for high-throughput services
setup_basic_tracing(
    service_name="high-throughput-api",
    service_version="1.0.0",
    sample_rate=0.01,  # 1% sampling for high-traffic services
    enable_console_exporter=False,
    additional_resource_attributes={
        "service.type": "high_throughput",
    }
)

# Custom batch processor settings via environment variables
os.environ.update({
    "OTEL_BSP_MAX_QUEUE_SIZE": "4096",      # Larger queue
    "OTEL_BSP_MAX_EXPORT_BATCH_SIZE": "1024",  # Larger batches
    "OTEL_BSP_EXPORT_TIMEOUT": "10000",     # Shorter timeout
    "OTEL_BSP_SCHEDULE_DELAY": "1000",      # More frequent exports
})
```

### Memory-Constrained Environments

```python
# Configuration for memory-constrained environments
setup_basic_tracing(
    service_name="memory-constrained-service",
    sample_rate=0.05,  # Lower sampling
    enable_baggage_processor=False,  # Disable if not needed
)

# Smaller batch processor settings
os.environ.update({
    "OTEL_BSP_MAX_QUEUE_SIZE": "512",       # Smaller queue
    "OTEL_BSP_MAX_EXPORT_BATCH_SIZE": "128", # Smaller batches
})
```

## Troubleshooting Configuration

### Debug Configuration

```python
import logging
from pharia_telemetry import setup_basic_tracing

# Enable debug logging for OpenTelemetry
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Enable console exporter to see traces locally
setup_basic_tracing(
    service_name="debug-service",
    enable_console_exporter=True,
    enable_baggage_processor=True,
)

# Verify configuration
import os
print("Configuration:")
for key, value in os.environ.items():
    if key.startswith("OTEL_"):
        print(f"  {key}: {value}")
```

### Validation

```python
from pharia_telemetry import get_tracer
import time

def validate_telemetry_setup():
    """Validate that telemetry is working correctly"""
    tracer = get_tracer(__name__)

    print("Testing telemetry setup...")

    with tracer.start_span("test_span") as span:
        span.set_attribute("test.validation", True)
        span.add_event("Test event")

        # Simulate some work
        time.sleep(0.1)

        print("✅ Span created successfully")

    print("✅ Telemetry validation complete")

if __name__ == "__main__":
    setup_basic_tracing("validation-test")
    validate_telemetry_setup()
```

## Next Steps

- **See complete examples** → [Integration Examples](integration-examples.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
- **Learn about context** → [Baggage & Context Guide](baggage-and-context.md)
- **Start with basics** → [Getting Started Guide](getting-started.md)
