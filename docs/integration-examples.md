# Integration Examples

This guide provides complete, real-world examples of integrating `pharia-telemetry` into different types of applications and architectures.

## FastAPI Microservice

Complete example of a production FastAPI service with full observability.

### Project Structure

```
user-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   ├── services/
│   └── api/
├── requirements.txt
├── Dockerfile
└── k8s/
```

### Main Application (`app/main.py`)

```python
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog
import httpx
import asyncio
import os
from typing import Optional

from pharia_telemetry import (
    setup_basic_tracing,
    create_trace_correlation_processor,
    BaggageKeys,
    BaggageValues,
    set_baggage_item,
    get_baggage_item,
)

# Import auto-instrumentation
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# App setup
app = FastAPI(
    title="User Service",
    description="Microservice for user management",
    version="1.2.3"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize observability and services on startup"""
    environment = os.getenv("ENVIRONMENT", "development")
    service_version = os.getenv("SERVICE_VERSION", "dev")

    # 1. Setup pharia-telemetry foundation
    setup_basic_tracing(
        service_name="user-service",
        service_version=service_version,
        deployment_environment=environment,
        enable_console_exporter=(environment == "development"),
        additional_resource_attributes={
            "service.type": "api",
            "service.domain": "user-management",
            "framework": "fastapi",
        }
    )

    # 2. Setup logging
    # Note: Using standard logging with OpenTelemetry instrumentation
    # For structured logging with baggage, see Background Task Processor example
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 3. Add auto-instrumentation
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument()  # Standard logging correlation

    # 4. Database instrumentation (if using SQLAlchemy)
    # from .database import engine
    # SQLAlchemyInstrumentor().instrument(engine=engine)

    global logger
    logger = logging.getLogger(__name__)
    logger.info("User service started", extra={"version": service_version, "environment": environment})

logger = logging.getLogger(__name__)

# Middleware for request context
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Set request context for observability"""
    # Extract user context from headers
    user_id = request.headers.get("x-user-id")
    session_id = request.headers.get("x-session-id")
    request_id = request.headers.get("x-request-id")

    # Set baggage context
    if user_id:
        set_baggage_item(BaggageKeys.USER_ID, user_id)
    if session_id:
        set_baggage_item(BaggageKeys.SESSION_ID, session_id)

    # Set user intent based on endpoint
    if request.url.path.startswith("/api/profile"):
        set_baggage_item(BaggageKeys.USER_INTENT, "profile_management")
    elif request.url.path.startswith("/api/settings"):
        set_baggage_item(BaggageKeys.USER_INTENT, "settings_management")

    # Log request start (trace_id and span_id added automatically by LoggingInstrumentor)
    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
            "user_agent": request.headers.get("user-agent"),
        }
    )

    response = await call_next(request)

    # Log request completion
    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "request_id": request_id,
        }
    )

    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with telemetry validation"""
    return {
        "status": "healthy",
        "service": "user-service",
        "version": os.getenv("SERVICE_VERSION", "unknown"),
        "environment": os.getenv("ENVIRONMENT", "unknown"),
    }

# API endpoints
@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user by ID with full observability"""
    # Set user context
    set_baggage_item(BaggageKeys.USER_ID, user_id)

    logger.info("Fetching user profile", extra={"user_id": user_id})

    try:
        # Simulate database call (auto-instrumented)
        user_data = await fetch_user_from_db(user_id)

        if not user_data:
            logger.warning("User not found", extra={"user_id": user_id})
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch additional data from other services
        async with httpx.AsyncClient() as client:
            # These calls automatically include trace context
            preferences_response = await client.get(
                f"http://preferences-service/users/{user_id}/preferences"
            )

            if preferences_response.status_code == 200:
                preferences = preferences_response.json()
            else:
                logger.warning(
                    "Failed to fetch preferences",
                    extra={
                        "user_id": user_id,
                        "status_code": preferences_response.status_code
                    }
                )
                preferences = {}

        result = {
            "user": user_data,
            "preferences": preferences,
        }

        logger.info(
            "User profile fetched successfully",
            extra={
                "user_id": user_id,
                "has_preferences": bool(preferences)
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to fetch user profile",
            extra={
                "user_id": user_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/users")
async def create_user(user_data: dict):
    """Create new user with audit logging"""
    # Get current user for audit trail
    created_by = get_baggage_item(BaggageKeys.USER_ID)

    logger.info(
        "Creating new user",
        extra={
            "email": user_data.get("email"),
            "created_by": created_by
        }
    )

    try:
        # Validate user data
        if not user_data.get("email"):
            raise HTTPException(status_code=400, detail="Email is required")

        # Create user in database (auto-instrumented)
        user_id = await create_user_in_db(user_data)

        # Send welcome email (background task)
        await send_welcome_email(user_id, user_data["email"])

        logger.info(
            "User created successfully",
            extra={
                "user_id": user_id,
                "email": user_data.get("email"),
                "created_by": created_by
            }
        )

        return {"user_id": user_id, "status": "created"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to create user",
            extra={
                "email": user_data.get("email"),
                "error": str(e),
                "created_by": created_by
            }
        )
        raise HTTPException(status_code=500, detail="Failed to create user")

# Service functions
async def fetch_user_from_db(user_id: str):
    """Simulate database fetch (would be auto-instrumented with SQLAlchemy)"""
    # Simulate async database call
    await asyncio.sleep(0.1)
    return {"id": user_id, "email": f"user{user_id}@example.com", "name": "John Doe"}

async def create_user_in_db(user_data: dict):
    """Simulate database create (would be auto-instrumented with SQLAlchemy)"""
    await asyncio.sleep(0.2)
    return f"user-{len(user_data.get('email', ''))}"

async def send_welcome_email(user_id: str, email: str):
    """Send welcome email with context preservation"""
    logger.info(
        "Sending welcome email",
        extra={"user_id": user_id, "email": email}
    )

    # Simulate email service call
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://email-service/send",
            json={
                "to": email,
                "template": "welcome",
                "user_id": user_id,
            }
        )

    if response.status_code == 200:
        logger.info("Welcome email sent successfully", extra={"user_id": user_id})
    else:
        logger.error(
            "Failed to send welcome email",
            extra={
                "user_id": user_id,
                "status_code": response.status_code
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Dependencies (`requirements.txt`)

```txt
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
httpx>=0.24.0
structlog>=23.0.0
pharia-telemetry

# Auto-instrumentation packages
opentelemetry-instrumentation-fastapi
opentelemetry-instrumentation-httpx
opentelemetry-instrumentation-logging
opentelemetry-instrumentation-sqlalchemy  # If using database
```

## Background Task Processor

Example of a Celery-like background task processor with full context preservation using **structured logging**.

### Task Processor (`worker.py`)

```python
import asyncio
import signal
import json
import os
from typing import Dict, Any
import structlog

from pharia_telemetry import (
    setup_basic_tracing,
    create_trace_correlation_processor,
    get_all_baggage,
    set_baggage_item,
    BaggageKeys,
    get_tracer,
    set_gen_ai_span_attributes,
    SpanAttributeValues,
)

class TaskProcessor:
    def __init__(self):
        self.running = False
        self.tracer = None
        self.logger = None
        self.setup_observability()

    def setup_observability(self):
        """Setup observability for background worker"""
        environment = os.getenv("ENVIRONMENT", "development")

        # Higher sampling for background tasks
        sample_rate = {
            "development": 1.0,
            "staging": 1.0,
            "production": 0.5,
        }.get(environment, 1.0)

        setup_basic_tracing(
            service_name="task-processor",
            service_version=os.getenv("SERVICE_VERSION", "dev"),
            deployment_environment=environment,
            sample_rate=sample_rate,
            additional_resource_attributes={
                "service.type": "worker",
                "worker.queue": os.getenv("QUEUE_NAME", "default"),
            }
        )

        # Setup structured logging
        structlog.configure(
            processors=[
                create_trace_correlation_processor(),
                structlog.processors.TimeStamper(),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ]
        )

        self.tracer = get_tracer(__name__)
        self.logger = structlog.get_logger(__name__)

    async def start(self):
        """Start the task processor with graceful shutdown"""
        self.running = True
        self.logger.info("Task processor starting")

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info("Received shutdown signal", signal=signum)
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                await self.process_next_task()
                await asyncio.sleep(1)
        finally:
            self.logger.info("Task processor shutting down")

    async def process_next_task(self):
        """Process the next available task"""
        # Simulate fetching task from queue
        task = await self.fetch_task_from_queue()

        if task:
            await self.execute_task(task)

    async def fetch_task_from_queue(self):
        """Simulate fetching task from Redis/SQS/etc"""
        # In real implementation, this would fetch from your queue
        return {
            "id": "task-123",
            "type": "ai_content_generation",
            "data": {
                "user_id": "user-456",
                "prompt": "Generate summary for document",
                "document_id": "doc-789",
            },
            "context": {
                "app.user.id": "user-456",
                "app.session.id": "session-123",
                "aa.data.dataset.ids": "dataset-001",
            }
        }

    async def execute_task(self, task: Dict[str, Any]):
        """Execute a task with full observability"""
        task_id = task["id"]
        task_type = task["type"]

        # Restore context from task
        context = task.get("context", {})
        for key, value in context.items():
            set_baggage_item(key, value)

        with self.tracer.start_span(f"task_execution_{task_type}") as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.type", task_type)

            self.logger.info("Starting task execution",
                           task_id=task_id,
                           task_type=task_type)

            try:
                # Route to appropriate handler
                if task_type == "ai_content_generation":
                    result = await self.handle_ai_content_generation(task["data"])
                elif task_type == "data_processing":
                    result = await self.handle_data_processing(task["data"])
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                span.set_attribute("task.status", "completed")
                span.set_attribute("task.result_size", len(str(result)))

                self.logger.info("Task completed successfully",
                               task_id=task_id,
                               result_size=len(str(result)))

                # Save result
                await self.save_task_result(task_id, result)

            except Exception as e:
                span.set_attribute("task.status", "failed")
                span.set_attribute("task.error", str(e))

                self.logger.error("Task execution failed",
                                task_id=task_id,
                                error=str(e),
                                error_type=type(e).__name__)

                # Handle retry logic
                await self.handle_task_failure(task, e)

    async def handle_ai_content_generation(self, data: Dict[str, Any]):
        """Handle AI content generation task"""
        user_id = data["user_id"]
        prompt = data["prompt"]
        document_id = data.get("document_id")

        with self.tracer.start_span("ai_content_generation") as span:
            # Set GenAI semantic attributes
            set_gen_ai_span_attributes(
                operation_name=SpanAttributeValues.GenAiOperationName.GENERATE_CONTENT,
                agent_id=SpanAttributeValues.GenAiAgentId.QA_CHAT,
                model_name="llama-3.1-8B",
            )

            span.set_attribute("ai.prompt_length", len(prompt))
            span.set_attribute("ai.user_id", user_id)

            if document_id:
                span.set_attribute("ai.document_id", document_id)

            self.logger.info("Generating AI content",
                           user_id=user_id,
                           prompt_length=len(prompt),
                           document_id=document_id)

            # Simulate AI model call
            await asyncio.sleep(2)  # Simulate processing time

            response = f"AI generated content for: {prompt[:50]}..."

            span.set_attribute("ai.response_length", len(response))
            span.set_attribute("ai.tokens_used", len(response.split()))

            self.logger.info("AI content generated",
                           user_id=user_id,
                           response_length=len(response))

            return {
                "content": response,
                "metadata": {
                    "user_id": user_id,
                    "document_id": document_id,
                    "model": "llama-3.1-8B",
                    "tokens_used": len(response.split()),
                }
            }

    async def handle_data_processing(self, data: Dict[str, Any]):
        """Handle data processing task"""
        dataset_id = data["dataset_id"]
        operation = data["operation"]

        with self.tracer.start_span("data_processing") as span:
            span.set_attribute("data.dataset_id", dataset_id)
            span.set_attribute("data.operation", operation)

            self.logger.info("Processing dataset",
                           dataset_id=dataset_id,
                           operation=operation)

            # Simulate data processing
            await asyncio.sleep(1)

            result = {
                "dataset_id": dataset_id,
                "operation": operation,
                "records_processed": 1000,
                "status": "completed"
            }

            span.set_attribute("data.records_processed", result["records_processed"])

            self.logger.info("Data processing completed",
                           dataset_id=dataset_id,
                           records_processed=result["records_processed"])

            return result

    async def save_task_result(self, task_id: str, result: Any):
        """Save task result to storage"""
        self.logger.info("Saving task result",
                        task_id=task_id,
                        result_type=type(result).__name__)

        # Simulate saving to database/storage
        await asyncio.sleep(0.1)

    async def handle_task_failure(self, task: Dict[str, Any], error: Exception):
        """Handle task failure with retry logic"""
        task_id = task["id"]
        retry_count = task.get("retry_count", 0)
        max_retries = 3

        self.logger.warning("Task failed",
                          task_id=task_id,
                          retry_count=retry_count,
                          max_retries=max_retries,
                          error=str(error))

        if retry_count < max_retries:
            # Reschedule with exponential backoff
            delay = 2 ** retry_count
            task["retry_count"] = retry_count + 1

            self.logger.info("Rescheduling task",
                           task_id=task_id,
                           delay_seconds=delay)

            # In real implementation, reschedule the task
        else:
            self.logger.error("Task failed permanently",
                            task_id=task_id,
                            final_error=str(error))

async def main():
    processor = TaskProcessor()
    await processor.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Microservice Communication

Example showing context propagation between multiple services.

### Service A: API Gateway

```python
from fastapi import FastAPI, HTTPException
import httpx
from pharia_telemetry import (
    setup_basic_tracing,
    BaggageKeys,
    set_baggage_item,
)

app = FastAPI(title="API Gateway")

@app.on_event("startup")
async def startup():
    setup_basic_tracing("api-gateway", service_version="1.0.0")

@app.post("/api/orders")
async def create_order(order_data: dict, user_id: str = None):
    """Create order - demonstrates context flowing through multiple services"""
    # Set user context at gateway
    if user_id:
        set_baggage_item(BaggageKeys.USER_ID, user_id)
        set_baggage_item(BaggageKeys.USER_INTENT, "order_creation")

    async with httpx.AsyncClient() as client:
        # 1. Validate user (context flows automatically)
        user_response = await client.get(f"http://user-service/users/{user_id}")
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid user")

        # 2. Check inventory (context flows automatically)
        inventory_response = await client.post(
            "http://inventory-service/check",
            json={"items": order_data["items"]}
        )
        if inventory_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Insufficient inventory")

        # 3. Process payment (context flows automatically)
        payment_response = await client.post(
            "http://payment-service/charge",
            json={
                "amount": order_data["total"],
                "user_id": user_id,
            }
        )
        if payment_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Payment failed")

        # 4. Create order record (context flows automatically)
        order_response = await client.post(
            "http://order-service/orders",
            json=order_data
        )

        return order_response.json()
```

### Service B: Order Service

```python
from fastapi import FastAPI
from pharia_telemetry import (
    setup_basic_tracing,
    get_baggage_item,
    BaggageKeys,
)
import structlog

app = FastAPI(title="Order Service")
logger = structlog.get_logger(__name__)

@app.on_event("startup")
async def startup():
    setup_basic_tracing("order-service", service_version="1.0.0")

@app.post("/orders")
async def create_order(order_data: dict):
    """Create order with automatic context from gateway"""
    # Context automatically available from API Gateway
    user_id = get_baggage_item(BaggageKeys.USER_ID)
    user_intent = get_baggage_item(BaggageKeys.USER_INTENT)

    # Log with automatic context inclusion
    logger.info("Creating order",
               order_total=order_data.get("total"),
               item_count=len(order_data.get("items", [])))

    # Process order (all database operations include user context)
    order_id = await process_order(order_data, user_id)

    logger.info("Order created successfully", order_id=order_id)

    return {"order_id": order_id, "status": "created"}

async def process_order(order_data: dict, user_id: str):
    # Database operations automatically include user context in spans
    await asyncio.sleep(0.5)  # Simulate processing
    return f"order-{user_id}-{len(order_data.get('items', []))}"
```

## Next Steps

- **Configure for your environment** → [Configuration Guide](configuration.md)
- **Learn about context flow** → [Baggage & Context Guide](baggage-and-context.md)
- **Setup logging** → [Structured Logging Guide](structured-logging.md)
- **Having issues?** → [Troubleshooting Guide](troubleshooting.md)
