# Pharia Telemetry

[![Python](https://img.shields.io/pypi/pyversions/pharia-telemetry.svg)](https://pypi.org/project/pharia-telemetry/)
[![PyPI version](https://img.shields.io/pypi/v/pharia-telemetry.svg)](https://pypi.org/project/pharia-telemetry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/aleph-alpha/pharia-telemetry/workflows/CI/badge.svg)](https://github.com/aleph-alpha/pharia-telemetry/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-check%20latest%20PR-blue)

**A clean, minimal OpenTelemetry foundation library for Pharia services providing observability, tracing, and context propagation utilities.**

## 🎯 What is pharia-telemetry?

`pharia-telemetry` provides a **simple, focused foundation** for observability in Pharia services:
- **Context Propagation**: User and session context flows automatically across all service calls
- **Structured Logging**: Logs automatically include trace IDs and user context
- **OpenTelemetry Setup**: Minimal, high-level setup for distributed tracing
- **Standardized Constants**: Clean, namespaced constants for consistent telemetry

**Key Principle**: `pharia-telemetry` handles the foundation with minimal API surface, you add framework-specific auto-instrumentation.

## 📦 Installation

Requires Python 3.10+.

```bash
# Basic installation
pip install pharia-telemetry

# With structlog support (for structured logging)
pip install pharia-telemetry[structlog]
```

### Install from GitHub (pinned to commit)

For services that depend on a specific commit from the GitHub repo, use a direct VCS reference:

```bash
# HTTPS (recommended)
pip install "pharia-telemetry @ git+https://github.com/aleph-alpha/pharia-telemetry.git@<commit-sha>"

# SSH (if you have SSH keys configured)
pip install "pharia-telemetry @ git+ssh://git@github.com/aleph-alpha/pharia-telemetry.git@<commit-sha>"

# With optional extras
pip install "pharia-telemetry[structlog] @ git+https://github.com/aleph-alpha/pharia-telemetry.git@<commit-sha>"
```

In requirements files (PEP 508):

```
pharia-telemetry @ git+https://github.com/aleph-alpha/pharia-telemetry.git@<commit-sha>
pharia-telemetry[structlog] @ git+https://github.com/aleph-alpha/pharia-telemetry.git@<commit-sha>
```

## 🚀 30-Second Setup

```python
from pharia_telemetry import setup_telemetry, constants, set_baggage_item

# 1. One-line setup
setup_telemetry("my-service", service_version="1.0.0")

# 2. Set context that flows everywhere
set_baggage_item(constants.Baggage.USER_ID, "user-123")

# 3. Add framework instrumentation (optional)
# FastAPIInstrumentor.instrument_app(app)  # for FastAPI
# SQLAlchemyInstrumentor().instrument()    # for databases
```

**Result**: Your service now has distributed tracing with user context flowing through all operations!

## 🎯 Clean API Design

pharia-telemetry features a **clean, focused API** designed for ease of use:

```python
from pharia_telemetry import (
    # Core setup (essential)
    setup_telemetry,           # One-function setup

    # GenAI instrumentation (most users)
    create_chat_span,          # Smart sync/async chat spans
    create_embeddings_span,    # Smart sync/async embeddings spans
    create_tool_execution_span,# Smart sync/async tool spans
    set_genai_span_usage,      # Token usage tracking
    set_genai_span_response,   # Response metadata

    # Context propagation (advanced)
    set_baggage_item,          # Set context for propagation
    get_baggage_item,          # Get propagated context

    # Logging integration (optional)
    create_context_injector,   # Custom logging integration
)
```

## 📚 Documentation Guide

Choose your path based on what you need:

### 🆕 New to pharia-telemetry?
**Start here** → [**Getting Started Guide**](docs/getting-started.md)
- Basic setup and first examples
- Understanding the concepts
- Your first instrumented service

### 🔌 Want automatic instrumentation?
**Go to** → [**Auto-Instrumentation Guide**](docs/auto-instrumentation.md)
- Available instrumentation packages
- FastAPI, SQLAlchemy, HTTPX setup
- When auto-instrumentation works (and when it doesn't)

### 🛠️ Need manual control?
**See** → [**Manual Instrumentation Guide**](docs/manual-instrumentation.md)
- SSE streaming issues and solutions
- HTTP/2 compatibility problems
- Custom span management
- Performance optimization

### 🧳 Working with context propagation?
**Read** → [**Baggage & Context Guide**](docs/baggage-and-context.md)
- User and session context
- Cross-service correlation
- Standardized baggage keys
- Custom context patterns

### 📊 Setting up logging?
**Check** → [**Structured Logging Guide**](docs/structured-logging.md)
- Automatic trace correlation
- Log configuration patterns
- Integration with structlog

### 🤖 Building GenAI applications?
**Visit** → [**GenAI Spans Guide**](docs/genai-spans.md)
- OpenTelemetry semantic conventions for AI
- Automatic span attributes for models
- Token usage tracking
- Agent and tool instrumentation

### ⚙️ Need advanced configuration?
**Visit** → [**Configuration Guide**](docs/configuration.md)
- Environment variables
- OTLP exporter setup
- Custom resource attributes
- Production deployment

### 🏗️ Building integrations?
**Browse** → [**Integration Examples**](docs/integration-examples.md)
- Complete FastAPI service
- Microservice communication
- Background task processing
- Real-world patterns

### 🐛 Having issues?
**Try** → [**Troubleshooting Guide**](docs/troubleshooting.md)
- Common problems and solutions
- Debug techniques
- Performance considerations

## 🌟 Core Features

- **🔬 OpenTelemetry Integration**: Minimal setup utilities for distributed tracing
- **🧳 Baggage Management**: Context propagation across service boundaries
- **📊 Structured Logging**: Automatic trace correlation for log records
- **🤖 Smart GenAI Spans**: Auto-detecting sync/async convenience functions for AI operations
- **🔧 Production Ready**: Graceful degradation when OpenTelemetry is unavailable
- **📈 Pharia Standards**: Standardized constants and conventions across all services
- **🎯 Focused API**: Clean, intuitive functions for common use cases

## 🏛️ Architecture

```
┌─────────────────────────────────────────┐
│         Your Application + Auto         │
│           Instrumentation              │
├─────────────────────────────────────────┤
│        pharia-telemetry Foundation      │
│     (Propagators, Baggage, Logging)     │
├─────────────────────────────────────────┤
│           OpenTelemetry SDK             │
├─────────────────────────────────────────┤
│        OTLP Exporters & Backend        │
└─────────────────────────────────────────┘
```

## 🔍 Quick Examples

### Context Propagation
```python
from pharia_telemetry import constants, set_baggage_item

# Set once, flows everywhere
set_baggage_item(constants.Baggage.USER_ID, "user-123")
set_baggage_item(constants.Baggage.SESSION_ID, "session-456")
```

### Structured Logging
```python
import structlog
from pharia_telemetry import add_context_to_logs

# Easy integration with any logging framework
injector = add_context_to_logs("structlog")
structlog.configure(processors=[
    injector,  # Adds trace_id + baggage automatically
    structlog.processors.JSONRenderer(),
])
```

### GenAI Operations
```python
from pharia_telemetry import create_chat_span, create_embeddings_span
from pharia_telemetry.sem_conv.gen_ai import GenAI

# Smart convenience functions that auto-detect sync/async context
with create_chat_span(
    model="llama-3.1-8B",
    agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
    conversation_id="conv-123"
) as span:
    # Works in both sync and async contexts
    pass

# Also works seamlessly in async contexts
async with create_embeddings_span(model="text-embedding-3-small") as span:
    # Automatic context detection
    pass
```

### Clean Constants Structure
```python
from pharia_telemetry import constants

# Namespaced and organized
user_id = constants.Baggage.USER_ID                    # "app.user.id"
qa_chat = constants.Baggage.Values.UserIntent.QA_CHAT  # "pharia_qa_chat"

# GenAI constants in separate module
model = constants.GenAI.REQUEST_MODEL                  # "gen_ai.request.model"
chat_op = constants.GenAI.Values.OperationName.CHAT    # "chat"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📞 Support

- 📧 **Email**: conrad.poepke@aleph-alpha.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/aleph-alpha/pharia-telemetry/issues)
