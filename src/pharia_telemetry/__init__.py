"""
Pharia Telemetry - OpenTelemetry utilities for Pharia services.

This package provides a clean, consolidated API for OpenTelemetry tracing
with semantic conventions for GenAI operations.
"""

try:
    from pharia_telemetry._version import __version__
except ImportError:
    # Fallback for development installations
    __version__ = "dev"

# Constants available via direct import: from pharia_telemetry.constants import ...
# Baggage utilities
from pharia_telemetry.baggage import (
    get_baggage_item,
    set_baggage_item,
)
from pharia_telemetry.baggage.propagation import set_gen_ai_span_attributes

# Logging integration
from pharia_telemetry.logging import create_context_injector

# GenAI convenience functions
# Constants
from pharia_telemetry.sem_conv.gen_ai import (
    # Data structures
    DataContext,
    GenAI,
    create_agent_creation_span,
    create_agent_invocation_span,
    # Convenience functions for specific operations
    create_chat_span,
    create_embeddings_span,
    # Core GenAI functions
    create_genai_span,
    create_tool_execution_span,
    set_genai_span_response,
    set_genai_span_usage,
)

# Core setup function
from pharia_telemetry.setup import get_tracer, setup_telemetry

__author__ = "Aleph Alpha Engineering"
__email__ = "engineering@aleph-alpha.com"

# Clean, consolidated public API
__all__ = [
    # Setup
    "setup_telemetry",
    "get_tracer",
    # GenAI spans
    "create_genai_span",
    "set_genai_span_usage",
    "set_genai_span_response",
    "DataContext",
    # GenAI convenience functions
    "create_chat_span",
    "create_embeddings_span",
    "create_tool_execution_span",
    "create_agent_creation_span",
    "create_agent_invocation_span",
    # Constants
    "GenAI",
    # Baggage
    "set_baggage_item",
    "get_baggage_item",
    "set_gen_ai_span_attributes",
    # Logging
    "create_context_injector",
    # Version
    "__version__",
]
