"""Pharia Telemetry - Shared telemetry utilities and components."""

from importlib.metadata import version

# Constants (namespaced)
from pharia_telemetry import constants

# Mid-level utilities for advanced use cases
from pharia_telemetry.baggage import (
    get_baggage_item,
    set_baggage_item,
)
from pharia_telemetry.baggage.propagation import set_gen_ai_span_attributes

# High-level convenience API (primary interface for most users)
from pharia_telemetry.gen_ai import (
    add_context_to_logs,
    # Core GenAI functions
    create_genai_span,
    get_current_context,
    set_genai_span_response,
    set_genai_span_usage,
    setAgentCreationSpan,
    setAgentInvocationSpan,
    # Convenience functions for specific operations
    setChatSpan,
    setEmbeddingsSpan,
    # Alias
    setGenAiSpan,
    setToolExecutionSpan,
    setup_telemetry,
)
from pharia_telemetry.logging import create_context_injector

__version__ = version("pharia-telemetry")
__author__ = "Aleph Alpha Engineering"
__email__ = "engineering@aleph-alpha.com"

# Clean Public API - Comprehensive GenAI support
__all__ = [
    # Version info
    "__version__",
    # High-level convenience API (90% of users)
    "setup_telemetry",  # One-function setup for most services
    "add_context_to_logs",  # Easy logging integration
    "get_current_context",  # Manual context access
    # GenAI convenience functions - Primary API
    "create_genai_span",  # Create GenAI spans with semantic conventions
    "set_genai_span_usage",  # Set token usage on GenAI spans
    "set_genai_span_response",  # Set response attributes on GenAI spans
    # Operation-specific convenience functions (TypeScript parity)
    "setChatSpan",  # Chat operations
    "setToolExecutionSpan",  # Tool execution
    "setAgentCreationSpan",  # Agent creation
    "setEmbeddingsSpan",  # Embeddings operations
    "setAgentInvocationSpan",  # Agent invocation
    "setGenAiSpan",  # Alias for create_genai_span
    # Essential utilities (advanced users)
    "set_baggage_item",  # Set context for propagation
    "get_baggage_item",  # Get propagated context
    "create_context_injector",  # Custom logging integration
    "set_gen_ai_span_attributes",  # Low-level GenAI span attributes
    # Constants (namespaced)
    "constants",  # Access via constants.Baggage.USER_ID, etc.
]

# Advanced API available via explicit imports:
# - pharia_telemetry.baggage.* (span processors, advanced baggage functions)
# - pharia_telemetry.logging.* (individual injector classes, structlog processor)
# - pharia_telemetry.utils.* (low-level setup functions)
